import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = load_module(
    'resolution_continuous_standalone',
    'projects/mmdet3d_plugin/bevformer/modules/resolution_continuous.py')
controller_module = load_module(
    'budget_controller_standalone',
    'projects/mmdet3d_plugin/bevformer/modules/budget_controller.py')


PC_RANGE = (-50.0, -25.0, -5.0, 50.0, 75.0, 3.0)


@pytest.mark.parametrize('shape', [
    (100, 100), (125, 125), (140, 140), (150, 150),
    (160, 160), (175, 175), (180, 180), (200, 200), (7, 11)])
def test_continuous_query_shape_dtype_and_finite(shape):
    generator = rc.ContinuousBEVQueryGenerator(
        embed_dims=16, num_bands=3, hidden_dims=24, pc_range=PC_RANGE)
    query, position = generator(
        *shape, dtype=torch.float16, device='cpu', batch_size=2)
    assert query.shape == (shape[0] * shape[1], 16)
    assert position.shape == (2, 16, shape[0], shape[1])
    assert query.dtype == torch.float16
    assert position.dtype == torch.float16
    assert query.device.type == 'cpu'
    assert torch.isfinite(query).all()
    assert torch.isfinite(position).all()


def test_continuous_query_backward_reaches_all_parameters():
    generator = rc.ContinuousBEVQueryGenerator(
        embed_dims=8, num_bands=2, hidden_dims=12, pc_range=PC_RANGE)
    query, position = generator(4, 5, batch_size=1)
    (query.square().mean() + position.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in generator.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in generator.parameters())


def test_cell_centres_and_row_major_flatten_order():
    grid = rc.ContinuousBEVQueryGenerator.physical_grid(2, 4, PC_RANGE)
    expected = torch.tensor([
        [-37.5, 0.0], [-12.5, 0.0], [12.5, 0.0], [37.5, 0.0],
        [-37.5, 50.0], [-12.5, 50.0], [12.5, 50.0], [37.5, 50.0],
    ])
    torch.testing.assert_close(grid, expected)


def test_deterministic_resolution_schedule():
    sizes = [100, 125, 150, 175, 200]
    first = [rc.deterministic_resolution(i, sizes, seed=7) for i in range(30)]
    second = [rc.deterministic_resolution(i, sizes, seed=7) for i in range(30)]
    third = [rc.deterministic_resolution(i, sizes, seed=8) for i in range(30)]
    assert first == second
    assert first != third
    assert set(first).issubset({(value, value) for value in sizes})


def make_field(shape, channels=1):
    grid = rc.ContinuousBEVQueryGenerator.physical_grid(*shape, PC_RANGE)
    values = grid[:, :channels]
    return values.reshape(shape[0] * shape[1], 1, channels)


def test_same_shape_sequence_first_is_exact_identity():
    field = torch.randn(35, 2, 4)
    migrated = rc.resize_prev_bev(field, (5, 7), (5, 7), PC_RANGE)
    assert migrated is field


def test_same_shape_batch_first_becomes_sequence_first_without_resampling():
    field = torch.randn(2, 35, 4)
    migrated = rc.resize_prev_bev(field, (5, 7), (5, 7), PC_RANGE)
    torch.testing.assert_close(migrated, field.permute(1, 0, 2))


@pytest.mark.parametrize('source,target', [
    ((100, 100), (150, 150)), ((150, 150), (200, 200)),
    ((200, 200), (100, 100)), ((9, 13), (7, 17))])
def test_cross_shape_constant_field_stays_constant(source, target):
    field = torch.full((source[0] * source[1], 2, 3), 4.25)
    migrated = rc.resize_prev_bev(field, source, target, PC_RANGE)
    assert migrated.shape == (target[0] * target[1], 2, 3)
    torch.testing.assert_close(migrated, torch.full_like(migrated, 4.25))


def test_coordinate_field_is_resampled_in_physical_space():
    source = (21, 31)
    target = (15, 19)
    source_xy = rc.ContinuousBEVQueryGenerator.physical_grid(*source, PC_RANGE)
    field = source_xy.reshape(source[0] * source[1], 1, 2)
    migrated = rc.resize_prev_bev(field, source, target, PC_RANGE)
    expected = rc.ContinuousBEVQueryGenerator.physical_grid(*target, PC_RANGE)
    # A linear coordinate field is reproduced by bilinear interpolation away from
    # the half-cell border, and target centres lie within that region here.
    torch.testing.assert_close(migrated[:, 0], expected, atol=1e-4, rtol=1e-5)


def test_missing_history_and_invalid_shape_handling():
    assert rc.resize_prev_bev(None, (10, 10), (20, 20), PC_RANGE) is None
    with pytest.raises(ValueError, match='explicit source shape'):
        # The error text comes from the size contract, never sqrt inference.
        rc.resize_prev_bev(torch.randn(99, 1, 2), (10, 10), (20, 20), PC_RANGE)


def test_controller_hysteresis_residency_and_scene_reset():
    controller = controller_module.BudgetAdaptiveResolutionController(
        min_residency=2, up_threshold=0.6, down_threshold=0.3)
    low = {'low_confidence_ratio': 0.0}
    high = {'low_confidence_ratio': 1.0, 'translation_m': 3.0, 'yaw_deg': 20.0}
    assert controller.select(low) == 200
    assert controller.select(low) == 200
    assert controller.select(low) == 150
    assert controller.select(low) == 150
    assert controller.select(low) == 100
    assert controller.select(high) == 100
    assert controller.select(high) == 150
    assert controller.select(high, scene_changed=True) == 200
    assert controller.last_record['reason'] == 'scene_reset'
    assert controller.last_record['controller_ms'] >= 0.0
