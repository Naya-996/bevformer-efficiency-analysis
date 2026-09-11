from pathlib import Path

from mmcv import Config
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401 - register custom modules
from projects.mmdet3d_plugin.bevformer.apis.mmdet_train import (
    _install_mmcv_ddp_sync_compat,
)


ROOT = Path(__file__).resolve().parents[1]


def config(relative):
    return Config.fromfile(str(ROOT / relative))


def test_official_base_remains_fixed_and_default_off():
    cfg = config('projects/configs/bevformer/bevformer_base.py')
    head = cfg.model.pts_bbox_head
    assert (head.bev_h, head.bev_w) == (200, 200)
    assert 'continuous_bev_query' not in head
    assert 'train_bev_shapes' not in head
    assert 'budget_controller' not in cfg.model


def test_multires_contract_and_single_shape_per_batch():
    cfg = config('projects/configs/bevformer_rc/rc_bev_multires.py')
    head = cfg.model.pts_bbox_head
    assert head.continuous_bev_query is not None
    assert head.train_bev_shapes == [100, 125, 150, 175, 200]
    assert head.resolution_seed == 0
    assert cfg.custom_hooks[0].type == 'ResolutionTraceHook'


def test_fixed150_keeps_official_table_shape_for_checkpoint_loading():
    cfg = config('projects/configs/bevformer_rc/rc_bev_fixed150.py')
    head = cfg.model.pts_bbox_head
    assert (head.bev_h, head.bev_w) == (200, 200)
    assert head.train_bev_shapes == [150]
    assert head.test_bev_shape == (150, 150)


def test_continuous_model_freezes_bypassed_legacy_tables_for_ddp():
    cfg = config('projects/configs/bevformer_rc/rc_bev_multires.py')
    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    head = model.pts_bbox_head
    assert not head.bev_embedding.weight.requires_grad
    assert all(not parameter.requires_grad
               for parameter in head.positional_encoding.parameters())
    assert all(parameter.requires_grad
               for parameter in head.continuous_query_generator.parameters())


def test_recent_pytorch_ddp_compat_respects_buffer_broadcast_setting():
    class FakeDDP:
        def __init__(self, broadcast_buffers):
            self.broadcast_buffers = broadcast_buffers
            self.sync_calls = 0

        def _sync_buffers(self):
            self.sync_calls += 1

    without_broadcast = _install_mmcv_ddp_sync_compat(FakeDDP(False))
    without_broadcast._sync_params()
    assert without_broadcast.sync_calls == 0

    with_broadcast = _install_mmcv_ddp_sync_compat(FakeDDP(True))
    with_broadcast._sync_params()
    assert with_broadcast.sync_calls == 1


def test_seen_unseen_and_dynamic_protocols_are_disjoint():
    seen = config('projects/configs/bevformer_rc/rc_bev_eval_seen.py')
    unseen = config('projects/configs/bevformer_rc/rc_bev_eval_unseen.py')
    dynamic = config('projects/configs/bevformer_rc/rc_bev_dynamic.py')
    assert seen.evaluation_resolutions == [100, 125, 150, 175, 200]
    assert unseen.evaluation_resolutions == [140, 160, 180]
    assert not set(seen.evaluation_resolutions) & set(unseen.evaluation_resolutions)
    assert dynamic.model.budget_controller.resolutions == (100, 150, 200)
    assert dynamic.model.budget_controller.down_threshold \
        < dynamic.model.budget_controller.up_threshold
