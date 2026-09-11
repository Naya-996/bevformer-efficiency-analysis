"""Resolution-continuous BEV query and temporal-state utilities.

This file intentionally depends only on PyTorch so its coordinate contracts can be
tested on CPU without importing the compiled BEVFormer operators.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shape2(value, name='shape'):
    if isinstance(value, int):
        value = (value, value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f'{name} must be an int or (height, width), got {value!r}')
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f'{name} values must be positive, got {(height, width)!r}')
    return height, width


def deterministic_resolution(step, resolutions, seed=0):
    """Return a reproducible resolution without mutable RNG state."""
    if not resolutions:
        raise ValueError('resolutions must not be empty')
    # SplitMix64 makes nearby step/seed values select unrelated list entries while
    # producing the same choice on every DDP rank with the same call count.
    value = (int(step) + int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return _shape2(resolutions[value % len(resolutions)], 'resolution')


class ContinuousBEVQueryGenerator(nn.Module):
    """Generate BEV content queries and positions from continuous coordinates.

    Args:
        embed_dims: Output channel count expected by BEVFormer.
        num_bands: Number of powers-of-two Fourier bands per coordinate.
        hidden_dims: Width of both coordinate MLPs.
        pc_range: Default physical extent [xmin, ymin, zmin, xmax, ymax, zmax].
    """

    def __init__(self, embed_dims=256, num_bands=16, hidden_dims=256,
                 pc_range=None):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_bands = int(num_bands)
        self.pc_range = tuple(pc_range) if pc_range is not None else None
        feature_dims = 4 * self.num_bands
        self.register_buffer(
            'frequencies', 2.0 ** torch.arange(self.num_bands, dtype=torch.float32),
            persistent=False)
        # Checkpointed so resumed training continues the deterministic schedule.
        self.register_buffer('schedule_step', torch.zeros((), dtype=torch.long))
        self.content = nn.Parameter(torch.zeros(1, self.embed_dims))
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_dims, hidden_dims), nn.GELU(),
            nn.Linear(hidden_dims, self.embed_dims))
        self.position_mlp = nn.Sequential(
            nn.Linear(feature_dims, hidden_dims), nn.GELU(),
            nn.Linear(hidden_dims, self.embed_dims))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.content, std=0.02)
        for module in (self.query_mlp, self.position_mlp):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def physical_grid(bev_h, bev_w, pc_range, device=None, dtype=torch.float32):
        """Return row-major cell centres as [H*W, 2] physical (x, y)."""
        bev_h, bev_w = _shape2((bev_h, bev_w), 'bev shape')
        if pc_range is None or len(pc_range) != 6:
            raise ValueError('pc_range must contain [xmin,ymin,zmin,xmax,ymax,zmax]')
        xmin, ymin, _, xmax, ymax, _ = [float(value) for value in pc_range]
        xs = torch.linspace(xmin, xmax, bev_w + 1, device=device, dtype=dtype)
        ys = torch.linspace(ymin, ymax, bev_h + 1, device=device, dtype=dtype)
        xs = (xs[:-1] + xs[1:]) * 0.5
        ys = (ys[:-1] + ys[1:]) * 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    def fourier_features(self, bev_h, bev_w, pc_range, device):
        coords = self.physical_grid(
            bev_h, bev_w, pc_range, device=device, dtype=torch.float32)
        xmin, ymin, _, xmax, ymax, _ = [float(value) for value in pc_range]
        u = 2.0 * (coords[:, 0] - xmin) / (xmax - xmin) - 1.0
        v = 2.0 * (coords[:, 1] - ymin) / (ymax - ymin) - 1.0
        angles_x = math.pi * u[:, None] * self.frequencies[None, :]
        angles_y = math.pi * v[:, None] * self.frequencies[None, :]
        return torch.cat(
            (angles_x.sin(), angles_x.cos(), angles_y.sin(), angles_y.cos()), dim=-1)

    def forward(self, bev_h, bev_w, pc_range=None, dtype=None, device=None,
                batch_size=1):
        bev_h, bev_w = _shape2((bev_h, bev_w), 'bev shape')
        pc_range = tuple(pc_range) if pc_range is not None else self.pc_range
        if pc_range is None:
            raise ValueError('pc_range is required')
        device = torch.device(device) if device is not None else self.content.device
        dtype = dtype or self.content.dtype
        if device != self.content.device:
            raise ValueError(
                f'generator parameters are on {self.content.device}, requested {device}; '
                'move the module before calling it')
        features = self.fourier_features(bev_h, bev_w, pc_range, device)
        # Linear kernels use their parameter dtype. Cast only the public outputs so
        # CPU float16 requests do not depend on half-precision GEMM availability.
        features = features.to(dtype=self.content.dtype)
        query = self.content + self.query_mlp(features)
        position = self.position_mlp(features)
        query = query.to(dtype=dtype)
        position = position.to(dtype=dtype)
        position = position.reshape(bev_h, bev_w, self.embed_dims)
        position = position.permute(2, 0, 1).unsqueeze(0)
        position = position.expand(int(batch_size), -1, -1, -1).contiguous()
        return query, position


def resize_prev_bev(prev_bev, source_shape, target_shape, source_pc_range,
                    target_pc_range=None, mode='bilinear'):
    """Resample `[HW,B,C]` or `[B,HW,C]` history in physical BEV coordinates.

    The returned layout is always `[target_H*target_W, B, C]`, matching the
    internal transformer contract. A same-grid migration returns the sequence-first
    input object unchanged and otherwise avoids any silent square-root inference.
    """
    if prev_bev is None:
        return None
    source_h, source_w = _shape2(source_shape, 'source_shape')
    target_h, target_w = _shape2(target_shape, 'target_shape')
    source_pc_range = tuple(float(value) for value in source_pc_range)
    target_pc_range = tuple(float(value) for value in (
        target_pc_range if target_pc_range is not None else source_pc_range))
    source_length = source_h * source_w
    if prev_bev.ndim != 3:
        raise ValueError(f'prev_bev must have 3 dimensions, got {prev_bev.shape}')
    if prev_bev.shape[0] == source_length:
        sequence_first = prev_bev
    elif prev_bev.shape[1] == source_length:
        sequence_first = prev_bev.permute(1, 0, 2)
    else:
        raise ValueError(
            f'prev_bev length does not match explicit source shape {source_shape}: '
            f'{tuple(prev_bev.shape)}')
    if (source_h, source_w) == (target_h, target_w) \
            and source_pc_range == target_pc_range:
        return sequence_first

    _, batch, channels = sequence_first.shape
    source = sequence_first.reshape(source_h, source_w, batch, channels)
    source = source.permute(2, 3, 0, 1)
    xmin_s, ymin_s, _, xmax_s, ymax_s, _ = source_pc_range
    xmin_t, ymin_t, _, xmax_t, ymax_t, _ = target_pc_range
    target_xy = ContinuousBEVQueryGenerator.physical_grid(
        target_h, target_w, target_pc_range,
        device=prev_bev.device, dtype=torch.float32)
    grid_x = 2.0 * (target_xy[:, 0] - xmin_s) / (xmax_s - xmin_s) - 1.0
    grid_y = 2.0 * (target_xy[:, 1] - ymin_s) / (ymax_s - ymin_s) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, target_h, target_w, 2)
    grid = grid.expand(batch, -1, -1, -1)
    compute = source
    restore_dtype = source.dtype
    if source.device.type == 'cpu' and source.dtype in (torch.float16, torch.bfloat16):
        compute = source.float()
    resized = F.grid_sample(
        compute, grid.to(dtype=compute.dtype), mode=mode,
        padding_mode='border', align_corners=False)
    resized = resized.to(dtype=restore_dtype)
    return resized.permute(2, 3, 0, 1).reshape(target_h * target_w, batch, channels)


def timed_resize_prev_bev(*args, **kwargs):
    """Return `(resized, elapsed_ms)` with CUDA synchronization when applicable."""
    tensor = args[0] if args else kwargs.get('prev_bev')
    if tensor is not None and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    start = time.perf_counter()
    output = resize_prev_bev(*args, **kwargs)
    if tensor is not None and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return output, (time.perf_counter() - start) * 1000.0
