#!/usr/bin/env python3
"""GPU smoke test for continuous queries and cross-grid temporal migration."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from projects.mmdet3d_plugin.bevformer.modules.resolution_continuous import (
    ContinuousBEVQueryGenerator, timed_resize_prev_bev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/rc_bev/gpu_component_smoke.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA device must be visible')
    device = torch.device('cuda', 0)
    pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    generator = ContinuousBEVQueryGenerator(
        embed_dims=256, num_bands=16, hidden_dims=256,
        pc_range=pc_range).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    shape_records = []
    for shape in ((100, 100), (125, 125), (140, 140), (150, 150),
                  (160, 160), (175, 175), (180, 180), (200, 200), (96, 144)):
        query, position = generator(
            *shape, dtype=torch.float16, device=device, batch_size=1)
        shape_records.append({
            'shape': list(shape), 'query_shape': list(query.shape),
            'position_shape': list(position.shape),
            'finite': bool(torch.isfinite(query).all() and torch.isfinite(position).all()),
        })
    transitions = []
    history = torch.ones(100 * 100, 1, 256, device=device, dtype=torch.float16)
    source = (100, 100)
    for target in ((150, 150), (200, 200), (100, 100), (96, 144)):
        history, elapsed_ms = timed_resize_prev_bev(
            history, source, target, pc_range)
        transitions.append({
            'source': list(source), 'target': list(target),
            'output_shape': list(history.shape), 'elapsed_ms': elapsed_ms,
            'constant_max_error': float((history - 1).abs().max().item()),
        })
        source = target
    torch.cuda.synchronize(device)
    report = {
        'schema_version': 1,
        'status': 'SMOKE_PASS',
        'scope': 'components_only_not_accuracy_or_production_profile',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'device': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'query_shapes': shape_records,
        'transitions': transitions,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    if not all(item['finite'] for item in shape_records):
        raise RuntimeError('non-finite continuous query output')
    if any(item['constant_max_error'] != 0.0 for item in transitions):
        raise RuntimeError('constant temporal field changed during migration')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

