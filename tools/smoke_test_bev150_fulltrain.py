#!/usr/bin/env python3
"""Run one real BEVFormer training batch and checkpoint round trip."""

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, load_checkpoint, save_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def import_plugin(cfg):
    if cfg.get('plugin', False):
        module = os.path.dirname(cfg.plugin_dir).replace('/', '.')
        importlib.import_module(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', nargs='?',
        default='projects/configs/bevformer_ablation/bev150_fulltrain_24ep.py')
    parser.add_argument(
        '--checkpoint-out',
        default='work_dirs/bev150_fulltrain_24ep/smoke_checkpoint.pth')
    parser.add_argument(
        '--report-out',
        default='experiments/bev150_fulltrain_24ep/smoke_test.json')
    args = parser.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Smoke test must run with CUDA_VISIBLE_DEVICES=1')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('GPU1 must be the only visible CUDA device')

    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    set_random_seed(0, deterministic=False)
    cfg.seed = 0
    cfg.gpu_ids = [0]  # logical GPU0 maps to physical GPU1.

    init_path = Path(cfg.load_from)
    if init_path.name != 'r101_dcn_fcos3d_pretrain.pth' or cfg.resume_from:
        raise RuntimeError('non-standard or resumed initialization configured')
    source = torch.load(str(init_path), map_location='cpu', weights_only=False)
    source_state = source.get('state_dict', source)
    forbidden_source_keys = [
        key for key in source_state
        if any(token in key for token in ('bev_embedding', 'row_embed', 'col_embed'))
    ]
    if forbidden_source_keys:
        raise RuntimeError(
            'initialization contains BEVFormer BEV/position embeddings: '
            + ', '.join(forbidden_source_keys))

    dataset = build_dataset(cfg.data.train)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    checkpoint_info = load_checkpoint(
        model, str(init_path), map_location='cpu', strict=False)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)

    torch.cuda.reset_peak_memory_stats()
    batch = next(iter(loader))
    optimizer.zero_grad()
    outputs = model.train_step(batch, optimizer)
    loss = outputs['loss']
    if not torch.isfinite(loss).item():
        raise RuntimeError(f'non-finite total loss: {loss.item()}')
    loss.backward()

    finite_gradient_tensors = 0
    nonfinite_gradient_tensors = []
    for name, parameter in model.module.named_parameters():
        if parameter.grad is None:
            continue
        if torch.isfinite(parameter.grad).all().item():
            finite_gradient_tensors += 1
        else:
            nonfinite_gradient_tensors.append(name)
    if nonfinite_gradient_tensors:
        raise RuntimeError(
            'non-finite gradients: ' + ', '.join(nonfinite_gradient_tensors[:20]))
    optimizer.step()
    torch.cuda.synchronize()

    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model.module,
        str(checkpoint_path),
        optimizer=optimizer,
        meta={'experiment': 'bev150_fulltrain_24ep_smoke', 'seed': 0})

    reloaded = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(reloaded, str(checkpoint_path), map_location='cpu', strict=True)

    log_vars = {
        key: float(value) for key, value in outputs.get('log_vars', {}).items()
    }
    if not all(torch.isfinite(torch.tensor(value)).item()
               for value in log_vars.values()):
        raise RuntimeError('one or more logged losses are non-finite')

    report = {
        'status': 'passed',
        'physical_gpu': 1,
        'visible_device': torch.cuda.get_device_name(0),
        'seed': 0,
        'dataset_samples': len(dataset),
        'batch_size': 1,
        'queue_length': cfg.data.train.queue_length,
        'total_loss': float(loss.detach().cpu()),
        'log_vars': log_vars,
        'finite_gradient_tensors': finite_gradient_tensors,
        'nonfinite_gradient_tensors': [],
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'initialization_checkpoint': str(init_path.resolve()),
        'initialization_state_dict_keys': len(source_state),
        'initialization_contains_bev_embeddings': False,
        'checkpoint_metadata_keys': sorted(checkpoint_info.get('meta', {}).keys()),
        'smoke_checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_round_trip': True,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
