#!/usr/bin/env python3
"""Audit the resolved BEV-150 Full-24 config and its nuScenes annotations."""

import argparse
import json
import os
from pathlib import Path

import mmcv
from mmcv import Config


CAMERAS = {
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def annotation_summary(path):
    payload = mmcv.load(path)
    infos = payload['infos'] if isinstance(payload, dict) else payload
    require(infos, f'annotation contains no samples: {path}')
    camera_sets = {frozenset(info.get('cams', {})) for info in infos}
    require(camera_sets == {frozenset(CAMERAS)},
            f'not every sample has exactly the expected six cameras: {path}')
    return {
        'path': str(Path(path).resolve()),
        'samples': len(infos),
        'cameras': sorted(CAMERAS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', nargs='?',
        default='projects/configs/bevformer_ablation/bev150_fulltrain_24ep.py')
    parser.add_argument('--output')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    base = Config.fromfile('projects/configs/bevformer/bevformer_base.py')
    head = cfg.model.pts_bbox_head
    transformer = head.transformer

    require((head.bev_h, head.bev_w) == (150, 150), 'BEV grid is not 150x150')
    require(head.bev_h * head.bev_w == 22500, 'BEV query count is not 22,500')
    require(transformer.embed_dims == 256, 'embedding dim changed')
    require(transformer.encoder.num_layers == 6, 'encoder depth changed')
    require(transformer.decoder.num_layers == 6, 'decoder depth changed')
    require(cfg.model.img_neck.num_outs == 4, 'FPN level count changed')
    require(cfg.model.img_backbone.depth == 101, 'backbone is not R101')
    require(cfg.model.img_backbone.dcn.type == 'DCNv2', 'DCNv2 is disabled')
    require(transformer.encoder.transformerlayers.attn_cfgs[0].type ==
            'TemporalSelfAttention', 'temporal self-attention is disabled')
    require(cfg.model.video_test_mode is True, 'video_test_mode is disabled')
    require(list(transformer.rotate_center) == [75.0, 75.0],
            'rotate_center is inconsistent with BEV-150')
    require(head.positional_encoding.row_num_embed == 150, 'row embedding changed')
    require(head.positional_encoding.col_num_embed == 150, 'column embedding changed')

    for split in ('train', 'val', 'test'):
        require(tuple(cfg.data[split].bev_size) == (150, 150),
                f'{split} BEV size changed')
        require(cfg.data[split].ann_file == base.data[split].ann_file,
                f'{split} annotations differ from Base')
        require(cfg.data[split].pipeline == base.data[split].pipeline,
                f'{split} image pipeline differs from Base')
        require(cfg.data[split].modality == base.data[split].modality,
                f'{split} modality differs from Base')
        require(cfg.data[split].modality.use_camera is True,
                f'{split} camera input is disabled')
    require(cfg.data.train.queue_length == base.data.train.queue_length == 4,
            'training queue length differs from Base')

    require(cfg.optimizer.type == 'AdamW', 'optimizer is not AdamW')
    require(cfg.optimizer.lr == base.optimizer.lr == 2e-4, 'base LR changed')
    require(cfg.optimizer.paramwise_cfg.custom_keys.img_backbone.lr_mult == 0.1,
            'backbone LR multiplier changed')
    require(cfg.optimizer.weight_decay == 0.01, 'weight decay changed')
    require(cfg.optimizer_config == base.optimizer_config, 'gradient clipping changed')
    require(cfg.lr_config == base.lr_config, 'learning-rate schedule differs from Base')
    require(cfg.total_epochs == cfg.runner.max_epochs == 24, 'run is not 24 epochs')
    require(cfg.load_from == 'ckpts/r101_dcn_fcos3d_pretrain.pth',
            'initialization is not the required FCOS3D checkpoint')
    require(cfg.resume_from is None, 'resume_from must be None')
    forbidden = ('bevformer_r101_dcn_24ep', 'bev150_interp', 'finetune')
    require(not any(token in cfg.load_from.lower() for token in forbidden),
            'a forbidden trained BEVFormer checkpoint is configured')
    require(cfg.checkpoint_config.interval == 1, 'checkpoints are not saved every epoch')
    require(cfg.checkpoint_config.max_keep_ckpts >= 24,
            'the config does not preserve all 24 checkpoints')

    train_ann = annotation_summary(cfg.data.train.ann_file)
    val_ann = annotation_summary(cfg.data.val.ann_file)
    require(val_ann['samples'] == 6019, 'validation split is not the full 6,019 samples')

    report = {
        'status': 'passed',
        'config': str(Path(args.config).resolve()),
        'seed': 0,
        'model': {
            'bev_grid': [150, 150],
            'bev_queries': 22500,
            'embedding_dim': 256,
            'encoder_layers': 6,
            'decoder_layers': 6,
            'fpn_levels': 4,
            'backbone': 'ResNet-101 + DCNv2',
            'temporal_history': True,
            'video_test_mode': True,
            'rotate_center': [75.0, 75.0],
        },
        'data': {'train': train_ann, 'validation': val_ann, 'queue_length': 4},
        'training': {
            'optimizer': 'AdamW', 'base_lr': 2e-4,
            'backbone_lr_multiplier': 0.1, 'weight_decay': 0.01,
            'epochs': 24, 'scheduler': 'CosineAnnealing', 'warmup_iters': 500,
        },
        'initialization': cfg.load_from,
        'resume_from': cfg.resume_from,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
