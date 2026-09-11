"""BEV-150 Full-24 from the standard FCOS3D initialization."""

_base_ = './bev150.py'

# This is the same initialization used by the original BEVFormer-base
# training recipe. It is not a trained BEVFormer detector checkpoint.
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
resume_from = None

work_dir = 'work_dirs/bev150_fulltrain_24ep'

# Keep the original BEVFormer-base optimization recipe explicit so the
# experiment remains auditable even if an ancestor config later changes.
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Validation is performed separately for the predeclared checkpoint set.
# Disk capacity permits preserving every epoch for reproducibility.
evaluation = dict(interval=999)
checkpoint_config = dict(interval=1, max_keep_ckpts=24)
data = dict(workers_per_gpu=4)
