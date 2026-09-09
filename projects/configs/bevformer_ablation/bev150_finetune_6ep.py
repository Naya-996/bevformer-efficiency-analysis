"""Extend the BEV-150 adaptation from epoch 2 through epoch 6.

The resume checkpoint retains the first run's AdamW moments but has its
optimizer learning rates rebased from 2e-5 to 5e-6.  Accuracy is evaluated
separately for epochs 3--6 so the selected result is based on full nuScenes
validation metrics rather than training loss.
"""

_base_ = './bev150.py'

load_from = None
resume_from = 'ckpts/ablation/bev150_epoch2_resume_lr5e-6.pth'
work_dir = 'work_dirs/bev150_finetune_6ep'

optimizer = dict(lr=5e-6)
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    min_lr_ratio=0.1)

total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=6)
evaluation = dict(interval=999)

data = dict(workers_per_gpu=4)
