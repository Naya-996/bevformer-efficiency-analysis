"""Two-epoch exploratory fine-tune for the selected BEV-150 candidate.

This config is intentionally separate from the completed zero-shot B1/L1
experiment. Its results must not overwrite or be merged with B1.
"""

_base_ = './bev150.py'

load_from = 'ckpts/ablation/bevformer_r101_dcn_24ep_bev150_interp.pth'
resume_from = None
work_dir = 'work_dirs/bev150_finetune_2ep'

# Low learning rate: adapt the resized BEV and positional embeddings without
# treating this short run as a full target-resolution retraining schedule.
optimizer = dict(lr=2e-5)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
# The standalone pipeline performs one provenance-tracked full evaluation
# after epoch 2. Avoid duplicating the 6,019-sample validation inside training.
evaluation = dict(interval=999)

data = dict(workers_per_gpu=4)
