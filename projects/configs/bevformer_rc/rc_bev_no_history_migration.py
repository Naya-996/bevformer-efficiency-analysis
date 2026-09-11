_base_ = ['./rc_bev_multires.py']

# Ablation: discard temporal memory whenever H/W changes instead of resampling it.
model = dict(pts_bbox_head=dict(prev_bev_migration='reset_on_change'))
work_dir = 'work_dirs/rc_bev_no_history_migration_seed0'

