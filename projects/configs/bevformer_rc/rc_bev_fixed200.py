"""Fixed-200 continuous-query control for the multi-resolution ablation."""
_base_ = ['./rc_bev_fixed150.py']
model = dict(pts_bbox_head=dict(train_bev_shapes=[200], test_bev_shape=(200, 200)))
data = dict(train=dict(bev_size=(200, 200)), val=dict(bev_size=(200, 200)),
            test=dict(bev_size=(200, 200)))
work_dir = 'work_dirs/rc_bev_fixed200_seed0'
