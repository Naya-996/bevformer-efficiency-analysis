_base_ = ['./rc_bev_multires.py']

model = dict(
    budget_controller=dict(
        resolutions=(100, 150, 200),
        safe_resolution=200,
        up_threshold=0.65,
        down_threshold=0.35,
        min_residency=3),
    pts_bbox_head=dict(train_bev_shapes=[], test_bev_shape=(200, 200)))

work_dir = 'work_dirs/rc_bev_dynamic'

