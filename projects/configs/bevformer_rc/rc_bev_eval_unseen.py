_base_ = ['./rc_bev_multires.py']

evaluation_resolutions = [140, 160, 180]
model = dict(pts_bbox_head=dict(train_bev_shapes=[], test_bev_shape=(140, 140)))

