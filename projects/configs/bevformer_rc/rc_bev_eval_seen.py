_base_ = ['./rc_bev_multires.py']

# run_rc_bev_experiments.sh iterates this declaration using --cfg-options.
evaluation_resolutions = [100, 125, 150, 175, 200]
model = dict(pts_bbox_head=dict(train_bev_shapes=[], test_bev_shape=(150, 150)))

