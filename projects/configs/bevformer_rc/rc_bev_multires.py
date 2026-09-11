_base_ = ['./rc_bev_fixed150.py']

# One resolution is selected for the entire temporal queue and batch. The schedule
# is a deterministic function of (iteration, seed), so DDP ranks agree.
train_resolutions = [100, 125, 150, 175, 200]
model = dict(
    pts_bbox_head=dict(
        train_bev_shapes=train_resolutions,
        test_bev_shape=(150, 150),
        resolution_seed=0))

custom_hooks = [
    dict(type='ResolutionTraceHook', filename='resolution_trace.jsonl', interval=1)
]
work_dir = 'work_dirs/rc_bev_multires_seed0'

