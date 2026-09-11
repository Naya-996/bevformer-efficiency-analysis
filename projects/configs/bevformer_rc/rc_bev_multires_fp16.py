_base_ = ['./rc_bev_multires.py']

fp16 = dict(loss_scale=512.0)
work_dir = 'work_dirs/rc_bev_multires_fp16_seed0'

