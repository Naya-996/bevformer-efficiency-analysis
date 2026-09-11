_base_ = ['../bevformer/bevformer_base.py']

# Method prototype. All historical configs remain untouched. The official learned
# tables stay in the module/checkpoint but are bypassed only when this block exists.
model = dict(
    pts_bbox_head=dict(
        # Keep the official 200x200 table shapes so its checkpoint loads without
        # size mismatches. The continuous path uses the runtime shapes below.
        bev_h=200,
        bev_w=200,
        continuous_bev_query=dict(num_bands=16, hidden_dims=256),
        train_bev_shapes=[150],
        test_bev_shape=(150, 150),
        resolution_seed=0))

data = dict(
    train=dict(bev_size=(150, 150)),
    val=dict(bev_size=(150, 150)),
    test=dict(bev_size=(150, 150)))

work_dir = 'work_dirs/rc_bev_fixed150_seed0'
