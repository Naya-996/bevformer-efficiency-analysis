"""BEVFormer-base with a 150 x 150 BEV grid."""

_base_ = '../bevformer/bevformer_base.py'

bev_h_ = 150
bev_w_ = 150

model = dict(
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        transformer=dict(rotate_center=[bev_w_ / 2, bev_h_ / 2]),
        positional_encoding=dict(
            row_num_embed=bev_h_,
            col_num_embed=bev_w_)))

data = dict(
    train=dict(bev_size=(bev_h_, bev_w_)),
    val=dict(bev_size=(bev_h_, bev_w_)),
    test=dict(bev_size=(bev_h_, bev_w_)))
