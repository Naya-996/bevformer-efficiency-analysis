"""BEVFormer-base with three encoder layers."""

_base_ = '../bevformer/bevformer_base.py'

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(num_layers=3))))
