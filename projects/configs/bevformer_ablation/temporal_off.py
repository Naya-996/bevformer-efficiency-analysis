"""BEVFormer-base inference without the previous-frame BEV."""

_base_ = '../bevformer/bevformer_base.py'

# The detector clears prev_bev before every frame when this flag is disabled.
model = dict(video_test_mode=False)
