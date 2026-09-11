# BEV-150 Full-24

This experiment trains the original BEVFormer-base architecture for 24 epochs
with the BEV grid changed from 200 x 200 to 150 x 150. All other model, data,
loss, optimizer, and scheduler settings follow the Base recipe.

Initialization is the standard `r101_dcn_fcos3d_pretrain.pth` checkpoint used
by Base. No trained BEVFormer detector checkpoint or interpolated BEV embedding
is loaded, so this run is distinct from the 2-epoch and 6-epoch adaptation
experiments.

Training uses seed 0 and physical GPU1 only. Every epoch checkpoint is retained.
Epochs 6, 12, 18, and 24 will be evaluated on all 6,019 nuScenes validation
samples. The predeclared selection rule is maximum NDS, then maximum mAP.

Accuracy and profiling fields remain empty until the corresponding measurements
finish successfully.
