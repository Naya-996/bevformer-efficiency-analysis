# BEVFormer Efficiency and Temporal Experiments

This document is the source of truth for the BEVFormer lightweight and
temporal ablations. Accuracy values come from the nuScenes evaluation toolkit;
efficiency values come from `tools/profile_bevformer.py` under one shared
measurement protocol.

## Measurement protocol

- Dataset: nuScenes v1.0-trainval, validation split (6,019 samples)
- GPU: NVIDIA GeForce RTX 5090. The original ablation matrix was measured on
  GPU0; the final Base/L2/L3 comparison was measured back-to-back on GPU1.
- Software: PyTorch 2.7.1+cu128, CUDA 12.8, MMCV 1.4.0
- Input: six cameras at 1600 x 900, padded internally to a multiple of 32
- Batch size: 1
- Latency scope: MMDataParallel scatter, model forward, and detector
  post-processing; DataLoader fetch/decode/collation is excluded
- Timing: 10 warmup iterations followed by 100 measured iterations, with
  `torch.cuda.synchronize()` before and after every measured forward
- Memory: per-process PyTorch CUDA peak allocated and peak reserved memory
- Parameters: all model parameters, including frozen parameters

The reported FPS is `1 / mean latency`. Accuracy and profiling runs preserve
the same input pipeline, dataset, checkpoint family, and GPU unless a row says
otherwise. Final profiles were collected with no competing compute workload
on the measured GPU.

## Experiment matrix

| ID | Variant | BEV | Queries | Encoder | Temporal history | NDS | mAP | mATE | mAOE | mAVE | Peak alloc. (MiB) | Peak reserved (MiB) | Latency avg (ms) | P95 (ms) | FPS | Params (M) | Status |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | Official base | 200x200 | 40,000 | 6 | ON | 0.5173 | 0.4163 | 0.6723 | 0.3712 | 0.3938 | 2184.634 | 3118 | 202.2956 | 202.8613 | 4.94326 | 69.034947 | Complete |
| B1 / L1 | BEV-150 (interpolated zero-shot) | 150x150 | 22,500 | 6 | ON | 0.4917 | 0.3885 | 0.7202 | 0.3833 | 0.4467 | 2150.081 | 3100 | 180.8861 | 181.5634 | 5.52834 | 64.542147 | Complete / selected zero-shot Lite |
| B2 | BEV-100 (interpolated zero-shot) | 100x100 | 10,000 | 6 | ON | 0.3887 | 0.2608 | 0.8989 | 0.4321 | 0.5588 | 2124.513 | 3078 | 173.5370 | 178.4354 | 5.76246 | 61.329347 | Complete |
| E1 | Encoder-4 early exit | 200x200 | 40,000 | 4 | ON | 0.0176 | 0.0000 | 1.0292 | 1.0709 | 0.9951 | 2178.351 | 3112 | 183.6681 | 184.4972 | 5.44460 | 67.387971 | Complete / invalid as untrained Lite |
| E2 | Encoder-3 early exit | 200x200 | 40,000 | 3 | ON | 0.0178 | 0.0000 | 1.0484 | 1.0609 | 1.1115 | 2175.209 | 3108 | 174.6946 | 175.9951 | 5.72428 | 66.564483 | Complete / invalid as untrained Lite |
| T1 | Temporal OFF (TSA retained) | 200x200 | 40,000 | 6 | OFF | 0.4128 | 0.3461 | 0.7553 | 0.4530 | 0.8918 | 2145.571 | 2574 | 202.6611 | 203.2064 | 4.93435 | 69.034947 | Complete |

All six full validation evaluations and final profiles are complete. B1 and B2
are interpolated zero-shot variants, with no fine-tuning or retraining. T1
disables recurrent history but does not remove the Temporal Self-Attention
(TSA) module or its computation.

<!-- BEGIN L2_STATUS -->
L2 (BEV-150, two-epoch low-learning-rate fine-tune) is
complete: NDS 0.5097, mAP 0.4074, FPS 5.487, and mean latency
182.24 ms. It recovers +0.0180 NDS and
+0.0189 mAP over zero-shot B1. See
`experiments/lite_finetuned/result_report.md` for the strict provenance and
comparison; it remains a short adaptation rather than a full retraining.
<!-- END L2_STATUS -->

## Final trained lightweight model (L3)

L2 was extended from epoch 2 to epoch 6 while preserving AdamW moments and
rebasing the peak learning rate from `2e-5` to `5e-6`. Epochs 3--6 were each
evaluated on all 6,019 validation samples; epoch 6 had the highest NDS.

| Variant | NDS | mAP | Mean / P95 latency (ms) | FPS | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 Base | 0.5173 | 0.4163 | 253.12 / 253.61 | 3.951 | 69.034947 |
| L2 BEV-150, 2 ep | 0.5097 | 0.4074 | 231.20 / 231.90 | 4.325 | 64.542147 |
| **L3 BEV-150, 6 ep** | **0.5123** | **0.4082** | **231.33 / 231.77** | **4.323** | **64.542147** |

These three profiles use physical GPU1, 20 warmup iterations, and 200 measured
iterations. L3 improves NDS by 0.0026 over L2 while remaining effectively
identical in runtime. Relative to Base it reduces mean latency by 8.61%, raises
FPS by 9.42%, and reduces parameters by 6.51%, with an NDS gap of 0.0051.
Detailed epoch metrics, class deltas, hashes, and paired profiles are in
`experiments/bev150_extended/result_report.md`.

## B0: official base

- Config: `projects/configs/bevformer/bevformer_base.py`
- Config SHA256: `45905ac50a8afa841f0be43c81152277dd694de62054c3595dfd56efd33d6324`
- Checkpoint: `ckpts/bevformer_r101_dcn_24ep.pth`, epoch 24
- Checkpoint SHA256: `5fda3ed3e0d2d1dc09baca337aa1ed64aece057fed7d8238525c5ed342649177`
- Backbone: ResNet-101 with DCNv2 and FPN
- BEV grid: 200 x 200 over `[-51.2, 51.2]` metres in X/Y
- Transformer: six encoder layers and six decoder layers
- Temporal mode: recurrent previous BEV enabled at inference
- Full result: `test/bevformer_base/Mon_Aug_10_17_02_09_2026/pts_bbox/`
- Profile: `experiments/baseline/profile.json`

Evaluation reproduced the published BEVFormer base result: mAP 41.63 and NDS
51.73. The comparison table reports both baseline peak allocated CUDA memory
(2,184.63 MiB) and peak reserved CUDA memory (3,118 MiB).

## Selected Lite candidate

BEV-150 was first selected as the zero-shot Lite candidate. Relative to B0 it reduces
the BEV query count by 43.75% and parameter count by 6.51%. On the shared RTX
5090 protocol it improves FPS by 11.84% and reduces mean latency by 10.58%,
while absolute NDS and mAP decrease by 0.0256 and 0.0278 respectively. Peak
allocated memory changes by only -1.58%, showing that BEV query resolution is
not the dominant source of allocated inference memory in this implementation.

BEV-100 is faster but loses 0.1287 NDS and 0.1555 mAP, so it lies outside the
target accuracy/efficiency balance. Encoder early exit is not a valid drop-in
compression method: removing trained upper layers without optimization makes
mAP collapse to zero. Temporal OFF loses 0.1046 NDS and 0.0477 mAVE while
providing no throughput benefit, so temporal history is retained in L1.

The zero-shot L1 numbers remain an interpolation robustness experiment. L3 is
the separately trained lightweight checkpoint used for the final claim.

## Visual comparison

Four validation keyframes were selected from different scenes using only
ground-truth metadata: maximum 30 m object density, maximum ego yaw change,
maximum low-visibility object count, and maximum 35–50 m object count. Model
predictions were not used for scene selection. Each image uses identical
tokens, a 0.30 score threshold, a 50 m range, and GT/Base/Lite rows.

- `docs/assets/visualizations/dense_94126983bc0c4de89fb27cefb81f24ef.png`
- `docs/assets/visualizations/turning_1eeeb68791084aa7b2e5d05223796e1e.png`
- `docs/assets/visualizations/occlusion_2ff5689a089844f0a2097b1883916003.png`
- `docs/assets/visualizations/far_cdc90732e957479da6c8a80c9fad6f78.png`
- Selection metadata: `experiments/lite/visualization_manifest.json`

## Interpretation rules

- Encoder-4 and Encoder-3 initially represent inference-time early exit using
  the first four or three trained encoder layers. They are not retrained
  smaller models unless explicitly marked as fine-tuned/retrained.
- Temporal OFF disables recurrent history while retaining the trained Temporal
  Self-Attention computation. It measures the value of history, not the speed
  benefit of deleting the attention module.
- BEV-150 and BEV-100 are interpolated zero-shot runs: the trained BEV and
  positional embeddings are resized without fine-tuning or retraining. They
  measure test-time resolution robustness; any trained lightweight claim must
  be based on a separately fine-tuned or retrained checkpoint.

## Reproduction

```bash
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}"

# Base evaluation
python tools/test.py \
  projects/configs/bevformer/bevformer_base.py \
  ckpts/bevformer_r101_dcn_24ep.pth \
  --eval bbox

# Base profiling
python tools/profile_bevformer.py \
  projects/configs/bevformer/bevformer_base.py \
  ckpts/bevformer_r101_dcn_24ep.pth \
  --warmup 10 --iters 100 --workers 2 \
  --output experiments/baseline/profile.json

# Two-epoch BEV-150 adaptation
python tools/train.py \
  projects/configs/bevformer_ablation/bev150_finetune.py \
  --work-dir work_dirs/bev150_finetune_2ep \
  --gpus 1 --seed 0 --no-validate

# Continue from epoch 2 with retained AdamW state and lower learning rate
python tools/rebase_optimizer_lr.py \
  work_dirs/bev150_finetune_2ep/epoch_2.pth \
  ckpts/ablation/bev150_epoch2_resume_lr5e-6.pth \
  --target-lr 5e-6
python tools/train.py \
  projects/configs/bevformer_ablation/bev150_finetune_6ep.py \
  --work-dir work_dirs/bev150_finetune_6ep \
  --resume-from ckpts/ablation/bev150_epoch2_resume_lr5e-6.pth \
  --gpus 1 --seed 0 --no-validate
```

The fine-tune uses
`projects/configs/bevformer_ablation/bev150_finetune.py`, starts from the
interpolated BEV-150 checkpoint, runs two epochs at learning rate `2e-5`, and
writes to `work_dirs/bev150_finetune_2ep`. Its epoch-2 checkpoint was evaluated
and profiled independently as L2; the original zero-shot B1/L1 result remains
unchanged for comparison. The extended L3 run evaluates epochs 3--6 and uses
epoch 6, selected by full-validation NDS.
