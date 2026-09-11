# BEVFormer efficiency study: journal upgrade plan

## Chosen direction

The recommended direction is a **systematic empirical study**. The existing
repository already contains resolution, encoder-depth, temporal-history,
accuracy, latency, memory, and short-adaptation experiments. Turning this into
an adaptive-grid method would require a new algorithm, convergence study, and
strong method baselines; it should remain a separate project unless a concrete
adaptive mechanism is designed first.

## Claims permitted before the new matrix is complete

- Existing Base, zero-shot BEV-150, Adapt-2, and Adapt-6 results are single-run
  measurements. They must not be described as statistically significant.
- Inference profiles describe an RTX 5090 FP32 software stack and cannot be
  generalized to embedded hardware.
- Full-24 is an active run and has no accuracy result yet.
- Early-exit and temporal-off experiments are diagnostic ablations, not trained
  competitive lightweight models.

## Controlled training matrix

All full-training rows use the same nuScenes train split, FCOS3D standard
initialization, seed-specific RNG state, optimizer, scheduler, loss, input
pipeline, R101+DCNv2 backbone, four-level FPN, six encoder layers, six decoder
layers, temporal history, batch size, and 24-epoch budget. Only BEV resolution
changes within the resolution curve.

| BEV | Seed 0 | Seed 1 | Seed 2 | Role |
| ---: | :---: | :---: | :---: | --- |
| 100 | required | optional | optional | resolution curve |
| 125 | required | optional | optional | resolution curve |
| 150 | running | required | required | curve plus three-seed final model |
| 175 | required | optional | optional | resolution curve |
| 200 | required | required | required | curve plus three-seed Base control |

Primary statistical comparison: Base-200 versus Full-24 BEV-150 across seeds
0, 1, and 2. Report mean, sample standard deviation, and a 95% t confidence
interval. With only three observations the confidence interval will be wide;
individual seed values must remain visible.

The zero-shot/Adapt-2/Adapt-6 sequence is reported separately as a transfer-cost
study and is not mixed into the equal-budget statistical comparison.

## Compute estimate and scheduling constraint

The active BEV-150 run processes 28,130 iterations per epoch at approximately
1.30 seconds per iteration. One 24-epoch run therefore needs roughly 10.2 GPU
days. The table above requires nine full runs in total, including the active
run, or approximately 92 single-GPU days before evaluation. Eight additional
runs should not be launched automatically without reserving GPU1 for about
three months and confirming storage/checkpoint retention policy.

## Competitive model comparison

Use the repository's official configurations first because they share the same
codebase and evaluation stack:

1. `projects/configs/bevformer/bevformer_small.py`
2. `projects/configs/bevformer/bevformer_tiny.py`

Evaluate their official checkpoints on all 6,019 validation samples and profile
them on physical GPU1 under the same FP32 and FP16 protocols. If retraining is
used, report it separately from official-checkpoint evaluation. Cross-repository
models such as Fast-BEV may be added only after input resolution, preprocessing,
precision, post-processing, and timing boundaries are made comparable.

## Efficiency protocol

For every selected model record:

- parameter count and trainable parameter count;
- FLOPs and MACs with the exact counting convention and unsupported-op list;
- peak allocated and reserved CUDA memory;
- model-only latency and end-to-end latency including data transfer,
  preprocessing, decoding, and post-processing;
- mean, standard deviation, P50, P95, FPS, batch size, warm-up, and repetitions;
- FP32 and FP16 results; numerical accuracy must be re-evaluated under FP16;
- board power sampled with timestamps and integrated energy per frame where
  NVML is available.

TensorRT or Jetson Orin results belong in a separate deployment table. They
must not be directly merged with PyTorch RTX 5090 latency.

## Error analysis

Persist per-sample predictions and ground truth so the following fixed bins can
be recomputed without rerunning inference:

- all ten nuScenes classes;
- radial distance: `[0, 20)`, `[20, 40)`, `[40, 60)`, and `>=60 m`;
- visibility token groups defined before inspecting model deltas;
- ground-truth speed: stationary, slow, medium, and fast, with thresholds stated
  in the paper;
- translation, orientation, velocity, scale, and attribute errors.

Report sample/object counts for every bin and bootstrap confidence intervals for
paired per-sample deltas where the metric supports resampling.

## Execution order

1. Freeze and verify provenance for all completed experiments.
2. Finish the active Full-24 BEV-150 seed-0 run and its predeclared evaluations.
3. Repair NVML before any energy experiment, without interrupting training.
4. Run Base-200 seed 0 under the local stack, then seeds 1 and 2 for both
   Base-200 and BEV-150.
5. Run seed-0 BEV-100/125/175 to complete the equal-budget curve.
6. Evaluate and profile BEVFormer-Small and BEVFormer-Tiny.
7. Add FP16, FLOPs/MACs, end-to-end latency, and energy measurements.
8. Produce stratified error analysis and statistical tables.
9. Update the paper only from immutable manifests; retain negative results.

## Completion gates

No table is considered publishable unless it binds metrics to config,
checkpoint, source commit, command, environment, dataset hashes, and measurement
protocol. A run is complete only when its checkpoint, raw prediction output,
official evaluation output, logs, and SHA256 manifest are all present.
