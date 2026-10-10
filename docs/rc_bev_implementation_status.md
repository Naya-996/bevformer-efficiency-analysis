# RC-BEV implementation and experiment status

## Follow-up status on 2026-10-10 (Asia/Shanghai)

- All 20 short jobs completed successfully at 04:09 on October 10: unseen
  140/160/180, two closed-loop dynamic policies and five FP16 resolutions, each
  with full validation and separate 20+200 profiling.
- Live evaluation state: experiments/rc_bev/runs/rc_followup_20261009/status.json.
- The serial dual-GPU training queue started at 04:10 and runs matched-budget
  Base-200, continuous fixed-200, RC-BEV Seeds 1/2, discrete fixed-150 and continuous
  fixed-150 controls. The full plan has 12 new 24-epoch runs. Each run is followed
  by standalone validation and GPU1 profiling. Base-200 Seed 0 is in epoch 2
  (13,000/14,065 steps at 17:55); the remaining runs are queued, not completed.
- Training state and plan:
  experiments/rc_bev/runs/rc_training_followup_20261009/.
- Detection visualizations are now available under docs/assets/rc_bev_visualizations/;
  the manifest records the original prediction paths, hashes and GT-selected frames.
- Dynamic reset/resample policies use prediction-dependent controllers, so their
  resolution sequences may differ. A fixed-sequence replay is still needed to
  isolate the causal contribution of history migration.

Both queues run in tmux and survive an SSH disconnect. They do not automatically
restart after a machine reboot. Reinvoke the corresponding Python queue command
with the same run directory to skip completed jobs and resume training from the
latest epoch checkpoint. Job failures are recorded explicitly; the long-training
queue stops on a training/evaluation failure.

## Completed scope

RC-BEV now has one complete Seed 0 training and evaluation cycle:

- multi-resolution training at 100, 125, 150, 175 and 200;
- 24 epochs / 337,560 iterations;
- full 6,019-frame nuScenes validation at all five seen resolutions;
- FP32 latency profiling at all five resolutions with 20 warmup and 200 measured
  iterations;
- archived metrics, per-iteration latency samples and provenance hashes;
- reproducible rendering of CSV, LaTeX tables and plots from the evidence files.
- three unseen-resolution evaluations, five FP16 evaluations and two dynamic
  policy evaluations, each paired with a complete latency profile;
- four six-camera detection comparison figures with fixed GT-selected samples.

The final local checkpoint is:

~~~text
experiments/rc_bev/runs/rc_multires_s0_dual_retry_20260911T2225CST/work_dir/epoch_24.pth
SHA256: 741e6b937fadc5083ca833ee2b1ed4a3339bcf4714a967124704f0baa6a71e73
Size: 746878834 bytes
~~~

It is intentionally excluded from Git because it exceeds normal GitHub limits.
The exact metadata is stored in
experiments/rc_bev/evidence/seed0_epoch24/provenance.json.

## Verified implementation

- ContinuousBEVQueryGenerator maps physical cell-centre coordinates through
  multi-frequency Fourier features and two MLPs to BEV content queries and
  positional encodings.
- Runtime BEV shapes can be changed without resizing a learned query table.
- Training selects one deterministic resolution for the complete batch and
  temporal queue; the schedule step is checkpointed.
- Previous-BEV tensors carry an explicit source shape and are resampled in physical
  coordinates before ego-motion rotation when the grid changes.
- The opt-in path preserves the original fixed-grid configuration. Official
  fixed-grid tables remain available for checkpoint compatibility and are frozen
  when the continuous path is active.
- CPU contract tests, GPU component smoke tests, a real training batch and strict
  checkpoint round-trip have passed.
- Standalone full-dataset validation succeeded after the earlier inline-validation
  DataContainer incompatibility.

## Seed 0 results

| Resolution | NDS | mAP | Mean latency (ms) | P50 (ms) | P95 (ms) | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.509989 | 0.406942 | 219.176 | 219.165 | 219.593 | 4.563 |
| 125 | 0.514720 | 0.412871 | 224.952 | 224.899 | 225.487 | 4.445 |
| 150 | 0.519398 | 0.415204 | 232.035 | 232.006 | 232.305 | 4.310 |
| 175 | 0.522037 | 0.416416 | 241.845 | 241.831 | 242.149 | 4.135 |
| 200 | 0.523164 | 0.417212 | 254.634 | 254.599 | 255.043 | 3.927 |

These are single-seed observations, not means or confidence intervals. The
previously reproduced official Base-200 result is useful context but is not a
matched-budget multi-seed baseline.

## Follow-up results

The new results are tabulated in the repository README and archived under
experiments/rc_bev/evidence/rc_followup_20261009/. Unseen 140/160/180 achieve NDS
0.516883/0.520736/0.522353. FP16 at 150 achieves 0.518590 NDS and 6.016 FPS.
The dynamic controller selects 200 for more than 96% of validation frames, so the
current policy does not establish a meaningful speed advantage. Reset and
resampling policies have different closed-loop resolution traces; a fixed-trace
comparison remains pending.

## Training recovery history

The run initially used Gloo because the loaded NVIDIA kernel module and NVML
library did not match. Epoch 1 completed before inline validation encountered an
MMCV DataContainer compatibility error. Training resumed with inline validation
disabled. After reboot repaired the driver/library mismatch, the run resumed from
epoch 3 with NCCL and completed epoch 24. Full standalone validation subsequently
passed.

This history describes execution recovery only; it is not presented as a method
contribution.

## Evidence and commands

- Truth source: experiments/rc_bev/results.json
- Archived raw summaries: experiments/rc_bev/evidence/seed0_epoch24/
- Generated results: experiments/rc_bev/results.csv and
  experiments/rc_bev/generated/
- Training manifest:
  experiments/rc_bev/runs/rc_multires_s0_dual_retry_20260911T2225CST/training_manifest.json

~~~bash
./run_rc_bev_experiments.sh cpu-test
./run_rc_bev_experiments.sh gpu-component-smoke
SEED=0 ./run_rc_bev_experiments.sh train-multires
./run_rc_bev_experiments.sh eval-seen /path/to/checkpoint.pth
./run_rc_bev_experiments.sh profile-seen /path/to/checkpoint.pth
.venv5090py39/bin/python tools/render_rc_bev_results.py
~~~

## Remaining work

- Train Seeds 1 and 2 for RC-BEV and matched Base-200/fixed-150 baselines.
- Optimize the adaptive controller and add a fixed-switch-sequence migration ablation.
- Complete MACs/FLOPs and energy measurements; FP16 and peak-memory profiles exist.
- Add BEVFormer-S/Tiny under the same hardware and timing protocol.
- Add per-class, distance, visibility/occlusion and velocity-error analyses.
- Report mean, standard deviation or confidence intervals only after multi-seed
  experiments are complete.

The paused fixed-grid BEV-150 Full-24 experiment is independent of RC-BEV and is
not included in the results above.
