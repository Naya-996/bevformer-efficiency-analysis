# RC-BEV implementation and handoff

## Current outcome

The repository now contains a runnable method prototype for continuous BEV queries,
physical-coordinate temporal-state migration and deterministic budget adaptation.
The additions are opt-in. `projects/configs/bevformer/bevformer_base.py` remains a
fixed 200x200 configuration and builds with zero continuous-generator parameters.

This is an implementation milestone, not a paper result. The method checkpoint has
not been trained or evaluated on the full nuScenes validation set.

## Delivered files

- Evidence audit: `audit/evidence_integrity_report.md`,
  `audit/evidence_integrity.json`, `audit/artifact_hashes.csv`,
  `audit/missing_artifacts.md`.
- Design: `docs/resolution_continuous_design.md`.
- Query and migration: `projects/mmdet3d_plugin/bevformer/modules/resolution_continuous.py`.
- Controller: `projects/mmdet3d_plugin/bevformer/modules/budget_controller.py`.
- Head/transformer/detector integration: `bevformer_head.py`, `transformer.py`,
  and `detectors/bevformer.py` under the plugin.
- Configs: fixed 150, multi-resolution, FP16, seen/unseen evaluation, adaptive
  inference and no-history-migration ablation under `projects/configs/bevformer_rc/`.
- Warm-start utility: `tools/convert_checkpoint_to_continuous_bev.py`.
- CPU/GPU component tests: `tests/test_resolution_continuous.py`,
  `tests/test_rc_bev_config_contract.py`, and
  `tools/smoke_test_rc_bev_components.py`.
- Experiment entry point: `run_rc_bev_experiments.sh`.
- Single truth source and rendering: `experiments/rc_bev/results.json`,
  `result_schema.json`, `experiment_matrix.json`, and
  `tools/render_rc_bev_results.py`.

## Observed verification

- 25 CPU tests pass (query shape/dtype/backward, physical coordinates, row-major
  order, same/cross-grid migration, constant and coordinate fields, non-square
  grids, deterministic sampling, controller hysteresis/reset, and config contracts).
- Official Base-200 detector checkpoint loads into the opt-in model without any
tensor-size mismatch. Exactly eleven new generator tensors (nine parameters plus
persistent Fourier-frequency and schedule-step buffers) are reported missing and initialized by the new module,
as designed. The retained legacy query table is
  `[40000,256]`; fixed-150 is selected only at runtime.
- GPU component smoke passes on physical GPU1 / RTX 5090 with PyTorch 2.7.1+cu128
  for 100, 125, 140, 150, 160, 175, 180, 200 and non-square 96x144 grids.
- GPU history migration passes for 100->150, 150->200, 200->100 and
  100->96x144; the constant-field maximum error is zero.
- A real nuScenes training batch at runtime 100x100 completed forward, backward,
  optimizer update and strict checkpoint round-trip. All 563 gradient-bearing
  tensors were finite; peak allocated/reserved memory was 18612.6/19688.0 MiB.
- One full detector inference at 100x100 completed with the official detector
  checkpoint plus newly initialized continuous parameters. Its one-sample timing
  is stored in `experiments/rc_bev/gpu_smoke_profile_100.json` and is not a valid
  benchmark.
- Historical manifest files now match the pre-repair archive byte-for-byte. The
  independent audit still reports the six original config-hash mismatches instead
  of concealing them.

`nvidia-smi` cannot initialize NVML because the driver and NVML library versions
differ. CUDA tensor allocation and model inference nevertheless work. Energy and
reliable process telemetry are blocked until NVML is repaired.

## Commands

```bash
# No proxy is used; all GPU actions expose physical GPU1 only.
./run_rc_bev_experiments.sh audit
./run_rc_bev_experiments.sh cpu-test
./run_rc_bev_experiments.sh gpu-component-smoke

# Full training, unique timestamped output directory.
SEED=0 ./run_rc_bev_experiments.sh train-multires

# After a real continuous checkpoint exists:
./run_rc_bev_experiments.sh eval-seen /path/to/checkpoint.pth
./run_rc_bev_experiments.sh eval-unseen /path/to/checkpoint.pth
./run_rc_bev_experiments.sh eval-dynamic /path/to/checkpoint.pth
./run_rc_bev_experiments.sh profile-seen /path/to/checkpoint.pth
./run_rc_bev_experiments.sh profile-seen-fp16 /path/to/checkpoint.pth
```

The renderer rejects a `COMPLETE` row unless raw metrics/profile paths exist and
the profile contains all declared per-iteration samples. Non-complete rows are
rejected if result numbers are inserted. Generated LaTeX therefore currently says
“No verified method results yet”.

## Remaining experiment work

All 20 matched-protocol result rows are `NOT RUN`: three seeds of Base-200, fixed-150 and
continuous multi-resolution training; eight fixed-resolution evaluations; dynamic
evaluation; and Small/Tiny baselines. Full validation, class/distance/visibility and
switch-window slices, 20+200 FP32/FP16 profiling, MACs, energy, and statistical
aggregation remain required before making method claims or updating NCA Results.

The paused Full-24 fixed-grid job is independent of this method. Its valid resume
point remains `work_dirs/bev150_fulltrain_24ep/epoch_1.pth`; no RC-BEV script resumes
that job.
