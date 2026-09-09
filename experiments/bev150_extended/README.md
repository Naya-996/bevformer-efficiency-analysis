# BEV-150 extended fine-tuning

This experiment extends the completed two-epoch BEV-150 adaptation through
epoch 6. It resumes both model and AdamW state from epoch 2, rebases the peak
learning rate from `2e-5` to `5e-6`, and uses cosine decay without a second
warm-up phase.

The run is intentionally checkpoint-selective rather than last-checkpoint
selective. Epochs 3, 4, 5, and 6 are each evaluated on all 6,019 nuScenes
validation samples. The checkpoint with the highest NDS (mAP as tie-breaker)
is then profiled with 10 warm-up and 100 measured iterations.

Expected generated artifacts:

- `epoch_3_eval.json` through `epoch_6_eval.json`: checkpoint-bound metrics
- `summary.json`: selection provenance and best checkpoint
- `paired_*_profile_gpu1.json`: back-to-back Base/L2/L3 latency, FPS,
  parameters, and peak GPU memory under one measurement protocol
- `result_report.md`: comparison against Base and the two-epoch model
- `COMPLETE`: written only after training, evaluation, and profiling succeed

Until `COMPLETE` exists, no accuracy gain from this run should be reported.
