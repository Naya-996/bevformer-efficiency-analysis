# BEV-150 two-epoch fine-tune result

Status: **complete**. This is the separately trained/evaluated L2 result; the
original B1 zero-shot result remains unchanged for provenance.

| Variant | NDS | mAP | FPS | Mean latency (ms) | Peak allocated (MiB) | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Base | 0.5173 | 0.4163 | 4.943 | 202.30 | 2184.6 | 69.035 |
| B1 BEV-150 zero-shot | 0.4917 | 0.3885 | 5.528 | 180.89 | 2150.1 | 64.542 |
| L2 BEV-150 fine-tuned 2 ep | 0.5097 | 0.4074 | 5.487 | 182.24 | 2150.1 | 64.542 |

## Comparison

- Fine-tuning recovery over zero-shot: NDS +0.0180, mAP +0.0189.
- L2 versus Base: NDS -0.0077, mAP -0.0089.
- L2 versus Base efficiency: FPS +11.00%, mean latency -9.91%.
- L2 versus zero-shot efficiency: FPS -0.74%, mean latency +0.75%.

## Provenance

- Config: `projects/configs/bevformer_ablation/bev150_finetune.py`
- Checkpoint: `work_dirs/bev150_finetune_2ep/epoch_2.pth`
- Metrics manifest: `experiments/lite_finetuned/eval_manifest.json`
- Profile: `experiments/lite_finetuned/profile.json`
- Full machine-readable table: `experiments/summary.json`

The fine-tune is a short two-epoch adaptation experiment, not a replacement
for a full target-resolution 24-epoch retraining study.
