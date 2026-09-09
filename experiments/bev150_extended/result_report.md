# BEV-150 extended fine-tuning result

Status: **complete**. Epoch 6 was selected by maximum NDS
with mAP as the tie-breaker after full evaluation of epochs 3--6.

| Variant | NDS | mAP | FPS | Mean / P95 latency (ms) | Peak allocated (MiB) | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Base | 0.5173 | 0.4163 | 3.951 | 253.12 / 253.61 | 2184.6 | 69.035 |
| L2 BEV-150, 2 ep | 0.5097 | 0.4074 | 4.325 | 231.20 / 231.90 | 2150.1 | 64.542 |
| L3 BEV-150, best of 3--6 ep | 0.5123 | 0.4082 | 4.323 | 231.33 / 231.77 | 2150.1 | 64.542 |

## Measured deltas

- L3 versus the two-epoch model: NDS +0.0026, mAP +0.0009.
- L3 versus Base: NDS -0.0051, mAP -0.0081.
- L3 versus Base efficiency: FPS +9.42%, mean latency -8.61%, parameters -6.51%.
- Largest class-level gains over L2: bicycle +1.50 AP, traffic_cone +0.58 AP, car +0.38 AP.
- Largest class-level regressions from L2: bus -0.91 AP, barrier -0.30 AP, motorcycle -0.29 AP.

## Provenance

- Selection rule: `maximum NDS, then maximum mAP`
- Best checkpoint: `work_dirs/bev150_finetune_6ep/epoch_6.pth`
- Checkpoint SHA256: `396fb656bcbf1848419613192e1316bfd11f7d86f032a5e2bf6b8c634444d9a6`
- Full epoch metrics: `experiments/bev150_extended/summary.json`
- Paired Base profile: `experiments/bev150_extended/paired_base_profile_gpu1.json`
- Paired L2 profile: `experiments/bev150_extended/paired_2ep_profile_gpu1.json`
- Paired L3 profile: `experiments/bev150_extended/paired_best_profile_gpu1.json`

All accuracy values come from the complete 6,019-sample nuScenes validation
set. Performance was measured back-to-back on physical GPU1 with 20 warm-up
and 200 synchronized iterations per model; data loading is excluded.
