# Evidence Integrity Report

Generated from the local repository by `tools/audit_evidence_integrity.py`.
The audit is read-only with respect to historical evidence and does not repair declarations.

## Scope and status

- Git commit: `b1a1d853d21ac9e0e554655443d2c4bce922a625`
- Worktree dirty during audit: `false`
- Evaluation manifests: 7
- Directly verifiable: 1
- Hash mismatch: 6
- Summary only: 0
- Completely missing: 0

## Evaluation evidence

| Manifest | Classification | Config | Checkpoint | Metrics |
|---|---|---:|---:|---:|
| `experiments/baseline/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |
| `experiments/bev_resolution/bev100/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |
| `experiments/bev_resolution/bev150/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |
| `experiments/encoder_depth/encoder3/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |
| `experiments/encoder_depth/encoder4/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |
| `experiments/lite_finetuned/eval_manifest.json` | directly_verifiable | verified | verified | verified |
| `experiments/temporal/temporal_off/eval_manifest.json` | hash_mismatch | HASH MISMATCH | verified | verified |

## Profiling evidence

A profile is directly verifiable here only when it stores per-iteration samples, warmup and measured counts, GPU identity, software versions, timing boundary, and synchronization policy.

- `experiments/baseline/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/bev150_extended/paired_2ep_profile_gpu1.json`: **directly_verifiable**; samples=200; missing fields=none.
- `experiments/bev150_extended/paired_base_profile_gpu1.json`: **directly_verifiable**; samples=200; missing fields=none.
- `experiments/bev150_extended/paired_best_profile_gpu1.json`: **directly_verifiable**; samples=200; missing fields=none.
- `experiments/bev_resolution/bev100/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/bev_resolution/bev150/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/encoder_depth/encoder3/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/encoder_depth/encoder4/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/lite_finetuned/profile.json`: **directly_verifiable**; samples=100; missing fields=none.
- `experiments/rc_bev/gpu_smoke_profile_100.json`: **directly_verifiable**; samples=1; missing fields=none.
- `experiments/temporal/temporal_off/profile.json`: **directly_verifiable**; samples=100; missing fields=none.

## Aggregate reproducibility

- 11/11 summary rows reproduce their stored raw metric/profile values exactly within `1e-12`.

## Interpretation

- `directly_verifiable`: referenced config, checkpoint and raw metrics exist and declared hashes match.
- `hash_mismatch`: artifacts exist, but at least one declared digest differs; the historical declaration is retained.
- `summary_only`: an aggregate/profile exists without all raw evidence required for independent recomputation.
- `completely_missing`: a declared path is absent or a manifest cannot be parsed.
- Existing fixed-grid measurements are historical observations. They are not evidence for the new resolution-continuous method.
- No new method accuracy, latency, memory or controller result has been run; those fields remain `NOT RUN`.
