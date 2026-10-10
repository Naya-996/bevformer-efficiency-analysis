# RC-BEV experiment status

As of 2026-10-10, results.json contains 15 completed accuracy/profile pairs:
five seen FP32 resolutions, three unseen FP32 resolutions, five FP16 resolutions,
and two closed-loop dynamic policies. All use the same Seed 0 epoch-24 checkpoint.
These are not 15 independent training runs.

Matched-budget Base-200 Seed 0 is training on two GPUs. The training queue also
includes continuous fixed-200, RC-BEV Seeds 1/2, and discrete/continuous fixed-150
controls. None of those new training runs is reported as a completed result yet.

Multi-seed means, fixed-trace migration ablations, Small/Tiny comparisons, MACs,
energy and vehicle-hardware deployment remain pending. The training-only
rc_multires_s0 row has training_status COMPLETE; its accuracy/profile evidence is
represented by the separate fixed-resolution evaluation rows.

See the repository README and docs/rc_bev_implementation_status.md for measured
results and limitations. Early smoke profiles remain excluded from benchmark tables.
