# RC-BEV experiment status

All 20 rows in `results.json` are currently `NOT RUN`. No accuracy or performance
claim for the continuous-query model, cross-grid temporal migration, or adaptive
controller has been inserted into the paper.

CPU implementation tests are not substitutes for nuScenes validation. A component
GPU smoke covered every declared grid and four history transitions, and one
100x100 end-to-end model forward completed. The latter used zero warmup and one
measurement, with newly initialized continuous parameters; its timing is therefore
only an execution check and is excluded from Results. Training, full validation,
protocol-compliant FP32/FP16 profiling, MACs and energy measurements remain
unmeasured. `nvidia-smi` still reports an NVML driver/library version mismatch.
