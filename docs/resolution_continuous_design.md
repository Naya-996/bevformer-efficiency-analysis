# Resolution-Continuous BEVFormer: code trace and design

## Evidence status

This document is a code-derived design, not an experimental result. Statements about
existing tensor flow are **observed** from commit
`ddb0a3d413cd89b9d04c8f39f5419fb11cb53eb5`; expected accuracy or speed effects are
**planned / not verified** until recorded by the result pipeline.

## Observed fixed-grid data flow

### Query and positional encoding

`BEVFormerHead._init_layers` creates `bev_embedding` with shape
`[bev_h * bev_w, 256]`. In `forward`, its weights become `bev_queries`; a zero mask
of shape `[B, bev_h, bev_w]` is passed to mmdetection's
`LearnedPositionalEncoding`. That encoder owns separate row and column tables and
returns `[B, 256, bev_h, bev_w]` by concatenating x then y embeddings.

`PerceptionTransformer.get_bev_features` changes the tensors to:

- query: `[H*W, 256] -> [H*W, B, 256]`;
- position: `[B, 256, H, W] -> [H*W, B, 256]` via `flatten(2)`;
- encoder output and cached history: `[H*W, B, 256]`.

PyTorch's flatten is row-major here: flat index `i = row * W + column`. The
corresponding normalized reference point is
`((column + 0.5) / W, (row + 0.5) / H)`.

### Reference points and physical extent

`BEVFormerEncoder.get_reference_points` constructs both 2-D temporal and 3-D
spatial reference points at cell centres. It maps normalized x/y/z through
`pc_range`; therefore changing H/W changes sampling density but not the physical
extent. The head passes `grid_length=(real_h/H, real_w/W)`. Translation is converted
to a normalized x/y shift, so its final value is independent of the chosen cell
count when the physical range is unchanged.

The decoder does not require a fixed grid size. It consumes the encoder memory with
`spatial_shapes=[[H,W]]`; its 900 object queries and learned 3-D reference points are
independent of H/W. Deformable attention does require the sequence length to equal
`H*W` and therefore cannot accept an unresized history tensor.

### Temporal state and ego motion

During training, `BEVFormer.obtain_history_bev` iterates over the queued frames and
passes the previous encoder output back into the head. During video inference,
`prev_frame_info` caches that output and resets it when `scene_token` changes or
video mode is disabled. The detector converts absolute CAN-bus pose into framewise
deltas before the head call.

Inside the transformer, history is interpreted as `[H*W,B,C]` (or conditionally
permuted from `[B,H*W,C]`), reshaped to `[H,W,C]`, rotated using the yaw delta, then
used by temporal self-attention with translation-shifted 2-D reference points.
The configured `rotate_center=[100,100]` is only correct for the original 200 square
grid. A continuous model must instead use `(W/2,H/2)` for the active grid.

The dataset's `bev_size` is stored but is not used to create queries or reference
points in the inspected pipeline. Runtime resolution must therefore have one source
of truth in the head and must travel with the cached temporal tensor; silently
inferring a non-square grid from `sqrt(sequence_length)` is forbidden.

## Candidate designs

| Candidate | Checkpoint compatibility | Continuous/non-square support | Main risk |
|---|---|---|---|
| Runtime interpolation of fixed query and row/column tables | Strong: directly reuses official weights | Technically supports arbitrary H/W | Still parameterizes one privileged source lattice; interpolation may blur high-frequency learned structure and is not truly coordinate generated |
| Fourier coordinates plus MLP | New query/position parameters need adaptation | Natural support for all H/W and physical extents | Cold-start optimization and possible spectral aliasing |
| Shared content vector plus continuous positional bias | Compact and resolution independent | Natural support; clean content/position separation | A single content vector may underfit spatial priors |

## Selected method

The prototype combines candidates 2 and 3. For every target cell centre, physical
coordinates are normalized within `pc_range` to `u,v in [-1,1]`. With frequency
bands `f_k = 2^k`, the feature is

`phi(u,v) = [sin(pi f_k u), cos(pi f_k u), sin(pi f_k v), cos(pi f_k v)]_k`.

Two small MLPs produce a query bias and positional encoding:

`q(u,v) = c + MLP_q(phi(u,v))`,

`p(u,v) = MLP_p(phi(u,v))`,

where `c` is one learned `[1,256]` content vector. For batch size B the generator
returns `q` as `[H*W,256]` and `p` as `[B,256,H,W]`, both on the requested device and
dtype. It creates no parameter whose size depends on H or W.

The generator is optional in `BEVFormerHead`. When its config is absent, the exact
original embedding and learned positional-encoding path is retained. Consequently
official fixed-grid checkpoints keep their original behavior. When enabled, the
new parameters are not present in an official checkpoint and are newly initialized;
this is not a lossless conversion. The conversion utility can regress the continuous
generator against an official grid's query and row/column tables to provide a
warm-start, after which multi-resolution adaptation training is required.

## Temporally consistent grid switching

Every produced BEV tensor carries its explicit `(H,W)` in detector state. Before
ego-motion rotation, a history tensor from `(Hs,Ws)` is resampled onto `(Ht,Wt)`.
For target cell centre `(x_t,y_t)` in metres, the source sampling location is:

`g_x = 2 * (x_t - x_min_source) / (x_max_source - x_min_source) - 1`,

`g_y = 2 * (y_t - y_min_source) / (y_max_source - y_min_source) - 1`.

`grid_sample(..., align_corners=False)` then maps target cell centres to source cell
centres and supports unequal H/W or physical ranges. With identical shape and range,
the migration returns the original tensor unchanged. Out-of-range physical regions
use border padding so a constant field remains constant. Resampling precedes yaw rotation; translation remains represented
by the existing shifted reference points. Scene changes and missing history bypass
migration.

## Training and runtime selection

Multi-resolution training samples one H/W for the whole batch and all batch replicas
from `{100,125,150,175,200}` using a seed-controlled deterministic schedule. The
active shape is recorded by a hook as JSON Lines. Evaluation overrides the shape
without changing parameters; `{140,160,180}` are predeclared unseen grids.

The first adaptive controller is deliberately non-learning. A low-cost difficulty
score combines the already available pose delta and previous detections. Separate
up/down thresholds, one-level transitions and a minimum residency time provide
hysteresis. A scene change resets to the safe 200 grid. Controller and migration
wall times are recorded separately. Correlation between this score and accuracy
benefit is an experimental hypothesis, not a result.

## Complexity and failure modes

Generator cost is `O(HW * (F*d + d^2))` and is small relative to six-layer spatial
cross-attention, but it is not assumed free and must be profiled. History migration
is `O(Ht*Wt*d)` and temporarily materializes NCHW source/output tensors.

Expected failure modes include poor warm-start fit, degradation at unseen frequencies,
boundary artifacts after large down/up switches, controller oscillation under noisy
confidence, CUDA memory spikes at 200, and differences between interpolation and
physical reprojection when `pc_range` changes. Each remains **not verified**.

## Test strategy

CPU tests cover generator shape/dtype, cell endpoints and row-major flatten order;
same-grid identity; constant and coordinate-field resampling; 100->150, 150->200,
200->100 and non-square transitions; missing history and reset behavior; deterministic
controller hysteresis/residency; and the default-off fixed-grid head contract. A GPU
smoke test performs real forward/memory measurement only when NVML/CUDA is healthy.
