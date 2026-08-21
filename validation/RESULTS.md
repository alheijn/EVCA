# EVCA2 Results Ledger

Running record of every benchmark run, default-behavior change, and gate decision.
Convention: one section per phase; benchmark rows record phase, commit SHA, CLI flags,
device, subset, fps, and correlation tables (or a pointer to the results directory).

## Environment

- Machine: Linux (CachyOS), NVIDIA GeForce RTX 5060 Ti, conda env `EVCA-gpu`
  (torch 2.13.0+cu130, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5). The `EVCA` env has a
  broken torch install (MKL `iJIT_NotifyEvent` symbol error) and is not used.
- ffmpeg n9.0.1 with libx265 available at `/usr/bin/ffmpeg`.
- Test sequences: `/home/albert/Desktop/test_sequences/` (UVG 1080p 8-bit yuv420:
  Beauty, Bosphorus, HoneyBee, ReadySteadyGo, ShakeNDry, YachtRide).
  `foodmarket_1920x1080_60fps` from the previous (macOS) configuration is **not**
  available on this machine; the benchmark set is YachtRide, ReadySteadyGo, HoneyBee,
  Bosphorus (paths configurable in `validation/sequences.json`).

## Phase 0 — bug fixes (default output may change)

Fixes 0.1–0.5 and the dead-code part of 0.6 were already applied on the base branch
before this work started (commit `b450068`, "Fix MV field registration, 8-bit optimized
loader, TC2 block CSV, MVC border padding, 10-bit plotting"). For completeness, the
default-output changes now in effect relative to the original Iteration-4 code:

| # | Fix | Commit | Default output change |
|---|-----|--------|----------------------|
| 0.1 | MV-field upsample `align_corners=False` (block MVs registered to block centers) | `b450068` | `TC_MC` changes (residual energy drops for coherent motion) |
| 0.2 | `load_gop_optimized`: int16 reinterpretation only for uint16 sources | `b450068` | `--loader optimized` on 8-bit input now produces correct (previously garbled) tensors |
| 0.3 | TC2 block CSV writes `TC2_blocks[i-2]` instead of `TC_blocks[i-2]` | `b450068` | `*_TC2_blocks.csv` content corrected |
| 0.4 | `MetricMVC` Laplacian uses replicate padding | `b450068` | **MVC values change** (no spurious border gradients under global motion) |
| 0.5 | `plot_block_info_EVCA` honors `args.bit_depth` | `b450068` | 10-bit plots read correct frames (plots only) |
| 0.6a | Dead code removed (`sub_tu_size`, `cached_weights_dct_sub`, `tc_uncomp_batch`) | `b450068` | none |
| 0.6b | `steps` reset per input file | `a95023d` | `--dir` runs where a short file precedes longer ones no longer shrink the GOP for later files (metric values unchanged, only batching; TC2 frame-0 padding could differ in the pathological case) |
| 0.7 | `libs/frame_to_edge.py` deleted (unreferenced, called `edge_detection` with wrong signature) | `793a23a` | none |
| 0.8 | `grid_sample` base grid / Gaussian kernel cached per `(H, W, device)` | `9339ad3` | none (identical numerics, fewer kernel launches) |
| 0.9 | Comments corrected: search runs at **half** resolution (2×2 pool), full-res MVs even-valued | `9339ad3` | none |

Also pre-applied on the base branch (commit `f144ce2`): matmul DCT replacing
`torch_dct` as the transform implementation, deferred host syncs, GOP prefetch
(`--prefetch`, default 1), MPS support. Phase 2's `--dct-impl` flag treats **matmul as
the current default** (rule 2: defaults reproduce current behavior) and adds
`torch_dct` back as the ablation reference.

### Gate 0a

- Full pytest suite: pending Phase 1 tests (existing 31 tests pass on CPU).
- `--profile full` end-to-end on a real sequence: pending (recorded below once run).

### Run `gate1` — 2026-08-21 19:38

- Phase: Phase 1 (post-fix baseline)
- Commit: `5560c9d8180f14920ba48fd50092b996db95ea13`
- Subset: **fast** (120 frames), sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Device: `auto`, loader: `optimized`, profiles: baseline, fast, full
- Extra EVCA args: `(none)`
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/gate1_5560c9d8`

**Throughput**

| profile | frames | seconds | fps |
|---|---|---|---|
| baseline | 480 | 1.28 | 375.00 |
| fast | 480 | 1.59 | 301.89 |
| full | 480 | 1.94 | 247.42 |

**Frame-level pooled correlations** (CI = 95 % bootstrap; `blk` = sequence-level block bootstrap)

| Domain | QP | metric | n | PCC | PCC_lo | PCC_hi | PCC_blk_lo | PCC_blk_hi | SRCC | PCC_log |
|---|---|---|---|---|---|---|---|---|---|---|
| Spatial | 22 | baseline_SC | 480 | 0.7805 | 0.7373 | 0.8185 | -0.7762 | 0.9992 | 0.7231 | 0.7671 |
| Temporal | 22 | baseline_TC | 476 | 0.4371 | 0.3951 | 0.4809 | -0.6931 | 0.9854 | 0.7826 | 0.5339 |
| Temporal | 22 | fast_MVC | 476 | 0.8588 | 0.8329 | 0.8834 | 0.2413 | 0.9822 | 0.8261 | 0.9045 |
| Temporal | 22 | fast_TC_SAD | 476 | 0.5188 | 0.4621 | 0.5919 | -0.1833 | 0.9375 | 0.7707 | 0.5668 |
| Temporal | 22 | full_TC_MC | 476 | 0.6437 | 0.6054 | 0.6893 | -0.9725 | 0.9841 | 0.7156 | 0.6367 |
| Spatial | 27 | baseline_SC | 480 | 0.9817 | 0.9785 | 0.9843 | 0.9018 | 0.9918 | 0.9886 | 0.9865 |
| Temporal | 27 | baseline_TC | 476 | 0.4879 | 0.4488 | 0.5272 | -0.6945 | 0.9819 | 0.7827 | 0.5972 |
| Temporal | 27 | fast_MVC | 476 | 0.8898 | 0.8691 | 0.9101 | 0.3719 | 0.9860 | 0.8528 | 0.9091 |
| Temporal | 27 | fast_TC_SAD | 476 | 0.5583 | 0.5081 | 0.6224 | -0.1365 | 0.9477 | 0.7699 | 0.5661 |
| Temporal | 27 | full_TC_MC | 476 | 0.6550 | 0.6238 | 0.6928 | -0.9799 | 0.9894 | 0.7115 | 0.5514 |
| Spatial | 32 | baseline_SC | 480 | 0.9899 | 0.9877 | 0.9916 | 0.9534 | 0.9987 | 0.9922 | 0.9849 |
| Temporal | 32 | baseline_TC | 476 | 0.5237 | 0.4856 | 0.5608 | -0.6881 | 0.9784 | 0.7830 | 0.6319 |
| Temporal | 32 | fast_MVC | 476 | 0.8849 | 0.8655 | 0.9030 | 0.4031 | 0.9854 | 0.8581 | 0.9021 |
| Temporal | 32 | fast_TC_SAD | 476 | 0.5992 | 0.5552 | 0.6574 | -0.0913 | 0.9546 | 0.7710 | 0.5880 |
| Temporal | 32 | full_TC_MC | 476 | 0.7007 | 0.6739 | 0.7334 | -0.9767 | 0.9914 | 0.7109 | 0.5667 |
| Spatial | 37 | baseline_SC | 480 | 0.9778 | 0.9737 | 0.9813 | 0.6971 | 0.9997 | 0.9624 | 0.9541 |
| Temporal | 37 | baseline_TC | 476 | 0.5769 | 0.5407 | 0.6119 | -0.6658 | 0.9764 | 0.7870 | 0.6721 |
| Temporal | 37 | fast_MVC | 476 | 0.8814 | 0.8621 | 0.8984 | 0.4244 | 0.9835 | 0.8615 | 0.9002 |
| Temporal | 37 | fast_TC_SAD | 476 | 0.6530 | 0.6144 | 0.7040 | 0.0272 | 0.9610 | 0.7858 | 0.6264 |
| Temporal | 37 | full_TC_MC | 476 | 0.7511 | 0.7305 | 0.7766 | -0.9671 | 0.9931 | 0.7218 | 0.6135 |

**Sequence-mean correlations** (legacy, n = sequences)

| Domain | QP | Metric | PCC | SRCC | n |
|---|---|---|---|---|---|
| Spatial | 22 | baseline_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | baseline_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | fast_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | fast_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | fast_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | fast_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | fast_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | fast_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | fast_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | fast_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | fast_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Spatial | 22 | full_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | full_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | full_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | full_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | full_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | full_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | full_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | full_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | full_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | full_TC_MC | 0.7252 | 0.8000 | 4 |
| Temporal | 22 | full_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | full_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Temporal | 22 | full_intra_frac | 0.9966 | 1.0000 | 4 |
| Spatial | 27 | baseline_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | baseline_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | baseline_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | baseline_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | fast_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | fast_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | fast_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | fast_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | fast_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | fast_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | fast_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | fast_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | fast_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Spatial | 27 | full_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | full_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | full_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | full_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | full_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | full_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | full_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | full_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | full_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | full_TC_MC | 0.7301 | 0.8000 | 4 |
| Temporal | 27 | full_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | full_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Temporal | 27 | full_intra_frac | 0.9974 | 1.0000 | 4 |
| Spatial | 32 | baseline_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | baseline_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | baseline_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | baseline_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | fast_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | fast_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | fast_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | fast_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | fast_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | fast_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | fast_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | fast_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | fast_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Spatial | 32 | full_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | full_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | full_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | full_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | full_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | full_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | full_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | full_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | full_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | full_TC_MC | 0.7744 | 0.8000 | 4 |
| Temporal | 32 | full_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | full_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Temporal | 32 | full_intra_frac | 0.9998 | 1.0000 | 4 |
| Spatial | 37 | baseline_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | baseline_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | baseline_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | baseline_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | fast_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | fast_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | fast_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | fast_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | fast_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | fast_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | fast_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | fast_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | fast_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Spatial | 37 | full_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | full_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | full_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | full_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | full_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | full_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | full_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | full_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | full_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | full_TC_MC | 0.8211 | 0.8000 | 4 |
| Temporal | 37 | full_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | full_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Temporal | 37 | full_intra_frac | 0.9959 | 1.0000 | 4 |


#### Gate 1 — verdict and analysis

**Gate 1 met.** The harness runs end to end (EVCA extraction → per-frame x265 ground
truth → frame-level and sequence-mean correlations → ledger), and the three profiles
were re-run with the Phase 0 fixes in place. The numbers above are the post-fix
baseline that Phases 2–5 are measured against.

Additional Phase 0 fix found by the new tests and folded into this gate:

| # | Fix | Commit | Default output change |
|---|-----|--------|----------------------|
| 0.10 | `load_gop_optimized` raised `BufferError: cannot close exported pointers exist` whenever chroma was enabled. The `if 'X' in locals(): del X` idiom materialises the frame's `f_locals` snapshot, which then holds its own reference to the last chroma view, so the mmap still had an exported buffer at `close()`. Replaced with plain rebinding. | `dbfa6cc` | `--loader optimized` with `-cc`/`-cf` went from crashing to working |

**Frame alignment verified against the encoder.** A 1080p probe sequence with a
4 px/frame pan starting at frame 8 and a hard cut at frame 14 was encoded LDP at
QP 32. The x265 P-frame bits jump at frame_idx 8 (712 → 3800 bits) and spike at
frame_idx 14 (829 072 bits); EVCA's `TC`, `TC_SAD`, `mean_mv_mag` first become
non-zero at row 8 and `TC_MC`/`intra_frac` peak at row 14. Identical indices, so
EVCA row `f` ↔ LDP P-frame `f` is the correct alignment. `mean_mv_mag` was exactly
4.00 during the pan, and `TC_MC` fell to 0.8 against a raw `TC` of 167.7, which is
the Phase 0.1 registration fix working on real-resolution content. This probe is
now pinned by `tests/test_frame_alignment.py` (synthetic, no ffmpeg needed).

**Pooled correlations are dominated by between-sequence variance.** Mean *within*-
sequence frame-level PCC against `TC_gt`, averaged over the four sequences:

| metric | QP 22 | QP 27 | QP 32 | QP 37 |
|---|---|---|---|---|
| `TC` (baseline) | 0.179 | 0.301 | 0.271 | 0.283 |
| `TC2` (baseline) | 0.234 | 0.543 | 0.487 | 0.412 |
| `TC_SAD` | 0.034 | 0.214 | 0.269 | 0.407 |
| `MVC` | −0.027 | 0.145 | 0.174 | 0.170 |
| `TC_MC` | **0.287** | **0.385** | **0.405** | **0.516** |

`MVC` has the highest *pooled* PCC of any temporal metric (0.86–0.89) but the
second-*lowest* within-sequence PCC (0.15–0.17). Its pooled score is almost entirely
the between-sequence effect that high-motion content costs more bits; it barely
tracks frame-to-frame variation inside a sequence. `TC_MC` is the best within-sequence
temporal metric at every QP, which is the ordering Phases 3–4 should try to improve.
The sequence-level block-bootstrap CIs are correspondingly near-vacuous (frequently
spanning [−0.97, +0.99]) because there are only four sequences: with n = 4 groups the
block bootstrap has very few distinct resamples. **Consequence for Gates 3 and 4:** the
specified decision rule (lower bound of the 95 % CI of pooled frame-level PCC of
`TC_MC`) is applied as written, but the frame-level CI is used for it, and the
per-sequence table is reported alongside every gate, since the pooled statistic can be
moved by between-sequence effects that say nothing about per-frame prediction quality.

**Motion search is saturated on half the corpus.** Per-sequence means over frames ≥ 1:

| sequence | `MV_sat_frac` | `mean_mv_mag` | `intra_frac` | `TC_SAD` PCC @ QP 32 |
|---|---|---|---|---|
| HoneyBee | 0.2 % | 0.08 | 7.0 % | 0.569 |
| Bosphorus | 2.8 % | 1.96 | 13.6 % | −0.054 |
| ReadySteadyGo | 50.5 % | 4.18 | 20.7 % | 0.708 |
| YachtRide | 63.0 % | 4.62 | 36.8 % | −0.146 |

Overall `MV_sat_frac` is **29.1 %**, against the Phase 3 acceptance threshold of < 5 %.
On YachtRide and ReadySteadyGo the majority of blocks pick a motion vector on the
boundary of the ±6 px pattern, i.e. the true motion is outside the search range and
the reported SAD measures search failure rather than content complexity. This is the
direct motivation for the hierarchical search in Phase 3, and it is the most likely
explanation for `TC_SAD` correlating *negatively* with bits on YachtRide.

**Throughput** (RTX 5060 Ti, CUDA, 1080p, `--loader optimized`, 480 frames total):
baseline 375 fps, fast profile 302 fps, full profile 247 fps. The full profile is
already well above the Phase 3 target of 100 fps at 1080p, leaving headroom for a
more expensive search.

### Run `gate2-defaults` — 2026-08-21 19:46

- Phase: Phase 2 (defaults, byte-identity check)
- Commit: `2dc129d1c7a0ee34ab2b445dc9f0d20722c448cb`
- Subset: **fast** (120 frames), sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Device: `auto`, loader: `optimized`, profiles: baseline, fast, full
- Extra EVCA args: `(none)`
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/gate2-defaults_2dc129d1`

**Throughput**

| profile | frames | seconds | fps |
|---|---|---|---|
| baseline | 480 | 0.96 | 500.00 |
| fast | 480 | 1.45 | 331.03 |
| full | 480 | 1.91 | 251.31 |

**Frame-level pooled correlations** (CI = 95 % bootstrap; `blk` = sequence-level block bootstrap)

| Domain | QP | metric | n | PCC | PCC_lo | PCC_hi | PCC_blk_lo | PCC_blk_hi | SRCC | PCC_log |
|---|---|---|---|---|---|---|---|---|---|---|
| Spatial | 22 | baseline_SC | 480 | 0.7805 | 0.7373 | 0.8185 | -0.7762 | 0.9992 | 0.7231 | 0.7671 |
| Temporal | 22 | baseline_TC | 476 | 0.4371 | 0.3951 | 0.4809 | -0.6931 | 0.9854 | 0.7826 | 0.5339 |
| Temporal | 22 | fast_MVC | 476 | 0.8588 | 0.8329 | 0.8834 | 0.2413 | 0.9822 | 0.8261 | 0.9045 |
| Temporal | 22 | fast_TC_SAD | 476 | 0.5188 | 0.4621 | 0.5919 | -0.1833 | 0.9375 | 0.7707 | 0.5668 |
| Temporal | 22 | full_TC_MC | 476 | 0.6437 | 0.6054 | 0.6893 | -0.9725 | 0.9841 | 0.7156 | 0.6367 |
| Spatial | 27 | baseline_SC | 480 | 0.9817 | 0.9785 | 0.9843 | 0.9018 | 0.9918 | 0.9886 | 0.9865 |
| Temporal | 27 | baseline_TC | 476 | 0.4879 | 0.4488 | 0.5272 | -0.6945 | 0.9819 | 0.7827 | 0.5972 |
| Temporal | 27 | fast_MVC | 476 | 0.8898 | 0.8691 | 0.9101 | 0.3719 | 0.9860 | 0.8528 | 0.9091 |
| Temporal | 27 | fast_TC_SAD | 476 | 0.5583 | 0.5081 | 0.6224 | -0.1365 | 0.9477 | 0.7699 | 0.5661 |
| Temporal | 27 | full_TC_MC | 476 | 0.6550 | 0.6238 | 0.6928 | -0.9799 | 0.9894 | 0.7115 | 0.5514 |
| Spatial | 32 | baseline_SC | 480 | 0.9899 | 0.9877 | 0.9916 | 0.9534 | 0.9987 | 0.9922 | 0.9849 |
| Temporal | 32 | baseline_TC | 476 | 0.5237 | 0.4856 | 0.5608 | -0.6881 | 0.9784 | 0.7830 | 0.6319 |
| Temporal | 32 | fast_MVC | 476 | 0.8849 | 0.8655 | 0.9030 | 0.4031 | 0.9854 | 0.8581 | 0.9021 |
| Temporal | 32 | fast_TC_SAD | 476 | 0.5992 | 0.5552 | 0.6574 | -0.0913 | 0.9546 | 0.7710 | 0.5880 |
| Temporal | 32 | full_TC_MC | 476 | 0.7007 | 0.6739 | 0.7334 | -0.9767 | 0.9914 | 0.7109 | 0.5667 |
| Spatial | 37 | baseline_SC | 480 | 0.9778 | 0.9737 | 0.9813 | 0.6971 | 0.9997 | 0.9624 | 0.9541 |
| Temporal | 37 | baseline_TC | 476 | 0.5769 | 0.5407 | 0.6119 | -0.6658 | 0.9764 | 0.7870 | 0.6721 |
| Temporal | 37 | fast_MVC | 476 | 0.8814 | 0.8621 | 0.8984 | 0.4244 | 0.9835 | 0.8615 | 0.9002 |
| Temporal | 37 | fast_TC_SAD | 476 | 0.6530 | 0.6144 | 0.7040 | 0.0272 | 0.9610 | 0.7858 | 0.6264 |
| Temporal | 37 | full_TC_MC | 476 | 0.7511 | 0.7305 | 0.7766 | -0.9671 | 0.9931 | 0.7218 | 0.6135 |

**Sequence-mean correlations** (legacy, n = sequences)

| Domain | QP | Metric | PCC | SRCC | n |
|---|---|---|---|---|---|
| Spatial | 22 | baseline_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | baseline_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | fast_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | fast_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | fast_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | fast_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | fast_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | fast_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | fast_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | fast_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | fast_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Spatial | 22 | full_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | full_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | full_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | full_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | full_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | full_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | full_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | full_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | full_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | full_TC_MC | 0.7252 | 0.8000 | 4 |
| Temporal | 22 | full_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | full_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Temporal | 22 | full_intra_frac | 0.9966 | 1.0000 | 4 |
| Spatial | 27 | baseline_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | baseline_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | baseline_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | baseline_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | fast_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | fast_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | fast_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | fast_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | fast_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | fast_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | fast_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | fast_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | fast_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Spatial | 27 | full_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | full_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | full_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | full_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | full_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | full_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | full_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | full_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | full_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | full_TC_MC | 0.7301 | 0.8000 | 4 |
| Temporal | 27 | full_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | full_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Temporal | 27 | full_intra_frac | 0.9974 | 1.0000 | 4 |
| Spatial | 32 | baseline_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | baseline_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | baseline_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | baseline_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | fast_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | fast_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | fast_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | fast_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | fast_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | fast_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | fast_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | fast_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | fast_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Spatial | 32 | full_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | full_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | full_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | full_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | full_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | full_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | full_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | full_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | full_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | full_TC_MC | 0.7744 | 0.8000 | 4 |
| Temporal | 32 | full_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | full_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Temporal | 32 | full_intra_frac | 0.9998 | 1.0000 | 4 |
| Spatial | 37 | baseline_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | baseline_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | baseline_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | baseline_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | fast_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | fast_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | fast_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | fast_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | fast_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | fast_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | fast_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | fast_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | fast_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Spatial | 37 | full_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | full_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | full_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | full_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | full_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | full_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | full_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | full_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | full_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | full_TC_MC | 0.8211 | 0.8000 | 4 |
| Temporal | 37 | full_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | full_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Temporal | 37 | full_intra_frac | 0.9959 | 1.0000 | 4 |


### Ablation `gate2-ablation` — 2026-08-21 19:49

- Phase: Phase 2 (post-fix Iteration 1-4 ranking)
- Commit: `2dc129d1c7a0ee34ab2b445dc9f0d20722c448cb`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC`
- Axes: `mc` ∈ {dense_smooth, dense}; `gate` ∈ {intra, none}
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Results: `validation/results/gate2-ablation_2dc129d1`

Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.

| variant | PCC_mean | PCC_lo_mean | PCC_hi_mean | SRCC_mean | perseq_PCC_mean | fps |
|---|---|---|---|---|---|---|
| mc=dense_smooth gate=none | 0.6965 | 0.6653 | 0.7340 | 0.7166 | 0.3659 | 243.6548 |
| mc=dense_smooth gate=intra | 0.6876 | 0.6584 | 0.7230 | 0.7150 | 0.3984 | 237.6238 |
| mc=dense gate=none | 0.6438 | 0.6125 | 0.6823 | 0.6966 | 0.3256 | 248.7047 |
| mc=dense gate=intra | 0.6311 | 0.6020 | 0.6674 | 0.6847 | 0.3605 | 243.6548 |


#### Gate 2 — verdict

**Gate 2 met.** With every new flag at its default, the per-frame CSVs of all three
profiles on all four sequences are **byte-identical** to the Gate 1 output
(`md5sum -c`, 12/12 OK), so the strategy refactor changed no default behaviour.

Phase 2 items and how they landed:

| Item | Status |
|---|---|
| `--me`, `--me-subpel`, `--me-predictor`, `--me-lambda`, `--me-merge`, `--me-criterion` | Flags parsed; non-default values raise `NotImplementedError` until Phase 3, so an ablation can never silently report a variant it did not run |
| `--mc {dense_smooth,dense,block,obmc}`, `--mc-smooth {gauss,median,none}`, `--residual-dc`, `--gate {intra,none}` | Implemented (`libs/motion_compensation.py`) |
| MC as a class hierarchy that `TemporalState` delegates to | Done: `MotionCompensator` → `DenseMC` / `BlockMC` / `OBMC`; `TemporalState.compensator` |
| `--dct-impl {torch_dct,matmul}` | Done. **Default stays `matmul`** — it was already the shipped behaviour (commit `f144ce2`), and it is faster everywhere tested |
| Remove per-GOP `.cpu()` syncs | Already done on the base branch (`f144ce2`); verified there is exactly one host transfer per file, after the GOP loop |
| `--preset iter4` | Added early (Phase 4 needs it) and pinned by a test asserting it equals today's defaults |

**DCT backend benchmark** (`validation/bench_dct.py`, 32 frames of 1080p worth of
32×32 blocks per batch). Max relative disagreement `3.4e-07` on CUDA and `2.6e-07` on
CPU, both far inside the 1e-4 tolerance:

| device | impl | ms/batch | frames/s | speedup |
|---|---|---|---|---|
| CUDA | matmul | 2.792 | 11462 | 1.00× |
| CUDA | torch_dct | 30.980 | 1033 | 0.09× |
| CPU | matmul | 13.979 | 572 | 1.00× |
| CPU | torch_dct | 97.399 | 82 | 0.14× |

matmul is 11× faster on CUDA and 7× on CPU, so the default is left at `matmul`.
`torch_dct` is retained only as the ablation reference and is an optional import.

**Post-fix Iteration 1–4 ranking** (`--mc dense_smooth|dense` × `--gate intra|none`,
fast subset, ranking metric `TC_MC`, values averaged over QPs 22/27/32/37):

| variant | PCC | CI lo | per-seq PCC | fps |
|---|---|---|---|---|
| `dense_smooth` + `gate none` | 0.6965 | 0.6653 | 0.3659 | 243.7 |
| `dense_smooth` + `gate intra` | 0.6876 | 0.6584 | **0.3984** | 237.6 |
| `dense` + `gate none` | 0.6438 | 0.6125 | 0.3256 | 248.7 |
| `dense` + `gate intra` | 0.6311 | 0.6020 | 0.3605 | 243.7 |

Two findings. First, **Gaussian smoothing of the MV field still helps after the
Phase 0 fixes**: `dense_smooth` beats `dense` by ≈ 0.05 PCC at both gate settings, so
Iteration 4's smoothing choice was not an artefact of the misregistered MV field.
Second, **the two gate settings rank differently depending on the statistic**: by the
gate rule (CI lower bound of pooled PCC) `gate none` edges ahead (0.6653 vs 0.6584),
but by mean within-sequence PCC `gate intra` is clearly better (0.398 vs 0.366), and
the pooled CIs overlap almost completely. No default is changed here (Phase 2 changes
no defaults); the disagreement is carried into Gate 4, where the gate axis is decided.

### Ablation `gate3-me` — 2026-08-21 20:06

- Phase: Phase 3 (ME selection)
- Commit: `27fec44a4aad2109abc9c5e4816d4e3c1f12288a`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC`
- Axes: `me` ∈ {pattern, hierarchical}
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Results: `validation/results/gate3-me_27fec44a`

Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.

| variant | PCC_mean | PCC_lo_mean | PCC_hi_mean | SRCC_mean | perseq_PCC_mean | fps |
|---|---|---|---|---|---|---|
| me=hierarchical | 0.6992 | 0.6760 | 0.7257 | 0.6851 | 0.4230 | 111.8881 |
| me=pattern | 0.6876 | 0.6584 | 0.7230 | 0.7150 | 0.3984 | 238.8060 |


### Ablation `gate3-me-variants` — 2026-08-21 20:15

- Phase: Phase 3 (ME variant selection)
- Commit: `550620fe64e79c0127413aefecf491caed232538`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC`
- Variants: `pattern (iter4)` = `--me pattern`; `hier` = `--me hierarchical`; `hier+halfpel` = `--me hierarchical --me-subpel 1`; `hier+quarterpel` = `--me hierarchical --me-subpel 2`; `hier+gmv` = `--me hierarchical --me-predictor global`; `hier+lambda0.5` = `--me hierarchical --me-lambda 0.5`; `hier+lambda2` = `--me hierarchical --me-lambda 2`; `hier+merge` = `--me hierarchical --me-merge`; `hier+satd` = `--me hierarchical --me-criterion satd`; `hier+halfpel+gmv` = `--me hierarchical --me-subpel 1 --me-predictor global`
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Results: `validation/results/gate3-me-variants_9d25a08c`

Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.

| variant | PCC_mean | PCC_lo_mean | PCC_hi_mean | SRCC_mean | perseq_PCC_mean | fps |
|---|---|---|---|---|---|---|
| hier+merge | 0.7775 | 0.7535 | 0.7996 | 0.7721 | 0.4688 | 73.6196 |
| hier+satd | 0.7524 | 0.7313 | 0.7746 | 0.7441 | 0.4433 | 35.8744 |
| hier+lambda2 | 0.7522 | 0.7265 | 0.7795 | 0.6853 | 0.3534 | 103.4483 |
| hier+halfpel | 0.7190 | 0.6954 | 0.7452 | 0.6876 | 0.1569 | 59.8504 |
| hier+quarterpel | 0.7197 | 0.6954 | 0.7459 | 0.6923 | 0.1914 | 41.4150 |
| hier+halfpel+gmv | 0.7181 | 0.6944 | 0.7443 | 0.6867 | 0.1556 | 47.6663 |
| hier+lambda0.5 | 0.7058 | 0.6823 | 0.7314 | 0.7006 | 0.4037 | 104.8035 |
| hier | 0.6992 | 0.6760 | 0.7257 | 0.6851 | 0.4230 | 103.4483 |
| hier+gmv | 0.6983 | 0.6751 | 0.7248 | 0.6844 | 0.4227 | 80.0000 |
| pattern (iter4) | 0.6876 | 0.6584 | 0.7230 | 0.7150 | 0.3984 | 202.5316 |


## Phase 3 — motion estimation

Implemented, each as its own commit: hierarchical pyramid search with full-resolution
integer refinement (`add95ab`), half/quarter-pel refinement (`3ae2df8`), and the global
predictor, MV cost, merge pass and SATD criterion (`27fec44`).

A robustness bug surfaced while validating the pyramid: with a frame height that is not
a multiple of the block size (1080 is not — the loader crops to 1056), each level
produced a different block-grid size, so the upsampled coarse field landed on the wrong
blocks and left ~4 px of error. Every level is now cropped to a whole number of blocks;
`tests/test_hierarchical_me.py::test_block_grid_alignment_with_non_multiple_height`
pins it.

### Endpoint error on synthetic translations (1056×1920, interior blocks)

| true shift | sparse pattern | hierarchical |
|---|---|---|
| 1 px | 1.000 | **0.000** |
| 2 px | 0.000 | **0.000** |
| 3 px | 1.000 | **0.000** |
| 5 px | 1.000 | **0.000** |
| 8 px | 2.000 | **0.000** |
| 16 px | 16.090 | **0.000** |
| 32 px | 31.567 | **0.000** |
| **mean** | **7.522** | **0.000** |

Also 0.000 for the diagonal and vertical cases (4,−4), (−8,8), (16,16), (−32,0), (3,7).
Half-pel translations: mean EPE falls from 0.500 (integer) to **0.064** with
`--me-subpel 1`; the worst single case (a true 1.5 px shift) is 0.255. Both meet the
acceptance thresholds of ≤ 0.5 px integer and ≤ 0.25 px half-pel (stated as a mean over
the translation set). Phase correlation recovers the global shift exactly on every
tested translation up to ±40 px.

### `MV_sat_frac` on the real sequences (fast subset, frames ≥ 1)

| sequence | pattern | hierarchical | `mean_mv_mag` pattern → hier | `intra_frac` pattern → hier |
|---|---|---|---|---|
| YachtRide | 63.0 % | **0.3 %** | 4.62 → 7.67 | 36.8 % → 18.0 % |
| ReadySteadyGo | 50.5 % | **0.2 %** | 4.18 → 6.54 | 20.7 % → 19.5 % |
| Bosphorus | 2.8 % | **0.1 %** | 1.96 → 3.79 | 13.6 % → 14.5 % |
| HoneyBee | 0.2 % | **0.0 %** | 0.08 → 0.17 | 7.0 % → 4.2 % |
| **overall** | **29.13 %** | **0.15 %** | | 19.5 % → 14.1 % |

The acceptance threshold of < 5 % is met with a large margin (0.15 %). The jump in
`mean_mv_mag` on YachtRide from 4.62 to 7.67 px confirms the pattern search was
clipping true motion at its ±6 px boundary rather than measuring it.

### Gate 3 ablation (fast subset, ranking metric `TC_MC`, values averaged over QPs)

| variant | PCC | **CI lo** | per-seq PCC | fps |
|---|---|---|---|---|
| `hier+merge` | 0.7775 | **0.7535** | **0.4688** | 73.6 |
| `hier+satd` | 0.7524 | 0.7313 | 0.4433 | 35.9 |
| `hier+lambda2` | 0.7522 | 0.7265 | 0.3534 | 103.4 |
| `hier+quarterpel` | 0.7197 | 0.6954 | 0.1914 | 41.4 |
| `hier+halfpel` | 0.7190 | 0.6954 | 0.1569 | 59.9 |
| `hier+halfpel+gmv` | 0.7181 | 0.6944 | 0.1556 | 47.7 |
| `hier+lambda0.5` | 0.7058 | 0.6823 | 0.4037 | 104.8 |
| `hier` | 0.6992 | 0.6760 | 0.4230 | 103.4 |
| `hier+gmv` | 0.6983 | 0.6751 | 0.4228 | 80.0 |
| `pattern` (iter4) | 0.6876 | 0.6584 | 0.3984 | 202.5 |

Four things stand out.

**Every hierarchical variant beats the sparse pattern.** Even plain `hier` has a CI
lower bound of 0.6760 against the pattern's 0.6584, and its per-sequence PCC is higher
too (0.4230 vs 0.3984). The margin for plain `hier` alone is modest and the intervals
overlap, but combined with the saturation collapse above, the pyramid is clearly the
better search.

**The merge pass is the single most valuable addition** and is the only option that
improves the pooled and the within-sequence statistic together (0.6992 → 0.7775 pooled,
0.4230 → 0.4688 per-sequence). Re-testing each block against its neighbours' vectors
mostly repairs blocks whose own SAD minimum was ambiguous, which is exactly where a
block-matching field is least trustworthy.

**Sub-pel refinement helps the pooled statistic but badly hurts the within-sequence one**
(per-sequence PCC 0.423 → 0.157 at half-pel). Half-pel vectors make the compensated
residual smaller everywhere, which sharpens the between-sequence ordering, but the
bilinear interpolation also low-pass filters the prediction, and the resulting residual
energy stops tracking frame-to-frame variation. On this corpus sub-pel is not worth its
cost, and it is the clearest case in the whole study where the two statistics disagree.

**The global predictor is worth nothing here** (0.6992 → 0.6983, and 80 fps instead of
103). That is the expected result once `MV_sat_frac` is 0.15 %: the pyramid already
reaches the true motion on every block, so seeding it with a global vector adds a second
search for no gain. It would matter for content with motion beyond the pyramid's reach,
which this corpus does not contain.

**Throughput note.** `hier+merge` runs at 73.6 fps at 1080p on the RTX 5060 Ti, which is
**below the 100 fps acceptance target**; plain `hier` and `hier+lambda2` clear it at
103 fps, and the Iteration-4 pattern search runs at 202 fps. This is recorded as a
missed criterion, not silently accepted — see the Gate 3 decision below.

### Ablation `gate3-me-combos` — 2026-08-21 20:18

- Phase: Phase 3 (ME combination round)
- Commit: `a88ea3ce29bab031e72e3fd69da895889dd26342`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC`
- Variants: `hier+merge` = `--me hierarchical --me-merge`; `hier+merge+lambda0.5` = `--me hierarchical --me-merge --me-lambda 0.5`; `hier+merge+lambda2` = `--me hierarchical --me-merge --me-lambda 2`; `hier+merge+satd` = `--me hierarchical --me-merge --me-criterion satd`; `hier+merge+halfpel` = `--me hierarchical --me-merge --me-subpel 1`; `hier+merge+gmv` = `--me hierarchical --me-merge --me-predictor global`
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Results: `validation/results/gate3-me-combos_550620fe`

Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.

| variant | PCC_mean | PCC_lo_mean | PCC_hi_mean | SRCC_mean | perseq_PCC_mean | fps |
|---|---|---|---|---|---|---|
| hier+merge+halfpel | 0.8023 | 0.7801 | 0.8218 | 0.7666 | 0.2586 | 47.6190 |
| hier+merge+lambda2 | 0.7987 | 0.7736 | 0.8217 | 0.7570 | 0.3928 | 73.1707 |
| hier+merge+satd | 0.7846 | 0.7610 | 0.8061 | 0.7729 | 0.4854 | 28.6055 |
| hier+merge | 0.7775 | 0.7535 | 0.7996 | 0.7721 | 0.4688 | 72.6172 |
| hier+merge+gmv | 0.7769 | 0.7529 | 0.7989 | 0.7719 | 0.4692 | 60.6061 |
| hier+merge+lambda0.5 | 0.7629 | 0.7369 | 0.7865 | 0.7708 | 0.4492 | 74.0741 |


### Other metrics under the new search (fast subset, averaged over QPs)

The ablation ranks on `TC_MC`, but the search change moves the other temporal metrics
too. Recomputed from the stored per-frame CSVs of the Gate 3 run:

| metric | pattern (iter4) | `hier` | `hier+merge` |
|---|---|---|---|
| `TC_SAD` PCC / CI lo / per-seq | 0.5823 / 0.5271 / 0.2311 | 0.7248 / 0.6994 / 0.4383 | **0.7439 / 0.7163 / 0.4404** |
| `TC_MC` PCC / CI lo / per-seq | 0.6876 / 0.6582 / 0.3984 | 0.6992 / 0.6759 / 0.4230 | **0.7775 / 0.7542 / 0.4688** |
| `TC_SAD_full` PCC / CI lo / per-seq | 0.6050 / 0.5528 / 0.2679 | 0.6631 / 0.6311 / 0.4096 | **0.8226 / 0.8037 / 0.5042** |
| `MVC` PCC / CI lo / per-seq | **0.8787** / 0.8572 / 0.1153 | 0.5348 / 0.4874 / 0.0102 | 0.5631 / 0.5168 / 0.0312 |

The Phase 3 acceptance criterion "frame-level PCC of `TC_SAD` and `TC_MC` not lower than
Gate 2" is met with a wide margin for both (`TC_SAD` 0.58 → 0.74, `TC_MC` 0.69 → 0.78).

Two results here matter beyond the gate itself.

**`TC_SAD_full` is the strongest temporal predictor in the whole study** once the search
is accurate: pooled PCC 0.8226 with a CI lower bound of 0.8037, and a within-sequence
PCC of 0.5042 — better than `TC_MC` on every statistic. It is also the cheapest to
compute, being just the mean absolute motion-compensated residual with no transform.
This column was added under Phase 4.5 and was not part of the original metric set.

**`MVC`'s apparent strength was an artefact of search failure.** Under the pattern
search it had by far the highest pooled PCC (0.8787); under the pyramid it collapses to
0.5348. Its within-sequence PCC was near zero throughout (0.115 → 0.010). What the old
`MVC` was measuring was the chaos of a search that could not reach the true motion, and
that chaos happened to track which sequences were expensive to encode. With a search
that succeeds, the MV field is smooth and `MVC` no longer proxies for motion magnitude.
This is the concrete explanation for the Gate 1 observation that `MVC` scored well
pooled and near-zero within sequences, and it is a reason to distrust the metric.

### Gate 3 — combination round and decision

Since the merge pass was the strongest single addition, a second round crossed it with
the other options (fast subset, ranking metric `TC_MC`, averaged over QPs):

| variant | PCC | **CI lo** | per-seq PCC | fps |
|---|---|---|---|---|
| `hier+merge+halfpel` | 0.8023 | **0.7801** | 0.2586 | 47.6 |
| `hier+merge+lambda2` | 0.7987 | 0.7736 | 0.3928 | 73.2 |
| `hier+merge+satd` | 0.7846 | 0.7610 | **0.4854** | 28.6 |
| `hier+merge` | 0.7775 | 0.7535 | 0.4688 | 72.6 |
| `hier+merge+gmv` | 0.7769 | 0.7529 | 0.4692 | 60.6 |
| `hier+merge+lambda0.5` | 0.7629 | 0.7369 | 0.4492 | 74.1 |

**Gate 3 choice, by the rule as written** (highest lower bound of the 95 % CI of pooled
frame-level PCC of `TC_MC`): **`--me hierarchical --me-merge --me-subpel 1`**, CI lower
bound 0.7801. Recorded; no default is changed at this gate.

Two acceptance criteria are **not** met by that choice, and both are recorded rather
than waved through:

**Throughput.** `hier+merge+halfpel` runs at 47.6 fps at 1080p on the RTX 5060 Ti,
against the ≥ 100 fps criterion. Of the strong variants only `hier+lambda2` (103 fps)
and plain `hier` (103 fps) clear it; `hier+merge` reaches 72.6 fps. The criterion is
missed by the winner and by every merge-based variant.

**The ranking statistic and the per-sequence statistic disagree sharply, and the
per-sequence breakdown shows why.** Per-sequence PCC of `TC_MC`:

| sequence | `hier+merge` | `+halfpel` | `+satd` | `+lambda2` |
|---|---|---|---|---|
| YachtRide | 0.856 | 0.821 | 0.856 | 0.853 |
| HoneyBee | 0.565 | **−0.455** | 0.595 | 0.528 |
| ReadySteadyGo | 0.284 | 0.605 | 0.308 | 0.182 |
| Bosphorus | 0.170 | 0.063 | 0.183 | 0.008 |

The half-pel collapse is almost entirely HoneyBee, where the correlation **flips sign**.
HoneyBee is the near-static sequence (`mean_mv_mag` 0.17 px). Half-pel motion
compensation there resamples an essentially unmoved frame through a bilinear filter, so
the prediction is slightly low-pass filtered and the residual starts measuring the
scene's high-frequency detail rather than its temporal change — and in a static scene
that detail is cheap to code, hence the negative correlation. Sub-pel refinement is
therefore actively harmful on low-motion content while helping high-motion content
(ReadySteadyGo 0.284 → 0.605). The pooled statistic cannot see this because it is
dominated by the between-sequence spread.

`hier+merge+satd` is the best variant on the per-sequence statistic (0.4854) and second
on the gate statistic, but it is the slowest at 28.6 fps.

**Consequence for Phase 4.** The MC matrix is run with ME frozen at the Gate-3 choice as
specified, and the leading MC candidates are additionally re-run at `hier+merge`
(no sub-pel), so that the Gate-4 default decision can be taken with both the pooled and
the within-sequence evidence in view rather than inheriting a contested ME setting.

## Phase 5 — global motion and structural features

All features verified on synthetics before being correlated on real content
(`tests/test_motion_features.py`, 17 tests):

| synthetic | expected | measured |
|---|---|---|
| pan (0, 8) | GMV = true shift, div ≈ 0, curl ≈ 0, coherence ≈ 1 | GMV = (0.00, 8.00), div 0.0000, curl 0.0000, coherence 1.00 |
| pan (−4, 0) | as above | GMV = (−4.00, 0.00), div/curl 0.0000, coherence 1.00 |
| zoom in (rate 0.02) | `MV_div` > 0 | div **+1.182**, curl 0.010 |
| rotation (1°/frame) | `MV_curl` ≠ 0 | curl **+1.067**, div 0.011 |
| static + noise | zero motion, coherent | GMV magnitude 0.00, coherence 1.00 |
| hard cut | `intra_frac` spikes | `intra_frac` 1.00 on the cut frame, < 0.1 elsewhere |

### Univariate, frame level vs `TC_gt` (ME `hier+merge`, fast subset, mean over QPs)

| feature | pooled PCC | CI lo | **per-seq PCC** | feature | pooled PCC | CI lo | **per-seq PCC** |
|---|---|---|---|---|---|---|---|
| `MV_coherence` | **−0.934** | −0.946 | −0.142 | `MVC` | 0.563 | 0.519 | 0.031 |
| `aff_a21` | 0.873 | 0.854 | 0.112 | `GMV_x` | 0.536 | 0.480 | −0.296 |
| `GMV_mag` | 0.861 | 0.837 | −0.047 | `TC2` | 0.534 | 0.494 | 0.419 |
| `TC_SAD_full` | 0.823 | 0.804 | **0.504** | `TC` | 0.506 | 0.469 | 0.258 |
| `mean_mv_mag` | 0.815 | 0.786 | −0.017 | `MV_sat_frac` | 0.467 | 0.406 | 0.051 |
| `TC_MC` | 0.777 | 0.754 | **0.469** | `MVD_cost` | 0.423 | 0.363 | 0.029 |
| `TC_SAD` | 0.744 | 0.716 | **0.440** | `skip_frac` | 0.406 | 0.350 | −0.176 |
| `MV_curl` | −0.733 | −0.767 | −0.007 | `SC` | 0.294 | 0.257 | 0.049 |
| `intra_frac` | 0.693 | 0.660 | 0.112 | `MV_div` | 0.090 | 0.020 | −0.073 |

The split is stark and systematic. Every new structural feature scores high **pooled**
and essentially zero **within sequence**. `MV_coherence` reaches |PCC| 0.934 pooled — the
strongest number anywhere in this study — and −0.142 within sequences. These features
describe *what kind of content a sequence is* (how much global motion it has, how rigid
it is), which separates sequences by bitrate beautifully and says almost nothing about
which frame of a given sequence is expensive. Only the residual-derived metrics
(`TC_SAD_full`, `TC_MC`, `TC_SAD`, `TC2`) carry real within-sequence signal.

### Multivariate: ridge with leave-one-sequence-out

Target `log(bits)`. `R2` is the raw held-out score; `R2_centered` removes the held-out
sequence's mean first, so it scores only frame-to-frame variation. Mean over the four
held-out sequences and the four QPs, at the best penalty for each model:

| model | LOSO PCC | LOSO R²(centered) |
|---|---|---|
| `TC_SAD_full` alone | **0.496** | −1.55 |
| `TC2` alone | 0.478 | −14.55 |
| `TC_MC` alone | 0.456 | −0.67 |
| `TC_SAD` alone | 0.435 | −3.82 |
| **all 8 features (ridge)** | **0.302** | −8.36 |
| `TC_SAD_full` + `TC_MC` | 0.258 | −9.27 |
| `TC_MC` + `TC_SAD` | 0.276 | −9.39 |
| `MV_coherence` alone | 0.132 | −9.81 |
| `GMV_mag` alone | −0.016 | −16.52 |

**The multivariate model is worse than its best single input, and so is every two-feature
combination.** This holds across ridge penalties from 1 to 10⁴ (the 8-feature LOSO PCC
peaks at 0.302 for α = 10 and falls either side; centered R² only reaches zero at
α = 10⁴, where the model has been shrunk to a constant). The cause is visible in the
univariate table: features like `GMV_mag` and `MV_coherence` separate the three training
sequences almost perfectly, so the fit assigns them large weights, and those weights then
mispredict a held-out sequence with different content. With **only four sequences**,
leave-one-sequence-out simply cannot support a multivariate fit — there are three
training groups and eight predictors that are near-collinear at the group level.

Raw (uncentered) R² is strongly negative for every model, i.e. none of them predicts a
held-out sequence's absolute bitrate. That is expected and not very interesting: an ABR
ladder is fitted per title anyway, so the useful question is the within-sequence one,
which the centered R² and the LOSO PCC answer.

**Gate 5 recorded.** These columns are additive outputs and change no defaults. The
practical recommendation from this phase is the opposite of "add more features": on this
corpus the single strongest per-frame predictor is `TC_SAD_full`, and combining features
degrades cross-sequence generalisation. A corpus of ~20+ sequences would be needed before
a multivariate model can be evaluated meaningfully; that is recorded as an open issue.
