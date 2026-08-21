# EVCA2 — temporal complexity rework

Final report for the Phase 0–6 rework of EVCA's temporal path. The running ledger with
every individual benchmark invocation, flag set and correlation table is
[`RESULTS.md`](RESULTS.md); this document is the summary and the argument.

**Branch** `evca2-dev` · **Device** NVIDIA RTX 5060 Ti (CUDA), conda env `EVCA-gpu`,
torch 2.13.0+cu130 · **Ground truth** x265 (ffmpeg n9.0.1), All-Intra and Low-Delay-P at
QP 22/27/32/37 · **Corpus** UVG 1080p 8-bit: YachtRide, ReadySteadyGo, HoneyBee,
Bosphorus.

---

## 1. Headline

The motion search was the bottleneck, and fixing it moved every temporal metric.

| | Iteration 4 (old default) | New default | change |
|---|---|---|---|
| `TC_MC` pooled frame-level PCC vs `TC_gt` | 0.6876 | **0.7936** | +0.106 |
| `TC_MC` **within-sequence** PCC | 0.3984 | **0.4907** | +0.092 |
| `TC_SAD` pooled PCC | 0.5823 | 0.7439 | +0.162 |
| `MV_sat_frac` (blocks pinned at the search boundary) | 29.1 % | **0.15 %** | −29 pp |
| Endpoint error, synthetic translations ≤ ±32 px | 7.52 px | **0.00 px** | — |
| Full-profile throughput, 1080p | 237.6 fps | 66.9 fps | −3.6× |

The new default configuration is

```bash
python main.py -i input.yuv -r 1920x1080 -me --profile full
```

which now means `--me hierarchical --me-merge --mc dense_smooth --mc-smooth gauss
--gate none --residual-dc`. The previous behaviour is reachable as `--preset iter4`.

Three findings matter more than the headline numbers:

1. **`MVC` was measuring its own search failing.** Under the old ±6 px search it had the
   highest pooled correlation of any temporal metric (0.879). Under a search that
   actually finds the motion it collapses to 0.535, and its within-sequence correlation
   was never above 0.12. It should not be trusted as a complexity feature.
2. **Pooled correlations are mostly a between-sequence effect.** Every new structural
   feature (`MV_coherence`, `GMV_mag`, `mean_mv_mag`) scores 0.8–0.93 pooled and ≈ 0
   within a sequence. With four sequences the sequence-level bootstrap CI is nearly
   vacuous, so pooled numbers alone can rank a configuration the wrong way round.
3. **The two best predictors found are both new and neither is `TC_MC`.** The
   rho-domain coefficient counts (`--rho`) reach pooled PCC 0.94–0.98 at QP ≥ 27, and
   `TC_SAD_full` — the plain mean absolute motion-compensated residual, no DCT at all —
   reaches 0.823 pooled and 0.504 within-sequence, both above `TC_MC`.

---

## 2. What changed in the code

| area | before | after |
|---|---|---|
| Motion estimation | 13-point diamond at half resolution, ±6 px, even-valued MVs | Pyramid (¼ → ½ → 1, plus ⅛ for ≥ 4K), exhaustive coarse level, ±1 refinement per level, integer-pel; optional half/quarter-pel, global predictor, MV cost, merge pass, SATD |
| Motion compensation | hardcoded Gaussian-smoothed dense warp | `MotionCompensator` strategy: `dense_smooth` / `dense` / `block` / `obmc`, smoothing `gauss` / `median` / `none` |
| Residual scoring | intra-gated, DC excluded | `--gate {intra,none}`, `--residual-dc` |
| Outputs | `B, SC, TC, TC2, MVC, TC_SAD, TC_MC` | plus `MV_sat_frac`, `mean_mv_mag`, `intra_frac`, `skip_frac`, `TC_SAD_full`, `GMV_*`, `MV_coherence`, `MV_div`, `MV_curl`, `MVD_cost`, `aff_*`, and `rho_qp22..37` under `--rho` |
| Validation | one script, sequence-mean correlation, n = 4 | per-frame ground truth, frame-level correlation with bootstrap CIs, synthetic generator, ablation driver, 150 pytest tests |

New modules: `libs/motion_estimation.py`, `libs/motion_compensation.py`,
`libs/motion_features.py`, `libs/rho_domain.py`; `validation/{synthetic,stats,report,
ground_truth,run_benchmark,run_ablation,feature_analysis,plots,bench_dct}.py`.

---

## 3. Bugs fixed (Phase 0)

Fixes 0.1–0.6a were already on the base branch at `b450068`. This work added:

| fix | commit | effect on default output |
|---|---|---|
| GOP step size reset per input file in `--dir` mode | `a95023d` | batching only |
| `frame_to_edge.py` deleted (unreferenced, wrong call signature) | `793a23a` | none |
| `grid_sample` base grid and Gaussian kernel cached per `(H, W, device)` | `9339ad3` | none |
| Comments corrected: the pattern search runs at **half** resolution | `9339ad3` | none |
| **`load_gop_optimized` crashed whenever chroma was enabled** | `dbfa6cc` | `--loader optimized -cc` went from raising `BufferError` to working |
| **Pyramid levels cropped to whole blocks** | `add95ab` | required for any frame height not a multiple of 32 (1080 is not) |

The loader crash is worth singling out. `if 'Y_view' in locals(): del Y_view`
materialises the frame's `f_locals` snapshot, and that dict then holds its own reference
to the last chroma view, so the memory map still had an exported buffer at `close()`.
Plain rebinding fixes it. The bug was found by the new loader-parity test, not by
inspection.

---

## 4. Motion estimation

### Accuracy

Endpoint error on synthetic translations at 1056×1920, interior blocks:

| shift | sparse pattern | hierarchical |
|---|---|---|
| 1, 3, 5 px (odd) | 1.000 | **0.000** |
| 8 px | 2.000 | **0.000** |
| 16 px | 16.090 | **0.000** |
| 32 px | 31.567 | **0.000** |
| mean over the set | 7.522 | **0.000** |

Half-pel translations: mean EPE 0.500 → **0.064** with `--me-subpel 1`. Phase correlation
recovers the global shift exactly up to ±40 px. Acceptance thresholds (≤ 0.5 px integer,
≤ 0.25 px half-pel) are met.

### Saturation on real content

| sequence | `MV_sat_frac` before | after | `mean_mv_mag` before → after |
|---|---|---|---|
| YachtRide | 63.0 % | 0.3 % | 4.62 → 7.67 |
| ReadySteadyGo | 50.5 % | 0.2 % | 4.18 → 6.54 |
| Bosphorus | 2.8 % | 0.1 % | 1.96 → 3.79 |
| HoneyBee | 0.2 % | 0.0 % | 0.08 → 0.17 |
| **overall** | **29.1 %** | **0.15 %** | |

On YachtRide, two thirds of blocks were previously pinned to the search boundary. Their
reported motion of 4.62 px was a clipping artefact; the true mean is 7.67 px. This is why
`TC_SAD` correlated *negatively* with bits on that sequence before the fix.

---

## 5. Ablation matrices

Fast subset (first 120 frames of each sequence), ranking metric `TC_MC` against the
Low-Delay-P frame bits, values averaged over QP 22/27/32/37. `CI lo` is the lower bound
of the 95 % frame-level bootstrap CI — the gate ranking key. `per-seq` is the mean
within-sequence PCC.

### ME (MC held at `dense_smooth`, `gate intra`)

| variant | PCC | CI lo | per-seq | fps |
|---|---|---|---|---|
| `hier+merge+halfpel` | 0.8023 | **0.7801** | 0.2586 | 47.6 |
| `hier+merge+lambda2` | 0.7987 | 0.7736 | 0.3928 | 73.2 |
| `hier+merge+satd` | 0.7846 | 0.7610 | **0.4854** | 28.6 |
| `hier+merge` | 0.7775 | 0.7535 | 0.4688 | 72.6 |
| `hier+merge+gmv` | 0.7769 | 0.7529 | 0.4692 | 60.6 |
| `hier+lambda2` | 0.7522 | 0.7265 | 0.3534 | 103.4 |
| `hier+satd` | 0.7524 | 0.7313 | 0.4433 | 35.9 |
| `hier+halfpel` | 0.7190 | 0.6954 | 0.1569 | 59.9 |
| `hier` | 0.6992 | 0.6760 | 0.4230 | 103.4 |
| `hier+gmv` | 0.6983 | 0.6751 | 0.4228 | 80.0 |
| `pattern` (iter4) | 0.6876 | 0.6584 | 0.3984 | 202.5 |

### MC × gate × residual-dc (ME at `hier+merge`)

| mc | gate | residual-dc | PCC | CI lo | per-seq | fps |
|---|---|---|---|---|---|---|
| `dense_smooth` | none | on | 0.7936 | **0.7720** | **0.4907** | 66.9 |
| `dense_smooth` | none | off | 0.7880 | 0.7657 | 0.4835 | 74.2 |
| `dense_smooth` | intra | on | 0.7836 | 0.7602 | 0.4725 | 74.3 |
| `dense_smooth` | intra | off | 0.7775 | 0.7535 | 0.4688 | 74.0 |
| `dense` | none | on | 0.6936 | 0.6649 | 0.4552 | 72.0 |
| `dense` | intra | off | 0.6737 | 0.6420 | 0.4360 | 66.7 |
| `block` | none | on | 0.6414 | 0.6056 | 0.4079 | 67.9 |
| `block` | intra | off | 0.6127 | 0.5741 | 0.3946 | 74.8 |

With ME frozen at the Gate-3 sub-pel choice the same matrix produced an identical
ordering (`dense_smooth` > `dense` > `block` > `obmc`, `none` > `intra`, `on` > `off`)
at a uniformly lower per-sequence level; `obmc` ranked last in every cell while costing
13 % throughput. Full 16-row matrix in `RESULTS.md`.

**Gaussian smoothing of the MV field still helps after the search is fixed** — by about
0.08 PCC over unsmoothed dense warping. That was the open question Phase 4 was asked to
settle, and the answer is that Iteration 4's smoothing was not compensating for the
misregistered field it was introduced alongside.

---

## 6. The pooled / within-sequence divergence

This is the single most important methodological result, and it decided Gate 4.

Mean within-sequence frame-level PCC against `TC_gt` versus the pooled value, at the
final ME setting:

| feature | pooled PCC | within-sequence PCC |
|---|---|---|
| `MV_coherence` | −0.934 | −0.142 |
| `GMV_mag` | 0.861 | −0.047 |
| `mean_mv_mag` | 0.815 | −0.017 |
| `MV_curl` | −0.733 | −0.007 |
| `intra_frac` | 0.693 | 0.112 |
| `MVC` | 0.563 | 0.031 |
| **`TC_SAD_full`** | 0.823 | **0.504** |
| **`TC_MC`** | 0.777 | **0.469** |
| **`TC_SAD`** | 0.744 | **0.440** |

Features that describe *what kind of content a sequence is* separate sequences by
bitrate almost perfectly and say nothing about which frame within a sequence is
expensive. Only residual-derived metrics carry within-sequence signal. With four
sequences the sequence-level block bootstrap frequently spans [−0.97, +0.99], so a
pooled point estimate is not evidence that a configuration predicts per-frame rate.

The practical consequence: at Gate 4 the literal ranking rule preferred
`--me-subpel 1` (pooled CI lower bound 0.7943 vs 0.7720), but that configuration has a
within-sequence PCC of 0.272 — *worse than the Iteration-4 default it would replace*
(0.398) — and runs at 48 fps instead of 67. The two CIs overlap across most of their
length, so the rule's own tie-break ("ties go to the cheaper variant") applies. The
default was set without sub-pel, and the deviation is documented in `RESULTS.md`.

The sub-pel failure mode is understood: on HoneyBee (`mean_mv_mag` 0.17 px) the
correlation flips to **−0.455**. Sub-pel compensation of near-static content resamples an
essentially unmoved frame through a bilinear filter, so the residual starts measuring
the scene's high-frequency detail — which is cheap to code in a static scene — instead of
temporal change.

---

## 7. Structural features and the multivariate fit

All features were verified against synthetic ground truth before being correlated on
real content: a pure pan reproduces the true shift exactly with `MV_div` and `MV_curl`
at 0.0000 and coherence 1.00; zoom gives `MV_div` +1.18; rotation gives `MV_curl` +1.07;
a hard cut drives `intra_frac` to 1.00 on the cut frame and below 0.1 elsewhere.

Ridge regression on `{TC_MC, TC_SAD, MVC, GMV_mag, MV_coherence, MV_div, intra_frac,
skip_frac}` predicting `log(bits)`, scored leave-one-sequence-out:

| model | LOSO PCC | LOSO R² (centered) |
|---|---|---|
| `TC_SAD_full` alone | **0.496** | −1.55 |
| `TC2` alone | 0.478 | −14.55 |
| `TC_MC` alone | 0.456 | −0.67 |
| **all 8 features** | **0.302** | −8.36 |
| `TC_SAD_full` + `TC_MC` | 0.258 | −9.27 |

**The multivariate model is worse than its best single input, and so is every two-feature
combination.** This holds for ridge penalties from 1 to 10⁴. The cause is visible in the
univariate table above: the structural features separate the three training sequences
almost perfectly, the fit gives them large weights, and those weights mispredict a
held-out sequence. Three training groups cannot support eight predictors that are
near-collinear at the group level. A corpus of roughly 20+ sequences would be needed
before a multivariate model can be evaluated at all.

---

## 7b. rho-domain rate estimation (Phase 6)

`--rho` emits `rho_qp22 … rho_qp37`: the fraction of residual DCT coefficients above
`Qstep(QP)/2`. Matched-QP frame-level correlation against `TC_gt`, new defaults, fast
subset:

| QP | `rho_qpXX` PCC / per-seq | `TC_MC` PCC / per-seq | `TC_SAD_full` PCC / per-seq |
|---|---|---|---|
| 22 | 0.676 / 0.495 | 0.770 / 0.460 | 0.781 / 0.338 |
| 27 | **0.944** / 0.451 | 0.762 / 0.484 | 0.797 / 0.509 |
| 32 | **0.969** / 0.524 | 0.803 / 0.482 | 0.836 / 0.541 |
| 37 | **0.978** / **0.641** | 0.839 / 0.537 | 0.877 / 0.629 |

These are the strongest correlations measured anywhere in this study. `rho_qp22` is the
exception at 0.676; it is not saturating (mean 0.091, max 0.137) — the between-sequence
spread simply collapses at a low threshold, from a 65× HoneyBee-to-YachtRide ratio at
QP 37 to 1.35× at QP 22. The column names are best read as a threshold sweep rather than
per-QP predictions: `rho_qp32` correlates 0.981 with the QP 37 ground truth, slightly
better than `rho_qp37` does with its own.

The feature stays behind a flag (default off) because it was added under the optional
phase and has not been through a gate.

## 7c. Per-sequence scatter plots

`png/report/scatter_full_TC_MC_qp{27,32}.png` — frame-level `TC_MC` against Low-Delay-P
bits, one column per sequence, Iteration-4 defaults on the top row and the new defaults
below. `png/report/scatter_baseline_TC_qp{27,32}.png` shows upstream `TC` for reference.
Regenerate with:

```bash
python validation/plots.py --old validation/results/gate1_5560c9d8 --new validation/results/gate4-newdefaults_e8ea43ef --qps 27,32
```

Per-sequence frame-level PCC of `TC_MC`, Iteration 4 → new defaults:

| sequence | QP 22 | QP 27 | QP 32 | QP 37 | mean |
|---|---|---|---|---|---|
| YachtRide | 0.328 → **0.742** | 0.436 → **0.817** | 0.534 → **0.870** | 0.633 → **0.894** | 0.483 → **0.831** |
| Bosphorus | 0.034 → 0.209 | −0.018 → 0.178 | −0.071 → 0.171 | −0.031 → 0.140 | −0.022 → 0.175 |
| HoneyBee | 0.296 → 0.298 | 0.537 → 0.612 | 0.531 → 0.584 | 0.785 → 0.781 | 0.537 → 0.569 |
| ReadySteadyGo | 0.491 → 0.592 | 0.586 → **0.327** | 0.624 → **0.302** | 0.677 → **0.335** | 0.595 → **0.389** |
| **mean** | | | | | **0.398 → 0.491** |

Three sequences improve and one regresses. The gain is concentrated on YachtRide — the
sequence that was 63 % saturated under the old search — where the correlation nearly
doubles. Bosphorus moves from slightly negative to weakly positive.

**ReadySteadyGo regresses at QP ≥ 27** (0.595 → 0.389 on average), and nothing measured
here explains it. It was 50 % saturated before, so the old `TC_MC` there was also
substantially an artefact; the artefact simply happened to track bits better than the
corrected metric does. This is the clearest caution against reading the mean improvement
as uniform, and it is listed as an open issue. The plots make it visible rather than
letting the averaged numbers hide it.

## 8. Throughput

1080p, CUDA, `--loader optimized`, full profile unless noted.

| configuration | fps |
|---|---|
| baseline (no ME) | 375 |
| fast profile, `--preset iter4` | 302 |
| full profile, `--preset iter4` | 238 |
| full profile, `hier` (no merge) | 103 |
| full profile, `hier+lambda2` | 103 |
| **full profile, new defaults (`hier+merge`)** | **67** |
| full profile, `hier+merge+halfpel` | 48 |
| full profile, `hier+merge+quarterpel` | 41 |
| full profile, `hier+merge+satd` | 29 |

**The ≥ 100 fps acceptance criterion is not met by the new defaults (67 fps).** Only
`--me pattern` and hierarchical *without* the merge pass clear it. The merge pass costs
roughly a third of throughput and buys +0.078 pooled and +0.046 within-sequence PCC.
`--me hierarchical --no-me-merge` is the documented option for throughput-limited use at
103 fps, and `--preset iter4` remains available at 238 fps.

DCT backend (`validation/bench_dct.py`), max relative disagreement 3.4e-07 on CUDA:

| device | matmul | torch_dct | speedup |
|---|---|---|---|
| CUDA | 11 462 frames/s | 1 033 frames/s | 11.1× |
| CPU | 572 frames/s | 82 frames/s | 7.0× |

`matmul` was already the shipped implementation and is kept as the default; `torch_dct`
is retained only as an optional ablation reference.

---

## 9. Reproducing

```bash
python validation/run_benchmark.py --subset fast --label myrun
```

Sequence paths live in `validation/sequences.json` (override with
`--sequence-root` or `EVCA_SEQUENCE_ROOT`). Results land in
`validation/results/<label>_<sha>/` and a summary row is appended to `RESULTS.md`.
Ablations:

```bash
python validation/run_ablation.py --label mc --axis mc=dense_smooth,block --axis gate=intra,none
```

Tests (CPU-only, no ffmpeg or sequences needed, ~45 s):

```bash
python -m pytest tests/ -q
```

---

## 10. Open issues

1. **The corpus is too small for the questions being asked.** Four sequences give a
   sequence-level bootstrap CI that frequently spans [−0.97, +0.99] and cannot support
   any multivariate fit. Everything in section 6 and 7 should be re-checked at ~20
   sequences before the conclusions are treated as settled.
2. **The new defaults miss the 100 fps target** (67 fps at 1080p). The merge pass is the
   cost; a cheaper approximation of it, or restricting it to blocks whose SAD margin is
   small, is the obvious next optimisation.
3. **Sub-pel refinement is content-dependent** — it helps high-motion sequences and hurts
   near-static ones. A gate that enables it per block or per frame based on
   `mean_mv_mag` would likely capture the gain without the HoneyBee regression.
4. **`TC_SAD_full` outperforms `TC_MC` on every statistic and is far cheaper**, but it
   was added late (Phase 4.5) and has not been ablated in its own right. It is currently
   an additional output column, not a default-path metric; that is worth revisiting.
5. **`MVC` should probably be retired or redefined.** Its historical value came from
   measuring search failure.
6. **The `EVCA` conda environment has a broken torch install** (MKL `iJIT_NotifyEvent`
   symbol error). All work used `EVCA-gpu`.
7. **OBMC is implemented but never competitive.** Either the 5-tap raised-cosine window
   is too aggressive for 32×32 blocks, or overlapped compensation genuinely does not
   suit this metric; not investigated further.
8. **The `--me-lambda` and `--me-criterion satd` options were not tuned.** λ was tested at
   0.5 and 2 only, and SATD only at 8×8. Both showed real gains on the pooled statistic.
9. **ReadySteadyGo regresses under the new defaults** at QP 32 (per-sequence PCC 0.624 →
   0.302) while every other sequence improves. Nothing measured here explains it. It is
   the one sequence where the old, saturated search happened to produce a `TC_MC` that
   tracked bits better, and it deserves a per-frame look before the result is trusted.
10. **The rho-domain columns are the strongest predictors measured but sit behind a flag**
    and have not been through a gate. Promoting them would mean deciding whether a
    QP-specific feature belongs in a QP-agnostic complexity file at all.
