# Learned Re-Planning v1: Threshold Sensitivity and State-Dependence Audit

## Verdict

**A. STATE-DEPENDENT RANKING PRESENT — CALIBRATION IS THE NEXT ISSUE**

The checkpoint's default-threshold retain-all result is primarily a decision-boundary/calibration failure, not an absence of learned relative ordering. Pass position explains most, but not all, of the beta variance (83.28%); the remaining 16.72% is measurable within position. Progress-only logit variation is comparable to dwell variation, the counterfactual time-only score has only 0.333 Spearman correlation with the full logit, and intermediate thresholds produce as many as 21 schedules. The ranking is exactly reproducible and has no strict float32-versus-float64 order reversals.

This does not establish that any threshold is optimal, nor that the adaptive policy improves task performance.

## Provenance and recovery

- Checkpoint: `/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_sanity/20260816T144122Z_adaptive_eta0_seed0_96step_sanity/checkpoints/step_96`
- Checkpoint-file SHA-256: `12a581d53efd1dc9335e6eca9e986186750528442759fc26ef6a7a5b7d1e0e28`
- Resolved config: `/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_sanity/20260816T144122Z_adaptive_eta0_seed0_96step_sanity/resolved_config.yaml`
- Resolved-config SHA-256: `f226fc0875777eafaf20851bcabc279a01caffa286cba864b9597ffaa2374b17`
- Loaded-model-state SHA-256: `a4971ef9fc45739acd99870bcdd70559842d79890488ae27e596140050b518a3`
- Held-out data/order digest: `e6b0a2d3aca095f2d76ef391d38e42ed39c5c7f9aefdb430bde9c463120af07f`
- Held-out episodes per threshold: 4,607
- Eligible decisions per threshold: 27,642 (exactly six per episode)
- Device: CPU; model explicitly in evaluation mode; deterministic hard decisions; no Bernoulli sampling.

The original tooling failure was at `/tmp/adaptive_threshold_audit.py:129`:

```python
array = tensor.detach().cpu().contiguous().numpy()
```

That expression received a bfloat16 state-dict tensor. The isolated audit-tool patch preserves deterministic ordering and hashes each entry's name, original PyTorch dtype, shape, and contents. Only bfloat16 contents are converted to float32 before NumPy byte serialization; the original `torch.bfloat16` identity remains in hash metadata. Checkpoint-file, loaded-state, and resolved-config hashes are deliberately separate.

Hash regression: 3 passed and 1 skipped. It covered float32, float16, bfloat16, integer and bool tensors; repeat determinism; sensitivity to one changed bfloat16 value; and bfloat16/float32 dtype distinction. The skipped test was accelerator-to-CPU logical hash equivalence because no CUDA/MPS device was available. Relevant adaptive tests: 23 passed. Ruff and `git diff --check`: passed.

The loaded-model-state hash matched before the sweep, after every one of the ten threshold calls, and after the deterministic repeat. The checkpoint and config file hashes also remained unchanged. The prior empty audit attempt is recorded in `provenance.json` as having completed zero scientific evaluations.

## Configuration isolation and invariants

The threshold grid, in order, was exactly:

`0.4984, 0.4985, 0.4986, 0.4987, 0.4988, 0.4989, 0.4990, 0.4991, 0.4992, 0.5000`

The prelaunch resolved-config comparison found only `arch.subgoal_head.trigger_threshold` changed; the 0.5000 call had no config difference. Across all evaluations: `M=8`; the pass-1 bootstrap emission was mandatory; passes 2–7 were the six eligible decisions; pass 8 never emitted; every episode's completed dwell lengths summed to 7; no model output or saved numeric diagnostic contained NaN/Inf; and no schema/fixed-compute violation occurred.

NumPy printed floating-point warnings during the optional float64 matrix-multiply diagnostic. Inspection showed finite outputs throughout: JSON was written with `allow_nan=False`, full-logit bounds were [-0.006249, -0.003513], and recomputed beta bounds were [0.498437771, 0.499121786]. Thus the warnings did not correspond to a model or artifact NaN/Inf.

## Reference soft-beta structure

These raw values are from the natural deterministic `theta=0.5000` retain-all trajectory. Threshold-specific later trajectories are summarized separately and are not mixed into this table.

- Count: 27,642
- Mean: 0.498709370
- Std: 0.000122001
- Min/max: 0.498437762 / 0.499121785
- Median: 0.498678029
- P10/P25/P75/P90: 0.498593599 / 0.498629153 / 0.498729065 / 0.498949945

### Beta by eligible pass position

| Pass | Count | Mean | Std | Min | Max | Median | P10 | P25 | P75 | P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4,607 | 0.498954698 | 0.000042354 | 0.498772591 | 0.499121785 | 0.498959690 | 0.498892725 | 0.498927385 | 0.498985678 | 0.499001676 |
| 3 | 4,607 | 0.498673390 | 0.000051658 | 0.498517275 | 0.498838097 | 0.498667061 | 0.498609424 | 0.498631269 | 0.498709530 | 0.498739988 |
| 4 | 4,607 | 0.498628223 | 0.000050013 | 0.498526573 | 0.498789281 | 0.498609126 | 0.498579741 | 0.498590529 | 0.498653948 | 0.498710918 |
| 5 | 4,607 | 0.498647926 | 0.000026783 | 0.498548806 | 0.498764008 | 0.498643041 | 0.498620009 | 0.498633355 | 0.498659998 | 0.498686397 |
| 6 | 4,607 | 0.498688253 | 0.000048536 | 0.498473912 | 0.498770177 | 0.498707891 | 0.498611987 | 0.498666704 | 0.498721808 | 0.498730898 |
| 7 | 4,607 | 0.498663727 | 0.000069987 | 0.498437762 | 0.498759329 | 0.498689473 | 0.498552889 | 0.498606980 | 0.498723477 | 0.498734319 |

At theta 0.5000, dwell is deterministically pass minus one, so the reference beta-by-dwell table for dwell 1–6 is numerically identical to the pass 2–7 table. Threshold-specific beta-by-dwell statistics are retained separately in `schedule_summary.json`.

### Variance decomposition

| Quantity | Variance | Fraction of total |
|---|---:|---:|
| Total beta variance | 1.488415e-8 | 100.00% |
| Between-position variance | 1.239497e-8 | 83.276% |
| Residual within-position variance | 2.489179e-9 | 16.724% |

The pooled residual within-position standard deviation is approximately 0.00004989. Individual within-position ranges are 0.000215–0.000349 and IQRs are 0.0000266–0.0001165. Position is dominant, but the residual is large enough to change actions at thresholds through the beta bulk.

## Trigger features and logit decomposition

Final parameters:

| Parameter | Value |
|---|---:|
| `w_c` | -0.0020735294 |
| `w_d` | -0.0018290979 |
| `w_rho` | -0.0005789109 |
| `w_dwell` | -0.0019584675 |
| `w_q` | 0.00001507495 |
| `w_alpha` | -0.0020746570 |
| bias | -0.0034299861 |

Reference-trajectory feature statistics and associations:

| Feature | Mean | Std | Min | Max | Pearson(beta) | Spearman(beta) | `abs(w)*std` | Contribution range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c | 0.063462 | 0.071716 | -0.204499 | 0.270034 | -0.8317 | -0.7764 | 1.4871e-4 | 9.8396e-4 |
| d | 0.008361 | 0.166811 | -0.358320 | 0.366600 | -0.3711 | -0.5580 | 3.0511e-4 | 1.32595e-3 |
| rho | 0.005226 | 0.069696 | -0.204499 | 0.198240 | -0.4485 | -0.5720 | 4.0348e-5 | 2.3315e-4 |
| dwell/M | 0.437500 | 0.213478 | 0.125000 | 0.750000 | -0.5562 | -0.3327 | 4.1809e-4 | 1.22404e-3 |
| q | -0.000625 | 0.000112 | -0.000782 | 0 | -0.2145 | -0.2021 | 1.6853e-9 | 1.1785e-8 |
| alpha | 0.349841 | 0.009241 | 0.336679 | 0.395584 | -0.0250 | -0.0682 | 1.9172e-5 | 1.2221e-4 |

Coefficient magnitude alone is misleading. Dwell is the largest single scale-adjusted component, but `d` is 73% of its standard deviation and `c` is 36%; the combined progress-only component (`c+d+rho`) has std 0.0004312 and range 0.0021857, comparable to and wider than dwell's std 0.0004181 and range 0.0012240. The q contribution is negligible.

### Counterfactual time-only diagnostic

- Full-logit std/variance: 0.0004880 / approximately 2.3815e-7
- Time-only (`w_dwell*dwell/M + bias`) std/variance: 0.0004181 / approximately 1.74799e-7
- Time-only versus full logit: Pearson 0.5562; Spearman 0.3327
- Full logit range: 0.0027361
- Progress-only logit range: 0.0021857

The trigger is position-sensitive but is not merely the learned dwell clock: a time-only score preserves only weak-to-moderate full-score ordering, especially by rank.

## Threshold-specific policies

### Intervention and schedule summary

| Theta | Mean total K | Mean adaptive | Hard rate | Frac K=1 | Frac K=7 | Unique schedules | Most common schedule | Common frac | Entropy bits |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.4984 | 6.9870 | 5.9870 | 0.99783 | 0 | 0.98871 | 5 | [1,2,3,4,5,6,7] | 0.98871 | 0.1067 |
| 0.4985 | 6.6716 | 5.6716 | 0.94526 | 0 | 0.70024 | 9 | [1,2,3,4,5,6,7] | 0.70024 | 1.2687 |
| 0.4986 | 4.3529 | 3.3529 | 0.55882 | 0 | 0.06186 | 21 | [1,2,6,7] | 0.59127 | 2.3441 |
| 0.4987 | 2.3983 | 1.3983 | 0.23305 | 0 | 0.01736 | 21 | [1,2] | 0.85457 | 1.0633 |
| 0.4988 | 2.0460 | 1.0460 | 0.17434 | 0.00087 | 0.00174 | 12 | [1,2] | 0.97938 | 0.2007 |
| 0.4989 | 1.8726 | 0.8726 | 0.14543 | 0.12741 | 0 | 2 | [1,2] | 0.87259 | 0.5503 |
| 0.4990 | 1.1116 | 0.1116 | 0.01859 | 0.88843 | 0 | 2 | [1] | 0.88843 | 0.5046 |
| 0.4991 | 1.00065 | 0.00065 | 0.000109 | 0.99935 | 0 | 2 | [1] | 0.99935 | 0.00783 |
| 0.4992 | 1.0000 | 0 | 0 | 1 | 0 | 1 | [1] | 1 | 0 |
| 0.5000 | 1.0000 | 0 | 0 | 1 | 0 | 1 | [1] | 1 | 0 |

Adaptive intervention counts at passes 2–7:

| Theta | p2 | p3 | p4 | p5 | p6 | p7 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4984 | 4607 | 4599 | 4607 | 4607 | 4567 | 4595 |
| 0.4985 | 4607 | 3478 | 4524 | 4604 | 4436 | 4480 |
| 0.4986 | 4607 | 582 | 1135 | 1380 | 3865 | 3878 |
| 0.4987 | 4607 | 221 | 440 | 334 | 364 | 476 |
| 0.4988 | 4603 | 71 | 86 | 21 | 15 | 23 |
| 0.4989 | 4020 | 0 | 0 | 0 | 0 | 0 |
| 0.4990 | 514 | 0 | 0 | 0 | 0 | 0 |
| 0.4991 | 3 | 0 | 0 | 0 | 0 | 0 |
| 0.4992 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5000 | 0 | 0 | 0 | 0 | 0 | 0 |

Divide each count by 4,607 for the corresponding fraction. Complete top-10 schedule tables are in `schedule_summary.json`. The main heterogeneous regime is 0.4985–0.4988; 0.4986 is the strongest schedule-diversity probe (21 schedules, 2.344 bits), although one schedule still contains 59.1% of episodes. At 0.4989 and above, heterogeneity is almost entirely a state-dependent pass-2 split rather than rich later scheduling.

### Dwell summaries

Histograms count completed commitment segments, not episodes.

| Theta | Mean dwell | Median | Segments | Nonzero dwell histogram |
|---:|---:|---:|---:|---|
| 0.4984 | 1.0019 | 1 | 32,189 | 1:32137, 2:44, 3:8 |
| 0.4985 | 1.0492 | 1 | 30,736 | 1:29355, 2:1250, 3:130, 4:1 |
| 0.4986 | 1.6081 | 1 | 20,054 | 1:15459, 2:937, 3:450, 4:2821, 5:40, 6:347 |
| 0.4987 | 2.9187 | 1 | 11,049 | 1:6344, 2:321, 3:150, 4:294, 5:3, 6:3937 |
| 0.4988 | 3.4213 | 1 | 9,426 | 1:4807, 2:32, 3:11, 4:55, 5:5, 6:4512, 7:4 |
| 0.4989 | 3.7381 | 6 | 8,627 | 1:4020, 6:4020, 7:587 |
| 0.4990 | 6.2974 | 7 | 5,121 | 1:514, 6:514, 7:4093 |
| 0.4991 | 6.9954 | 7 | 4,610 | 1:3, 6:3, 7:4604 |
| 0.4992 | 7.0000 | 7 | 4,607 | 7:4607 |
| 0.5000 | 7.0000 | 7 | 4,607 | 7:4607 |

All per-episode dwell sums were exactly 7, and segment count equaled total interventions for every episode.

### Diagnostic operating points and fixed-budget context

These are descriptive probes, not selected or optimal thresholds:

- Around K=1.25: theta 0.4990 gives K=1.112, near the fixed `P=inf` budget K=1.
- Around K=1.5: the grid brackets it sharply (0.4989 gives 1.873; 0.4990 gives 1.112). No close point occurs.
- Around K=2: theta 0.4988 gives K=2.046, approximately budget-matched to fixed P=4/P=6 (K=2). Theta 0.4989 gives K=1.873 and is another nearby diagnostic.
- Around K=3: no close grid point occurs; 0.4987 gives K=2.398 and is closer to K=2 than K=3, while 0.4986 jumps to K=4.353.
- Theta 0.4984/0.4985 approach fixed P=1's K=7 budget.
- Fixed P=3's K=3 budget is not closely matched by this grid.

The sharp budget changes reinforce that calibration remains important even though a stable ranking exists.

## Same-position state dependence

Every mixed-action position has full intervene/retain counts and mean/std for c, d, rho, dwell/M, q, alpha and beta in `schedule_summary.json`. The most informative intermediate probes are summarized below; values are `intervene mean ± std` versus `retain mean ± std`.

### Theta 0.4986

| Pass | n intervene / retain | c | d | rho | dwell/M | alpha | beta |
|---:|---:|---|---|---|---|---|---|
| 3 | 582 / 4025 | 0.1687±0.0848 / 0.3304±0.0310 | 0.1687±0.0848 / 0.3304±0.0310 | 0.1687±0.0848 / 0.3304±0.0310 | 0.125±0 / 0.125±0 | 0.3824±0.0294 / 0.3572±0.0096 | 0.498694±0.000081 / 0.498526±0.000038 |
| 4 | 1135 / 3472 | 0.1257±0.136 / 0.3210±0.023 | 0.0868±0.106 / 0.2765±0.020 | -0.0168±0.071 / -0.0155±0.017 | 0.1867±0.062 / 0.2497±0.006 | 0.3893±0.058 / 0.3580±0.009 | 0.498747±0.000123 / 0.498544±0.000023 |
| 5 | 1380 / 3227 | 0.0965±0.101 / 0.2820±0.016 | 0.0571±0.056 / 0.1284±0.030 | 0.0297±0.069 / -0.0348±0.036 | 0.1775±0.102 / 0.3712±0.030 | 0.4229±0.060 / 0.3602±0.022 | 0.498756±0.000102 / 0.498574±0.000016 |
| 6 | 3865 / 742 | 0.1670±0.108 / 0.2859±0.049 | -0.0742±0.092 / 0.1506±0.152 | -0.0356±0.069 / 0.1148±0.166 | 0.3960±0.168 / 0.3280±0.182 | 0.3869±0.058 / 0.4214±0.067 | 0.498701±0.000100 / 0.498530±0.000063 |
| 7 | 3878 / 729 | 0.0730±0.098 / 0.2725±0.050 | 0.0609±0.099 / 0.0914±0.163 | 0.0575±0.102 / 0.0639±0.129 | 0.1363±0.058 / 0.3987±0.223 | 0.4630±0.037 / 0.4221±0.062 | 0.498762±0.000103 / 0.498536±0.000051 |

### Theta 0.4987

| Pass | n intervene / retain | c | d | rho | dwell/M | alpha | beta |
|---:|---:|---|---|---|---|---|---|
| 3 | 221 / 4386 | 0.0761±0.060 / 0.3218±0.042 | 0.0761±0.060 / 0.3218±0.042 | 0.0761±0.060 / 0.3218±0.042 | 0.125±0 / 0.125±0 | 0.4096±0.027 / 0.3579±0.010 | 0.498784±0.000056 / 0.498535±0.000048 |
| 4 | 440 / 4167 | 0.0505±0.121 / 0.3070±0.039 | -0.0007±0.075 / 0.2581±0.048 | -0.0646±0.040 / -0.0195±0.016 | 0.1878±0.062 / 0.2499±0.003 | 0.4251±0.071 / 0.3577±0.010 | 0.498814±0.000097 / 0.498560±0.000044 |
| 5 | 334 / 4273 | 0.0439±0.058 / 0.2674±0.035 | 0.0326±0.060 / 0.1185±0.036 | 0.0343±0.056 / -0.0312±0.038 | 0.1355±0.050 / 0.3679±0.041 | 0.4684±0.061 / 0.3624±0.030 | 0.498791±0.000063 / 0.498586±0.000029 |
| 6 | 364 / 4243 | 0.1120±0.127 / 0.2226±0.033 | -0.1256±0.110 / -0.0564±0.103 | -0.0513±0.069 / -0.0327±0.066 | 0.3458±0.183 / 0.4766±0.086 | 0.4283±0.085 / 0.3678±0.041 | 0.498758±0.000105 / 0.498634±0.000050 |
| 7 | 476 / 4131 | -0.0379±0.111 / 0.1847±0.039 | -0.0819±0.103 / -0.1128±0.098 | -0.0834±0.098 / -0.0387±0.030 | 0.1670±0.080 / 0.6097±0.071 | 0.5022±0.053 / 0.3640±0.034 | 0.498869±0.000117 / 0.498617±0.000057 |

The q feature is also recorded in the artifact; its logit contribution is only about 1e-9 and is omitted from these display tables. At pass 3 for both probes, dwell/M is identical between actions, yet c/d/rho and beta separate substantially. This is direct descriptive evidence of state-conditioned ordering at fixed time and fixed dwell. It is not a causal claim about any individual feature.

At higher probes, the mixed pass-2 split is similarly state-conditioned: at theta 0.4989, intervene/retain c means are -0.0589/-0.0192 (4,020/587); at 0.4990 they are -0.1089/-0.0414 (514/4,093). Theta 0.4988 has mixed actions at all passes, but pass-2 retain has only four examples and later intervene groups are small; those descriptive mean/std values remain in the artifact rather than being overinterpreted.

## Goal-change diagnostics

| Theta | Replacements | Mean cosine | Std | Min | Max |
|---:|---:|---:|---:|---:|---:|
| 0.4984 | 27,582 | 0.8916 | 0.0463 | 0.6548 | 0.9942 |
| 0.4985 | 26,129 | 0.8797 | 0.0677 | 0.3445 | 0.9942 |
| 0.4986 | 15,447 | 0.7499 | 0.2533 | -0.3083 | 0.9941 |
| 0.4987 | 6,442 | 0.8380 | 0.1195 | 0.1294 | 0.9940 |
| 0.4988 | 4,819 | 0.8494 | 0.0373 | 0.6029 | 0.9877 |
| 0.4989 | 4,020 | 0.8527 | 0.0293 | 0.7921 | 0.9293 |
| 0.4990 | 514 | 0.8843 | 0.0184 | 0.8244 | 0.9196 |
| 0.4991 | 3 | 0.8726 | 0.0224 | 0.8409 | 0.8888 |
| 0.4992 | 0 | — | — | — | — |
| 0.5000 | 0 | — | — | — | — |

The 0.4986 replacements are not merely nominal: their broad cosine distribution includes materially changed and occasionally oppositely directed goals. Complete by-position cosine statistics are in `schedule_summary.json`.

## Numerical ranking stability

- Repeated deterministic extraction: beta arrays exactly identical; maximum difference 0; schedules exactly identical; dataset/order digest identical.
- Float32 beta levels: 5,846 unique values among 27,642 decisions; 21,796 decisions participate in ties beyond the first representative per level.
- Beta range: 11,476 float32 spacings at 0.5.
- Minimum/median positive level gap: 2.98e-8 / 5.96e-8 (0.5 / 1.0 spacing at 0.5).
- Float64 affine recomputation: maximum beta difference 5.93e-8; Spearman 0.999999919.
- Exact stable argsort differs because float64 refines float32 ties, but there are zero strict order reversals between distinct float32 beta levels.

The ordering is numerically real and repeatable. Individual adjacent levels are often only one float32 step apart, so fine ordering inside float32 ties should not be overinterpreted.

## Exploratory task metrics

These values are contextual only and are not used to rank or select thresholds. LM loss is shown per refinement pass; exact accuracy was zero for every threshold.

| Theta | Token accuracy | Exact accuracy | LM loss/pass |
|---:|---:|---:|---:|
| 0.4984 | 0.434390 | 0 | 2.173641 |
| 0.4985 | 0.434227 | 0 | 2.173527 |
| 0.4986 | 0.435586 | 0 | 2.172776 |
| 0.4987 | 0.433191 | 0 | 2.172146 |
| 0.4988 | 0.432917 | 0 | 2.171980 |
| 0.4989 | 0.433208 | 0 | 2.171761 |
| 0.4990 | 0.437848 | 0 | 2.170504 |
| 0.4991 | 0.438556 | 0 | 2.170444 |
| 0.4992 | 0.438556 | 0 | 2.170442 |
| 0.5000 | 0.438556 | 0 | 2.170442 |

No adaptive-superiority or threshold-quality claim follows from these held-out diagnostics.

## Clock-likeness assessment

The policy has a strong clock component: reference position/dwell explains 83.28% of beta variance, several thresholds are dominated by one schedule, and the budget moves sharply over a 0.0006 threshold interval. It is nevertheless not a pure learned clock:

1. pooled within-position beta variance is 16.72% of total;
2. fixed-time/fixed-dwell pass-3 examples split on state features at intermediate probes;
3. progress-only logit variation matches dwell variation in scale;
4. time-only rank correlation with the full trigger is only 0.333;
5. theta 0.4986 and 0.4987 each produce 21 schedules;
6. the state ranking is exactly reproducible and numerically stable.

This combination meets category A, while retaining an explicit warning that the current policy remains substantially time-structured.

## Recommended next scientific decision

Study decision calibration before changing eta or retraining. The cleanest next analysis is threshold-independent intervention-budget evaluation: use the learned ranking to impose predefined budgets or quantiles, preferably on a calibration split distinct from final held-out reporting, and compare schedule behavior at matched K. This separates ranking quality from absolute sigmoid calibration without declaring a held-out threshold “best.” Only after calibration and matched-budget behavior are understood should continued training or eta experiments resume.

Do not yet change the default threshold, initialization, feature normalization, temperature, eta, or model architecture on the basis of this audit alone.

## Artifacts

Artifact directory:

`/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_threshold_audit/20260816T183355Z_eta0_step96_threshold_audit`

- `threshold_summary.csv`
- `decision_level_metrics.csv` — 0.5000 reference trajectory, explicitly labeled here and in feature statistics
- `schedule_summary.json` — threshold-specific trajectories, hard actions, dwell, schedules, same-position comparisons, and goal changes
- `feature_beta_statistics.json`
- `clock_likeness.json`
- `hashes.json`
- `provenance.json`
