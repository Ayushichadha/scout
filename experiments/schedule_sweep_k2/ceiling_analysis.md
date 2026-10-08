# Exact K=2 Forced-Schedule Ceiling Analysis

The headline aggregation throughout this report is **micro token accuracy** (total correct valid tokens divided by total valid tokens). Macro means of per-episode token accuracy are shown separately. Every schedule contains the same 3,686 ordered episodes.

## Schedule aggregates

| Schedule | Dwells | Micro token accuracy | Macro token accuracy | LM loss/pass | Exact accuracy |
|---|---|---:|---:|---:|---:|
| `[1,2]` | (1, 6) | 0.509655374 | 0.467002003 | 1.957840790 | 0.000000000 |
| `[1,3]` | (2, 5) | 0.516701315 | 0.481425983 | 1.956873793 | 0.000000000 |
| `[1,4]` | (3, 4) | 0.520236403 | 0.487511013 | 1.956718239 | 0.000000000 |
| `[1,5]` | (4, 3) | 0.521205786 | 0.488242241 | 1.956982955 | 0.000000000 |
| `[1,6]` | (5, 2) | 0.520748490 | 0.486622191 | 1.957439763 | 0.000000000 |
| `[1,7]` | (6, 1) | 0.518896125 | 0.482351787 | 1.957844850 | 0.000000000 |

## Fixed policy, floor, and held-out-label upper bound

- Best fixed clock: `[1,5]`, micro token accuracy `0.521205786`.
- Floor (analytical expectation under a uniform random schedule per episode): `0.517907249`.
- **Oracle ceiling (held-out-label upper bound; not an achievable policy or performance result):** `0.522556601`.
- Total K=2 state-conditioned timing headroom: `0.001350815` (95% episode-bootstrap CI `[0.000857608, 0.001942788]`).

The bootstrap resamples episodes with replacement 10,000 times using NumPy Philox seed `20260822`. In each headroom replicate, the best fixed schedule is recomputed; the oracle choice remains the per-episode held-out-label argmax. Percentile intervals are reported.

## Oracle argmax distribution

Ties are broken deterministically in favor of the smallest `k`; 3,252 episodes have two or more tied maxima.

| Schedule | Episodes | Fraction |
|---|---:|---:|
| `[1,2]` | 2149 | 0.583017 |
| `[1,3]` | 502 | 0.136191 |
| `[1,4]` | 656 | 0.177971 |
| `[1,5]` | 313 | 0.084916 |
| `[1,6]` | 35 | 0.009495 |
| `[1,7]` | 31 | 0.008410 |

Schedule `[1,2]` receives the deterministic argmax for 58.3% of episodes. The reported allocation is concentrated and the measured headroom is thin. Because most episodes tie and ties go to the smallest `k`, the identity and share of the dominant schedule are tie-break-sensitive; the ceiling and headroom are not.

## All pairwise schedule differences

Differences are left minus right in headline micro token accuracy, with paired episode-bootstrap 95% CIs.

| Difference | Estimate | 95% CI |
|---|---:|---:|
| `[1,2] - [1,3]` | -0.007045941 | [-0.007768192, -0.006304670] |
| `[1,2] - [1,4]` | -0.010581029 | [-0.011566327, -0.009630230] |
| `[1,2] - [1,5]` | -0.011550413 | [-0.012556747, -0.010599207] |
| `[1,2] - [1,6]` | -0.011093117 | [-0.011987651, -0.010233632] |
| `[1,2] - [1,7]` | -0.009240752 | [-0.010064252, -0.008466111] |
| `[1,3] - [1,4]` | -0.003535088 | [-0.003933410, -0.003161785] |
| `[1,3] - [1,5]` | -0.004504471 | [-0.005060292, -0.003990533] |
| `[1,3] - [1,6]` | -0.004047175 | [-0.004720499, -0.003461074] |
| `[1,3] - [1,7]` | -0.002194810 | [-0.003084859, -0.001459250] |
| `[1,4] - [1,5]` | -0.000969383 | [-0.001225601, -0.000750097] |
| `[1,4] - [1,6]` | -0.000512087 | [-0.001028459, -0.000093060] |
| `[1,4] - [1,7]` | +0.001340278 | [+0.000490770, +0.002071584] |
| `[1,5] - [1,6]` | +0.000457296 | [+0.000134539, +0.000737503] |
| `[1,5] - [1,7]` | +0.002309661 | [+0.001618097, +0.002930414] |
| `[1,6] - [1,7]` | +0.001852365 | [+0.001444008, +0.002230105] |

## Paired reversed-order dwell test

| Matched pair difference | Estimate | 95% CI | Verdict |
|---|---:|---:|---|
| `[1,2] - [1,7]` | -0.009240752 | [-0.010064252, -0.008466111] | distinguishable |
| `[1,3] - [1,6]` | -0.004047175 | [-0.004720499, -0.003461074] | distinguishable |
| `[1,4] - [1,5]` | -0.000969383 | [-0.001225601, -0.000750097] | distinguishable |

At least one reversed-order pair separates ([1,2] vs [1,7], [1,3] vs [1,6], [1,4] vs [1,5]); position carries information beyond dwell.

## Preregistered predictions

The three predictions equated the reversed-order pairs `[1,5]`/`[1,4]`, `[1,3]`/`[1,6]`, and `[1,7]`/`[1,2]`. A prediction is counted as holding when its paired CI includes zero. Therefore, 0 of 3 predictions held by the preregistered paired-order criterion.
