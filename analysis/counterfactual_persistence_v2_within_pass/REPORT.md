# Counterfactual Persistence v2: Within-Pass Diagnostic

## Provenance

- Source states: `/Users/ayushi/Documents/hrm_research/analysis/counterfactual_persistence_v2_diagnostic/decision_states.csv`
- Source SHA-256: `f76df29fbc818bec95b8a951a36a63b0910bc625994d7ef3403f413cfb9d6185`
- Frozen checkpoint represented by source: `/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_400step/20260820T101100Z_adaptive_eta0_seed0_400step/checkpoints/step_400`
- 64 episodes, 384 eligible states, fixed M=8, passes 2-7.
- Original episode-level train/held-out split retained (270/114 states).
- Pass means come from training states only. Critic inputs are exactly residual progress and residual disagreement; pass and dwell/time features are excluded.

## Within-pass statistics

| Pass | N | REPLAN wins | Mean A | Median A | Progress Pearson/Spearman | Disagreement Pearson/Spearman |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 54.7% | -5.81499e-05 | 0.000428438 | -0.486 / -0.476 | -0.311 / -0.383 |
| 3 | 64 | 3.1% | -0.00532369 | -0.00476229 | -0.737 / -0.688 | -0.654 / -0.731 |
| 4 | 64 | 0.0% | -0.00938328 | -0.00888991 | -0.660 / -0.712 | -0.638 / -0.691 |
| 5 | 64 | 0.0% | -0.00939063 | -0.00908923 | -0.412 / -0.426 | -0.432 / -0.398 |
| 6 | 64 | 0.0% | -0.00794121 | -0.00686622 | -0.063 / -0.115 | -0.105 / -0.121 |
| 7 | 64 | 0.0% | -0.00703083 | -0.00631368 | 0.020 / -0.056 | 0.102 / 0.047 |

## Residual signal and critic (held out)

- Residual progress vs residual A: Pearson -0.425, Spearman -0.520.
- Residual disagreement vs residual A: Pearson -0.483, Spearman -0.530.
- Residual critic: MSE 2.49606e-05, R² -0.007, Pearson 0.321, Spearman 0.418.
- MSE improvement over the zero/train-mean residual baseline: 9.8%.

## Advantage prediction comparison (held out)

| Predictor | MSE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|
| pass only | 2.76585e-05 | 0.292 | 0.625 | 0.615 |
| pass plus residual critic | 2.49606e-05 | 0.361 | 0.666 | 0.687 |
| previous global two signal critic | 2.31943e-05 | 0.406 | 0.693 | 0.729 |

Adding the residual critic changes MSE versus pass-only by +9.8%.

## Rare REPLAN classification (held out)

Held-out positives/negatives: 12/102.

| Classifier | Accuracy | Balanced accuracy | REPLAN precision | REPLAN recall | AUROC |
|---|---:|---:|---:|---:|---:|
| trivial majority | 89.5% | 50.0% | 0.0% | 0.0% | 0.500 |
| pass only | 89.5% | 50.0% | 0.0% | 0.0% | 0.966 |
| pass plus residual critic | 97.4% | 87.5% | 100.0% | 75.0% | 0.993 |
| previous global two signal critic | 95.6% | 79.2% | 100.0% | 58.3% | 0.995 |

The pass-only AUROC above is a cross-pass ranking statistic: it assigns one constant score per pass and therefore has no within-pass ranking ability. Per-pass results are:

| Pass | Pos/Neg | Pass-majority balanced accuracy | Residual-combined balanced accuracy | Residual-score AUROC |
|---:|---:|---:|---:|---:|
| 2 | 12/7 | 50.0% | 87.5% | 0.893 |
| 3 | 0/19 | n/a | n/a | n/a |
| 4 | 0/19 | n/a | n/a | n/a |
| 5 | 0/19 | n/a | n/a | n/a |
| 6 | 0/19 | n/a | n/a | n/a |
| 7 | 0/19 | n/a | n/a | n/a |

## Held-out REPLAN-positive states

| Episode | Pass | A | Progress | Disagreement | Progress residual | Disagreement residual | Residual prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| all:00000001 | 2 | 0.000501633 | 0.03968 | 0.16811 | -0.01242 | -0.01093 | 0.00135089 |
| all:00000002 | 2 | 0.000205994 | 0.03593 | 0.17115 | -0.01617 | -0.00789 | 0.001831 |
| all:00000004 | 2 | 0.00121212 | 0.03967 | 0.16896 | -0.01243 | -0.01008 | 0.00136731 |
| all:00000012 | 2 | 0.00176907 | 0.04551 | 0.15596 | -0.00659 | -0.02308 | 0.000485011 |
| all:00000015 | 2 | 0.00197875 | 0.03660 | 0.15134 | -0.01550 | -0.02770 | 0.00139152 |
| all:00000020 | 2 | 0.00205064 | 0.03899 | 0.15465 | -0.01311 | -0.02439 | 0.00118848 |
| all:00000032 | 2 | 0.00177598 | 0.04763 | 0.12042 | -0.00447 | -0.05862 | -0.000259461 |
| all:00000046 | 2 | 0.00107193 | 0.06136 | 0.22519 | 0.00926 | 0.04614 | -0.000274421 |
| all:00000056 | 2 | 0.00293124 | 0.05649 | 0.22174 | 0.00439 | 0.04270 | 0.000289513 |
| all:00000057 | 2 | 0.00415611 | 0.05583 | 0.21763 | 0.00372 | 0.03859 | 0.000300573 |
| all:00000059 | 2 | 0.00126433 | 0.05296 | 0.21120 | 0.00085 | 0.03216 | 0.000546511 |
| all:00000062 | 2 | 0.00258398 | 0.06257 | 0.22175 | 0.01047 | 0.04271 | -0.000489867 |

Corresponding PERSIST-state summary (N=102): mean progress 0.11402, disagreement 0.73190, residual progress 0.00603, residual disagreement 0.03605, and residual prediction -0.000123548.

Because all held-out positive states occur at pass 2, the matched pass-2 comparison is:

| Oracle class | N | Mean A | Mean progress | Mean disagreement | Mean progress residual | Mean disagreement residual | Mean residual prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| REPLAN | 12 | 0.00179181 | 0.04777 | 0.18234 | -0.00433 | 0.00330 | 0.000643921 |
| PERSIST | 7 | -0.00295332 | 0.06507 | 0.22730 | 0.01297 | 0.04826 | -0.000723126 |

## Residual clock audit

- Prediction variance explained by pass: 15.8%.
- Prediction vs pass: Pearson 0.243, Spearman 0.233.
- Mean residual prediction by pass: {"2": 0.00014027230147468415, "3": -0.0006279597965706335, "4": -0.0009320637970966729, "5": -0.00031123136317259386, "6": 0.000476754172460029, "7": 0.000997660559062895}.

## Scientific verdict

Within-pass residual signal is **weak**. Rare-state discrimination is **better than trivial**. The remaining evidence is **not strong enough to justify full v2 training; another state feature should be evaluated first**. No training was launched.
