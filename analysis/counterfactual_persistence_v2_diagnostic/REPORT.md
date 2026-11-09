# Counterfactual Learned Subgoal Persistence v2 Diagnostic

## Provenance and invariants

- Frozen checkpoint: `/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_400step/20260820T101100Z_adaptive_eta0_seed0_400step/checkpoints/step_400`
- Checkpoint SHA-256: `165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889`
- Model state unchanged during collection: `True`
- Episodes / decision states: 64 / 384
- Fixed reasoning budget: exactly M=8; eligible decisions are passes 2-7.
- Critic inputs: progress and goal disagreement only. Pass position is logged only.
- Counterfactual branches start from the same latent carry; only the selected directional commitment (goal, its gate, and its emission anchor) differs.
- Branch objectives and critic target are detached. Only critic MSE updates the critic; there is no straight-through estimator.

## Counterfactual behavior

- PERSIST wins: 90.365%
- REPLAN wins: 9.635%
- Advantage mean / median: -0.0065213 / -0.00625622
- Advantage std and [5%, 95%]: 0.00495817, [-0.0150536, 0.00140314]

## Signal usefulness (held out by episode)

- Progress vs A: Pearson -0.7046, Spearman -0.7073
- Goal disagreement vs A: Pearson -0.6240, Spearman -0.5182
- Tiny critic: MSE 2.31943e-05, R² 0.4064, Pearson 0.6927, Spearman 0.7287
- Sign accuracy: 95.614%; trivial majority baseline: 89.474%
- MSE improvement over train-mean baseline: 44.701%

## Clock-shortcut audit

- Prediction variance explained by pass: 79.464%
- Prediction vs position: Pearson -0.5149, Spearman -0.2823

| Pass | Mean A | Median A | Mean prediction |
|---:|---:|---:|---:|
| 2 | 4.36055e-05 | 0.00107193 | -0.000399558 |
| 3 | -0.00615862 | -0.00571966 | -0.00689078 |
| 4 | -0.0112487 | -0.00951767 | -0.00890067 |
| 5 | -0.0114041 | -0.00957608 | -0.00839721 |
| 6 | -0.00952602 | -0.00794244 | -0.00747176 |
| 7 | -0.00798264 | -0.00648856 | -0.006833 |

## Scientific stop decision

The two signals are **predictive** on this split. The clock audit is **concerning**. A full run is therefore **not yet justified**. This is a small single-checkpoint diagnostic, not a training result.
