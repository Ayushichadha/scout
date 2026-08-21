# K=2 Forced-Schedule Sweep Preregistration

Recorded before running any schedule evaluation in this sweep.

## Frozen design

- Checkpoint: `experiments/meta_agents_adaptive_400step/20260820T101100Z_adaptive_eta0_seed0_400step/checkpoints/step_400`
- Expected checkpoint SHA-256: `165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889`
- Evaluation split: the ordered 3,686-episode final split from the Aug 20 frozen-schedule control; the 921-episode calibration split is excluded.
- Compute: exactly eight outer refinement passes per episode.
- Intervention budget: exactly two forced manager emissions, after pass 1 and after pass `k`, for each `k` in `2..7`; no adaptive trigger or threshold selection is used.
- Primary headline aggregation: micro token accuracy, defined as total correct valid tokens divided by total valid tokens. Macro per-episode accuracy will also be reported.

## Predictions

The preregistered hypothesis is that schedule value is a function of the dwell multiset and not of the position of the intervention. It predicts:

- `[1,5]`, with dwell `(4,3)`, scores close to `[1,4]`, with dwell `(3,4)`.
- `[1,3]`, with dwell `(2,5)`, scores close to `[1,6]`, with dwell `(5,2)`.
- `[1,7]`, with dwell `(6,1)`, scores close to `[1,2]`, with dwell `(1,6)`.

The paired order tests and their episode-bootstrap 95% confidence intervals will determine whether each reversed-order pair is distinguishable. If all three pairs are indistinguishable, the results support the dwell-multiset-only hypothesis; if any pair separates, intervention position carries information beyond dwell.

The per-episode oracle ceiling will be labeled only as a held-out-label upper bound, never as an achievable performance result.
