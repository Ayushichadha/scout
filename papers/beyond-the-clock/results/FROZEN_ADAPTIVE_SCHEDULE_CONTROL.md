# Frozen Adaptive-Checkpoint Schedule Control

## Design

The exact adaptive step-400 weights were frozen and evaluated on the same 3,686 final episodes used by the matched-budget audit. No retraining or performance-driven selection occurred. Forced policies use the model's causal trajectory and manager-generated goal at the prescribed second intervention, then retain thereafter.

| Condition | Mean K | Token accuracy | Exact accuracy | LM loss/pass | Schedule |
|---|---:|---:|---:|---:|---|
| calibrated_adaptive | 2.000814 | 0.467015 | 0.000000 | 1.957886 | [1, 2] |
| forced_[1,2] | 2.000000 | 0.467002 | 0.000000 | 1.957841 | [1, 2] |
| forced_[1,4] | 2.000000 | 0.487511 | 0.000000 | 1.956718 | [1, 4] |
| forced_[1,6] | 2.000000 | 0.486622 | 0.000000 | 1.957440 | [1, 6] |

## Primary diagnostic

- Forced `[1,2]` minus calibrated adaptive accuracy: -0.000012607
- Forced `[1,2]` minus calibrated adaptive LM loss/pass: -0.000044937
- Forced `[1,4]` minus calibrated adaptive accuracy: +0.020496403
- Forced `[1,6]` minus calibrated adaptive accuracy: +0.019607581

Calibrated adaptive and strict `[1,2]` are effectively indistinguishable at aggregate precision; their tiny accuracy and loss differences point in opposite directions. The rare adaptive exceptions provide no clear aggregate value.

Forced `[1,4]` and `[1,6]` isolate timing within the same learned representations. They must not be conflated with the separately trained fixed-P checkpoints.

The within-checkpoint result points to timing selection—not generally weak adaptive-checkpoint representations—as the immediate failure: the same weights perform materially better when the second goal is delayed to pass 4 or 6.

All schedules used exactly eight refinement passes and two interventions per episode. Checkpoint file and loaded model-state hashes were unchanged, and verified checkpoint/config/source copies are stored under `frozen_inputs/`.
