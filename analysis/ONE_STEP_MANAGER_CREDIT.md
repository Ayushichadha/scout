# One-step manager credit repair

Date: 2026-08-11

## Problem

The corrected directional baseline stored the emitted goal in recurrent carry
with `detach()`. On the first pass that consumed the goal, directional loss
trained the worker and `V_L`, but it had no gradient to `goal_proj` (`W_g`) or
the gate projection.

Training performs one outer ACT forward, backward, and optimizer step before
the returned carry is reused. A live emission graph therefore cannot safely be
stored across calls: its shared graph has already been backpropagated and its
parameters may have been updated.

## Repair

Recurrent goal, gate, anchor, `z_H`, and `z_L` remain detached. A separate
credit record stores only the detached manager representation associated with
the latest emission and a per-sample `pending` bit.

On the first consuming pass, the manager goal and gate are locally recomputed
from that detached representation. A value-preserving surrogate uses the exact
cached commitment in the forward pass and the recomputed local Jacobian in the
backward pass:

```text
credit_goal = cached_goal + recomputed_goal - stopgrad(recomputed_goal)
credit_gate = cached_gate + recomputed_gate - stopgrad(recomputed_gate)
```

The pending credit is consumed once. A later emission replaces the detached
manager representation and schedules one new consuming-pass credit operation.
No computation graph is stored in carry or crosses an optimizer step.

## Tiny real-model probe

The probe uses the real HRM forward path with a goal emitted after call 1 and
consumed on call 2. It executes backward and an optimizer step between those
calls, matching `train_batch()`. Norms are from directional alignment loss only.

| Gradient target | Before-equivalent detached path | Repaired path |
| --- | ---: | ---: |
| `L_level` | 1.756205 | 1.756205 |
| `V_L` | 0.052824 | 0.052824 |
| `goal_proj / W_g` | 0.000000 | 0.737961 |
| gate projection | 0.000000 | 1.047371 |

The active forward goal exactly equals the emitted cached goal. The active
surrogate has a local gradient, while recurrent goal, anchor, credit manager
representation, returned `z_H`, and returned `z_L` remain detached. Fixed-P=3
emissions remain on calls 1, 3, 6, ... .
