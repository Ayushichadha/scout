# Scout: Directional Subgoals for Hierarchical Reasoning

[![CI](https://github.com/Ayushichadha/scout/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayushichadha/scout/actions/workflows/ci.yml)

Scout is an experimental extension of the
[Hierarchical Reasoning Model (HRM)](https://github.com/sapientinc/HRM). It
studies whether a slow manager can improve long-horizon reasoning by emitting
temporally persistent directional goals for a fast worker in latent space.

It is research code: interfaces and conclusions may change as controls and
replications are added.

## Current baseline

The corrected fixed-period baseline implements the following causal sequence:

1. the worker consumes the goal and gate cached in the ACT carry;
2. the manager optionally emits the next commitment after the worker pass;
3. the post-pass pooled worker representation becomes that commitment's anchor;
4. on subsequent calls, the worker receives `gate * V_L(goal)`;
5. alignment uses `worker_repr - anchor`, not the absolute worker state.

Manager state is reset per sample at ACT episode boundaries. Goals, gates, and
anchors persist between fixed-period interventions. `manager_period` counts
outer ACT refinement calls, not inner H/L cycle micro-steps.

The first worker pass of a fresh episode remains unsteered because manager
emission happens after worker computation. The forced initial goal is first
consumed on the following outer call.

## Scientific caveat

Cached commitments are intentionally detached across ACT calls to preserve the
existing truncated-gradient contract. Consequently, causal displacement loss
trains the worker and `V_L`, but does not backpropagate into `goal_proj` or the
gate projection. Treat this as a baseline limitation, not evidence that the
manager direction policy is learned end to end.

## Repository layout

```text
HRM/                                  upstream HRM code with Scout extensions
  models/subgoal_head.py              fixed-period manager commitment state
  models/losses.py                    task, halt, and displacement losses
  models/hrm/hrm_act_v1.py            ACT recurrence and projected injection
analysis/visualize_hidden_states.ipynb shared PCA and per-cell UMAP analysis
scripts/dump_hidden_states.py          checkpoint representation extraction
scripts/verify_ablation_flags.py       five-cell CPU ablation probe
tests/                                 focused state, timing, and gradient tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r HRM/requirements.txt
pip install pytest ruff black
```

## Validation

```bash
pytest -q tests/
ruff check \
  HRM/models/subgoal_head.py \
  HRM/models/hrm/hrm_act_v1.py \
  HRM/models/losses.py \
  scripts/check_param_count.py \
  scripts/dump_hidden_states.py \
  scripts/verify_ablation_flags.py \
  tests/
python scripts/check_param_count.py
python scripts/verify_ablation_flags.py
```

CI runs these checks on every pull request and every push to `main`.

## Compact experiment configuration

The historical Scout headline runs used the following effective architecture:

```text
H_layers=1, L_layers=1, H_cycles=1, L_cycles=1
hidden_size=32, halt_max_steps=4, batch_size=4
manager_period=3 or 4, feudal_loss_weight=0.05
```

Do not compare new runs against those results unless architecture, dataset,
seed, halting behavior, and alignment semantics are matched explicitly. The
default training dataset is currently `data/conceptarc-mini`.

## Training

```bash
cd HRM
python pretrain.py device=cuda enable_wandb=true
```

Hydra overrides can select the fixed-period controls, for example:

```bash
python pretrain.py \
  arch.subgoal_head.manager_period=3 \
  arch.loss.feudal_loss_weight=0.05 \
  arch.subgoal_head.directional_displacement=true
```

## Scope

Scout currently provides a corrected fixed-period baseline. It does not yet
implement a learned intervention trigger, intervention penalty,
straight-through estimator, reduced manager space, multi-lag alignment, or
adaptive halting changes.

## Upstream attribution

Scout builds directly on Sapient's HRM implementation. Consult
[`HRM/README.md`](HRM/README.md) for upstream model and dataset documentation.
