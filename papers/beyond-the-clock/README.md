# Beyond the Clock: Scout's second paper

Code release for the Meta-Agents study. The current manuscript title is
**When Should a Manager Intervene? Diagnosing Learned Subgoal Timing in
Hierarchical Reasoning Models**. This work extends Scout's latent manager/worker
mechanism with persistent directional commitments and a learned replanning trigger.

## Main result

Under eight refinement passes and a calibrated mean of two commitments per episode,
the seed-0 adaptive trigger chooses schedule [1,2] in 99.19% of final episodes.
Its macro token accuracy is 46.70%, versus 48.13% and 48.56% for separately trained
fixed P=4 and P=6 models. Delaying intervention under frozen adaptive weights raises
accuracy to 48.75% ([1,4]) and 48.66% ([1,6]). The exhaustive six-clock sweep peaks
at [1,5]; its held-out-label oracle has only 0.135 percentage points of micro
accuracy headroom. The oracle is a diagnostic ceiling, not an achievable policy.
These are single-seed findings, not a general claim against adaptive management.

## Setup and mechanism checks

Run from the repository root. HRM source is tracked directly in Scout; the historical
`.gitmodules` entry does not replace the modified source with upstream HRM.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HRM/requirements.txt
pip install pytest
python -m pytest tests/ -q
python scripts/run_meta_agents_fixed_p.py smoke --fixed-refinement-steps 8
```

The smoke check uses synthetic inputs and needs no dataset or checkpoint.
The tests cover causal goal consumption, fixed compute, trigger gradients,
deterministic seeding, matched-budget evaluation, and forced schedules.

## Training and evaluation

The paper uses the preprocessed `HRM/data/conceptarc-mini` dataset. It is not
included in this release. The upstream ARC preprocessing entry point is
`HRM/dataset/build_arc_dataset.py`, but its default corpus/augmentation settings
must not be assumed to reproduce this specific dataset. Exact regeneration of the
paper dataset remains to be documented; use the original dataset for numerical replay.

Preview the fixed-clock training matrix (add `--execute` to train):

```bash
python scripts/run_meta_agents_fixed_p.py matrix \
  --manager-periods 4 6 --seeds 0 --fixed-refinement-steps 8 \
  --training-steps 400 --device cpu --final-eval
python scripts/run_adaptive_400step.py
# Launch adaptive training after reviewing the preflight:
python scripts/run_adaptive_400step.py --execute
```

The audited 96-step reference config is included at its original path under
`experiments/meta_agents_adaptive_sanity/`. The adaptive runner validates its
400-step config against that reference before training.

Calibrate and evaluate with explicit checkpoint/config paths:

```bash
python -m scripts.run_adaptive_matched_budget \
  --adaptive-checkpoint /path/to/adaptive/step_400 \
  --adaptive-config /path/to/adaptive/resolved_config.yaml \
  --fixed-p4-checkpoint /path/to/p4/step_400 \
  --fixed-p4-config /path/to/p4/all_config.yaml \
  --fixed-p6-checkpoint /path/to/p6/step_400 \
  --fixed-p6-config /path/to/p6/all_config.yaml
```

`run_adaptive_step400_audit.py`, `run_frozen_schedule_control.py`, and
`run_schedule_sweep_k2.py` are historical replay drivers: they require the original
archived experiment paths and, where specified, checkpoint/split hashes. They do
not silently substitute newly trained weights. Checkpoints are not distributed
in this code release. Original P=4/P=6 weights were not saved; the audited comparison
uses deterministic reconstructions, as described in the archived report.

## Code map and evidence

- `HRM/models/subgoal_head.py`: goals, gates, fixed and learned replanning.
- `HRM/models/hrm/hrm_act_v1.py`: persistent goal injection and causal consumption.
- `HRM/models/losses.py`: displacement alignment and manager/trigger credit.
- `HRM/utils/seeding.py`, `HRM/pretrain.py`: reproducible training and runtime audits.
- `scripts/run_adaptive_matched_budget.py`: calibration, held-out splits, clock diagnostics.
- `scripts/run_frozen_schedule_control.py`: intervention under frozen weights.
- `scripts/run_schedule_sweep_k2.py`: six-clock sweep and oracle ceiling analysis.
- `results/`: selected archived summaries copied without changing reported values.
- `analysis/`: earlier mechanism and baseline audits; experimental counterfactual
  persistence-v2 code is included separately from the main paper protocol.
- `experiments/schedule_sweep_k2/figures/`: previously tracked publication figures.

This is a code and summary release. It supports mechanism checks and training;
exact historical numerical replay additionally requires the original data and
checkpoint archives. Manuscript drafts, scratch work, and checkpoints are excluded.
