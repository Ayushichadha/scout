# Paper 2 controller implementation and diagnostic evidence

The controller implementation is tracked directly in this repository, including
the modified HRM source. It is not necessary to initialize the historical HRM
submodule entry to obtain Scout's model code.

## Implementation map

- `HRM/models/subgoal_head.py`: detached six-feature affine adaptive trigger,
  persistent goal state, and the experimental two-feature counterfactual critic.
- `HRM/models/hrm/hrm_act_v1.py`: retain/replace execution and causal consumption
  of goals, gates, and anchors on subsequent refinement passes.
- `HRM/models/losses.py`: directional alignment and local controller credit.
- `scripts/run_adaptive_400step.py`: training preflight and execution.
- `scripts/run_adaptive_matched_budget.py`: threshold calibration and evaluation.
- `scripts/run_adaptive_replication_audit.py`: independent-seed audits.
- `scripts/run_frozen_schedule_control.py` and `scripts/run_schedule_sweep_k2.py`:
  frozen-checkpoint timing interventions and exhaustive two-emission schedules.
- `scripts/run_counterfactual_persistence_v2_diagnostic.py`: collection of paired
  PERSIST/REPLAN branch targets and fitting of a diagnostic critic.
- `scripts/run_counterfactual_v2_within_pass_diagnostic.py`: position-confound
  audit, residual critic, and majority/pass-only comparisons.

## Released diagnostic inputs

The original 64-episode, 384-state collection and its reports are now included in
`analysis/counterfactual_persistence_v2_diagnostic/`. The within-pass reports and
held-out positive-state table are in
`analysis/counterfactual_persistence_v2_within_pass/`. Absolute paths inside these
historical reports record the original collection environment; use the relative
paths above in a new checkout. Checkpoint provenance hashes are retained.

The branch target is a bounded local objective, not final-episode return.
These files document diagnostic critic fitting, not a trained replacement policy.
The held-out within-pass residual R-squared is approximately -0.007; pass-2 AUROC
is approximately 0.893 on only 19 held-out states. All held-out REPLAN-positive
states occur at pass 2. These results do not establish endpoint adaptive value.
Full v2 controller training was not launched.

## Checkpoint-free verification

Use the project Python environment with PyTorch, NumPy, SciPy, scikit-learn, and
the HRM dependencies installed. From the repository root:

```bash
python -m pytest tests/ -q
python scripts/run_counterfactual_v2_within_pass_diagnostic.py \
  --input-csv analysis/counterfactual_persistence_v2_diagnostic/decision_states.csv \
  --output-dir /tmp/scout-controller-within-pass
python scripts/reproduce_paper_main_results.py
```

Recollecting latent counterfactual states requires the original checkpoint and
dataset; reanalyzing the released CSV does not. See [the release guide](README.md)
and [result-to-code map](MAIN_RESULTS.md) for the historical replay constraints.

The proposed review fixes and endpoint-supervised redesign are recorded in
`analysis/PAPER2_REVISION_PLAN.md`. They are plans, not completed experiments.
