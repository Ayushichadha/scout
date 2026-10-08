# Main paper results: code and published inputs

The [published arXiv paper](https://arxiv.org/abs/2609.00874) contains three
400-step adaptive seeds. The exhaustive timing landscape, oracle, and
counterfactual diagnostics use seed 0. Earlier feudal-weight and manager-period
sweep plots belong to the first study and historical motivation.

| Paper result | Training / evaluation / analysis code | Published evidence |
|---|---|---|
| Published Table 1: three adaptive seeds versus their best tested forced schedules | `scripts/run_adaptive_400step.py --seed`, `scripts/run_adaptive_replication_audit.py` | Seed 0: `results/schedule_comparison.csv`; seeds 1–2: `results/replications/seed*/schedule_comparison.csv` |
| Seed-0 matched training: fixed P=4, P=6 versus calibrated adaptive | `scripts/run_meta_agents_fixed_p.py`, `scripts/run_adaptive_400step.py`, `scripts/run_adaptive_matched_budget.py`, `scripts/run_adaptive_step400_audit.py` | `results/final_comparison.csv`, `results/calibration_summary.json`, `results/META_AGENTS_ADAPTIVE_MATCHED_BUDGET_K2_STEP400.md` |
| Clock collapse: 96.76% variance explained, 99.19% schedule [1,2] | `build_reference_audit` in `scripts/run_adaptive_step400_audit.py`; `summarize_schedules` in `scripts/run_adaptive_matched_budget.py` | `results/decision_level_metrics.csv`, `results/feature_beta_statistics.json`, `results/clock_likeness.json`, `results/adaptive_schedule_summary.json` |
| Seed-0 frozen adaptive weights, forced [1,2], [1,4], [1,6] | `scripts/run_frozen_schedule_control.py` | `results/schedule_comparison.csv`, `results/FROZEN_ADAPTIVE_SCHEDULE_CONTROL.md` |
| Persistence optimum [1,5], reversed-order controls, oracle headroom and bootstrap CIs | `scripts/run_schedule_sweep_k2.py` (`analyze`, `bootstrap_analysis`) | `experiments/schedule_sweep_k2/per_episode_matrix.csv`, `analysis_summary.json`, `ceiling_analysis.md`, `provenance.json` |
| Main figures: architecture, persistence curve, clock headroom | `experiments/schedule_sweep_k2/figures/generate_figures.py` | Published matrix and analysis above; tracked PNG and SVG figures |

Paths in the code column are relative to the repository root. `results/` is
relative to this document. Model, loss, and gradient-credit code is in
`HRM/models/{subgoal_head.py,losses.py,hrm/hrm_act_v1.py}`.

## Recompute the published analysis

From the repository root, using the environment described in the release README:

```bash
python scripts/reproduce_paper_main_results.py
```

This prints the archived matched and frozen comparison tables, independently
recomputes the clock variance decomposition from decision records, and recomputes
the six-clock aggregates, label-oracle ceiling, and all paired bootstrap intervals
from 22,116 episode/schedule rows. It checks the recomputed values against the
archived summary and writes reports to `outputs/paper-main-results/`.

The command does not rerun checkpoint inference. The matched/frozen table values
are archived aggregates, not newly evaluated predictions. The published matrix
contains metrics and episode identifiers, not puzzle inputs or model weights.

To regenerate all main and supplementary figures:

```bash
pip install matplotlib pandas
python experiments/schedule_sweep_k2/figures/generate_figures.py
```

To rerun training or checkpoint inference, use the release README's commands.
The original preprocessed dataset and checkpoint archives are still required for
exact historical replay. The P=4/P=6 comparison uses audited deterministic
reconstructions of weights that were not originally saved. This distinction is
recorded in the published step-400 report.
