# Scout K=2 paper figures

Regenerate every figure from the repository root:

```bash
.venv/bin/python experiments/schedule_sweep_k2/figures/generate_figures.py
```

Each figure is written as an editable SVG and a 300 dpi PNG. The script sets a
fixed SVG hash salt and strips date metadata so repeated runs are byte-stable.

| Figure | Source data | Intended reading |
|---|---|---|
| `fig1_persistence_curve` | `../per_episode_matrix.csv` | Macro per-episode accuracy rises sharply with the first commitment dwell, peaks at the observed 4-pass dwell, then decays gently. The y-axis is explicitly marked as zoomed. |
| `fig2_clock_headroom` | `../ceiling_analysis.md` | A fixed clock captures most of the total gain above the uniform-random floor; the remaining oracle headroom and its published bootstrap CI are foregrounded below. |
| `fig3_reversed_order` | `../ceiling_analysis.md` | All three matched dwell-multiset comparisons favor putting the longer commitment first, and every published paired-bootstrap CI excludes zero. Signs are reoriented from the source table to “longer first minus shorter first.” |
| `fig4_schedule_distributions` | `../per_episode_matrix.csv` | Exact empirical CDF small multiples show every per-episode accuracy under all six schedules, with no smoothing, binning, interpolation, or subsampling. |
| `fig5_manager_worker_architecture` | `../../../HRM/models/hrm/hrm_act_v1.py`, `../../../HRM/models/subgoal_head.py`, and `../per_episode_matrix.csv` | The manager observes a completed worker pass, retains or replaces the goal, and the result first affects the next pass; the observed `[1,2]` and `[1,5]` schedules make persistence literal. |
| `fig6_sparse_schedule_grid` | `../per_episode_matrix.csv` | The six observed dwell combinations occupy one anti-diagonal, making the first/second-dwell confound explicit; neutral cells were not evaluated. |
| `fig7_oracle_concentration` | `../per_episode_matrix.csv` and `../ceiling_analysis.md` | An exact extra-correct-token decomposition shows how much of the published micro headroom is concentrated in the highest-gain episodes. |
| `fig8_oracle_tie_structure` | `../per_episode_matrix.csv` and `../ceiling_analysis.md` | The multiplicity of each episode's oracle maximum shows why the deterministic schedule allocation is tie-break-sensitive. |

`../provenance.json` is also read as an integrity check: its split size must
match the number of unique episodes in the matrix.

## Optional fixed-P bridge figure

Not generated. The closest corrected source located in the repository is
`../../../analysis/META_AGENTS_FIXED_P_3SEED_BASELINE.md`, which reports three
seeds for `P ∈ {1,3,4,6,inf}`. It does not satisfy the requested five-seed
`P=3/P=4` provenance requirement, so no values from it are plotted.
