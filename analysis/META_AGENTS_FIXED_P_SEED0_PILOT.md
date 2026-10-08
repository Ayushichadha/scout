# Meta-Agents Corrected Fixed-P Seed-0 Pilot

## Scope

This pilot ran the frozen corrected directional-manager baseline sequentially in
the prescribed order `P=3 -> P=1 -> P=4 -> P=6 -> P=inf`.

All conditions used:

- seed: `0`
- fixed refinement budget: `M=8`
- training steps: `400`
- epochs: `1`
- global batch size: `4`
- dataset: `data/conceptarc-mini`
- device: `cpu`
- feudal loss weight: `0.05`
- final held-out evaluation: enabled

No architecture, loss, manager timing, credit-assignment, fixed-compute,
dataset, or evaluation code was changed. No condition was restarted or tuned
after observing its metrics.

## Pre-launch Git provenance

- Commit: `3a0721ce21250376edee24e4e5aaad228db39de3`
- Branch: `subgoal_ablations`
- Upstream state: `origin/subgoal_ablations`, ahead 0, behind 0
- Working tree: dirty

Pre-existing staged paths:

- `AGENTS.md` (added)
- `eval_checkpoint.py` (added)
- `scripts/analyze_alignment_metrics.py` (added)
- `scripts/eval_checkpoints_local.py` (added)
- `scripts/plot_feudal_weight_sweep.py` (modified)
- `scripts/plot_manager_period_sweep.py` (modified)
- `scripts/plot_replication_distribution.py` (modified)
- `scripts/summarize_paper_metrics.py` (added)

Pre-existing unstaged modified paths:

- `HRM/config/arch/hrm_v1.yaml`
- `HRM/config/cfg_pretrain.yaml`
- `HRM/models/hrm/hrm_act_v1.py`
- `HRM/models/losses.py`
- `HRM/models/subgoal_head.py`
- `HRM/pretrain.py`
- `tests/test_fixed_period_directional.py`

Pre-existing untracked paths:

- `.claude/`
- `SubGoal_Augmented_HRM_Research.pdf`
- `SubGoal_Augmented_HRM_Research_extracted.txt`
- `analysis/ONE_STEP_MANAGER_CREDIT.md`
- `analysis/figures/`
- `scratch/`
- `scripts/run_meta_agents_fixed_p.py`

The runner recorded this dirty-tree provenance inside every `result.json`.
None of the pre-existing paths were modified or deleted by this pilot task.

## Exact commands

```bash
/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_meta_agents_fixed_p.py single --manager-period 3 --seed 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_meta_agents_fixed_p.py single --manager-period 1 --seed 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_meta_agents_fixed_p.py single --manager-period 4 --seed 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_meta_agents_fixed_p.py single --manager-period 6 --seed 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_meta_agents_fixed_p.py single --manager-period inf --seed 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute
```

## Result artifacts

| P | Result directory |
|---:|---|
| 1 | `experiments/meta_agents_fixed_p/runs/20260814T081914Z_corrected_fixed_p_p1_seed0/` |
| 3 | `experiments/meta_agents_fixed_p/runs/20260814T080716Z_corrected_fixed_p_p3_seed0/` |
| 4 | `experiments/meta_agents_fixed_p/runs/20260814T083824Z_corrected_fixed_p_p4_seed0/` |
| 6 | `experiments/meta_agents_fixed_p/runs/20260814T090630Z_corrected_fixed_p_p6_seed0/` |
| inf | `experiments/meta_agents_fixed_p/runs/20260814T091826Z_corrected_fixed_p_pinf_seed0/` |

Every directory contains `result.json`, `runtime_summary.json`, `stdout.log`,
and `stderr.log`.

## Comparison

`Final/train loss` is the latest logged training LM loss. `Directional loss`
is the training directional-alignment loss per executed refinement pass.

| P | Held-out accuracy | Exact accuracy | Final/train loss | Directional loss | Mean directional cosine | Mean gate | Mean interventions/episode | Mean commitment duration | Fixed-compute violations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.490851 | 0.000000 | 1.921046 | 0.391843 | 0.067331 | 0.472099 | 7.0 | 1.000000 | 0 |
| 3 | 0.491022 | 0.000000 | 1.932339 | 0.379313 | 0.073991 | 0.465528 | 3.0 | 2.333333 | 0 |
| 4 | 0.490502 | 0.000000 | 1.940004 | 0.440256 | -0.027364 | 0.478809 | 2.0 | 3.500000 | 0 |
| 6 | 0.491521 | 0.000000 | 1.894071 | 0.338085 | 0.088144 | 0.421181 | 2.0 | 3.500000 | 0 |
| inf | 0.491752 | 0.000000 | 1.949683 | 0.298979 | 0.093638 | 0.377632 | 1.0 | 7.000000 | 0 |

For reference, the task-loss average over all executed training passes was
2.043429 (P=1), 2.047543 (P=3), 2.039433 (P=4), 2.036941 (P=6), and
2.035197 (P=inf).

## Fixed-compute and schedule diagnostics

| P | Emissions after passes | Consumed durations | Completed episodes | Executed refinement passes | Mean passes/episode | Terminal emissions |
|---:|---|---|---:|---:|---:|---:|
| 1 | 1, 2, 3, 4, 5, 6, 7 | 1, 1, 1, 1, 1, 1, 1 | 200 | 1,600 | 8.0 | 0 |
| 3 | 1, 3, 6 | 2, 3, 2 | 200 | 1,600 | 8.0 | 0 |
| 4 | 1, 4 | 3, 4 | 200 | 1,600 | 8.0 | 0 |
| 6 | 1, 6 | 5, 2 | 200 | 1,600 | 8.0 | 0 |
| inf | 1 | 7 | 200 | 1,600 | 8.0 | 0 |

All runs completed exactly 400 optimizer steps. All structured scalar totals
were finite. Held-out and exact-accuracy fields were non-null. No NaN, Inf,
warning, exception, or traceback was found in the persisted logs. All runs
reported zero fixed-compute violations and zero unconsumed terminal emissions.

## First-pass interpretation

### Observed

1. Training was numerically and operationally stable for all five periods.
2. Held-out accuracy ranged from 0.490502 to 0.491752: an absolute spread of
   0.001250, or approximately 0.125 percentage points.
3. P=1 was 0.000170 below P=3 and 0.000670 below P=6, but 0.000350 above P=4.
4. P=inf had the highest seed-0 held-out accuracy; it was not worse than a
   finite-persistence condition in this run.
5. P=4 and P=6 had the same two interventions per episode but differed by
   0.001019 absolute held-out accuracy, with P=6 higher.
6. Accuracy was not monotonic in intervention frequency because P=4 dipped
   below both more- and less-frequent schedules.
7. Directional behavior varied: P=4 alone had a negative mean cosine and the
   largest directional loss; P=inf had the highest cosine, lowest directional
   loss, and lowest gate. Gate strength generally decreased toward longer
   persistence except for the P=4 deviation.
8. No fixed-compute, terminal-emission, missing-metric, or numerical evidence
   suggests a manager/fixed-P implementation failure.

### Interpretation

The held-out differences are very small and this single seed does not establish
a meaningful performance ordering. In particular, the pilot does not show
that P=1 is reliably worse than moderate persistence, and it does not show that
P=inf is worse than a finite schedule. P=4 versus P=6 is directionally
interesting because their intervention counts match, but its magnitude is too
small to separate a schedule effect from ordinary run variation.

The P=4 directional-cosine deviation is worth monitoring across additional
seeds, but it is not accompanied by any invariant violation and therefore looks
like an observed optimization/scientific outcome rather than a runtime defect.

## Anomalies and reproducibility caveat

The training loader seeds group and puzzle selection through its configured
generator, but example selection within a puzzle currently calls global
`np.random.choice` rather than the seeded generator. Separate seed-0 processes
therefore need not receive identical within-puzzle example draws. This was a
known provenance limitation before launch and was not changed during the pilot.
It can plausibly exceed the very small accuracy differences observed here, so
the seed-0 differences should not be attributed solely to manager period.

Exact held-out accuracy was zero for every condition. The field is present and
finite; this is a scientific result of the current small pilot rather than a
missing-evaluation failure.

## Recommendation

The mechanism and runner are stable enough to justify broader replication, but
the within-puzzle NumPy seeding issue should be resolved or explicitly accepted
as part of the protocol before launching seeds 1 and 2. For a clean paired
fixed-P comparison, prefer fixing that reproducibility issue and rerunning the
complete seed-0 matrix rather than comparing new deterministic seeds against
these partially unpaired seed-0 runs.

Do not implement adaptive replanning yet. After reproducible seeds 0, 1, and 2,
reassess whether the P=4 cosine anomaly and the small P=6/P=4 difference persist.
