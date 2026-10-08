# Meta-Agents Genuine Three-Seed Fixed-P Baseline

## Scope and run status

The deterministic corrected fixed-P baseline completed successfully for
`P = 1, 3, 4, 6, inf` and genuine experiment seeds `0, 1, 2`.

- Successful runs: **15/15**
- Technical failures or reruns: **0**
- Optimizer steps per run: **400**
- Completed episodes per run: **200**
- Executed refinement passes per run: **1,600**
- Mean refinement passes per episode: **8.0**
- Fixed-compute violations: **0 in every run**
- Unconsumed terminal emissions: **0 in every run**
- NaN, Inf, warning, traceback, or runtime exception: **none found**
- Final held-out and exact-accuracy fields: **present and finite in every run**

The earlier pre-seeding-fix seed-0 pilot is explicitly excluded. Only the 15
timestamped result directories listed below enter this analysis.

## Reproducibility gate

Before launch, the independent-process probe was rerun:

```bash
.venv/bin/python scripts/probe_seed_reproducibility.py \
  --seed 0 --different-seed 1 --trace-count 16 --loss-steps 4
```

It confirmed identical seed-0 sample traces, initial model SHA-256, and first
four training losses across two independent processes. Seed 1 changed both the
sample trace and model checksum.

## Git provenance

- Commit: `3a0721ce21250376edee24e4e5aaad228db39de3`
- Branch: `subgoal_ablations`
- Upstream at launch: `origin/subgoal_ablations`, ahead 0, behind 0
- Working tree: dirty
- Every included `result.json` records this commit and `dirty: true`.

Dirty/staged/untracked paths recorded before launch:

- Staged: `AGENTS.md`, `eval_checkpoint.py`,
  `scripts/analyze_alignment_metrics.py`,
  `scripts/eval_checkpoints_local.py`,
  `scripts/plot_feudal_weight_sweep.py`,
  `scripts/plot_manager_period_sweep.py`,
  `scripts/plot_replication_distribution.py`, and
  `scripts/summarize_paper_metrics.py`.
- Unstaged modified: `HRM/config/arch/hrm_v1.yaml`,
  `HRM/config/cfg_pretrain.yaml`, `HRM/models/hrm/hrm_act_v1.py`,
  `HRM/models/losses.py`, `HRM/models/subgoal_head.py`, `HRM/pretrain.py`,
  `HRM/puzzle_dataset.py`, and `tests/test_fixed_period_directional.py`.
- Untracked: `.claude/`, `HRM/utils/seeding.py`,
  `SubGoal_Augmented_HRM_Research.pdf`,
  `SubGoal_Augmented_HRM_Research_extracted.txt`,
  `analysis/META_AGENTS_FIXED_P_SEED0_PILOT.md`,
  `analysis/ONE_STEP_MANAGER_CREDIT.md`, `analysis/figures/`, `scratch/`,
  `scripts/probe_seed_reproducibility.py`,
  `scripts/run_meta_agents_fixed_p.py`, and
  `tests/test_experiment_seeding.py`.

No unrelated path was deleted or modified during experiment execution.

## Exact matrix commands

```bash
.venv/bin/python scripts/run_meta_agents_fixed_p.py matrix --manager-periods 1 3 4 6 inf --seeds 0 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

.venv/bin/python scripts/run_meta_agents_fixed_p.py matrix --manager-periods 1 3 4 6 inf --seeds 1 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute

.venv/bin/python scripts/run_meta_agents_fixed_p.py matrix --manager-periods 1 3 4 6 inf --seeds 2 --fixed-refinement-steps 8 --training-steps 400 --epochs 1 --batch-size 4 --dataset data/conceptarc-mini --device cpu --feudal-loss-weight 0.05 --final-eval --execute
```

## Resolved-configuration audit

The 15 structured resolved configurations were compared after removing only:

- `seed`;
- run name and run-summary output path;
- `arch.subgoal_head.manager_period`; and
- `arch.subgoal_head.initial_goal_only` for `P=inf` semantics.

All remaining resolved fields were exactly identical. Common effective settings
include CPU execution, ConceptARC-mini, batch size 4, 400 steps, one epoch,
constant dense LR `1e-4`, puzzle-embedding LR `1e-2`, no warmup, M=8,
hidden size 32, one H/L layer and cycle, two heads, directional displacement,
and feudal loss weight 0.05. Each result retains the complete resolved
configuration and executable command for audit.

## Primary aggregate results

Standard deviations below are sample standard deviations over seeds 0, 1, and
2. `Train loss` is the mean of each run's final logged training LM loss.

| P | Held-out accuracy mean | Std | Exact acc. mean | Exact std | Train loss | Directional loss | Mean cosine | Mean gate | Interventions | Commitment duration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.483455 | 0.007422 | 0.000000 | 0.000000 | 1.939122 | 0.384776 | 0.104594 | 0.488523 | 7.0 | 1.000000 |
| 3 | 0.482618 | 0.006666 | 0.000000 | 0.000000 | 1.932040 | 0.399620 | 0.096415 | 0.508476 | 3.0 | 2.333333 |
| 4 | 0.481043 | 0.004700 | 0.000000 | 0.000000 | 1.927533 | 0.407181 | 0.085123 | 0.506601 | 2.0 | 3.500000 |
| 6 | **0.483522** | 0.005193 | 0.000000 | 0.000000 | 1.928446 | 0.402222 | 0.092719 | 0.504662 | 2.0 | 3.500000 |
| inf | 0.480696 | 0.009970 | 0.000000 | 0.000000 | 1.922706 | 0.416694 | 0.045290 | 0.496581 | 1.0 | 7.000000 |

## Individual seed results

| Seed | P | Held-out acc. | Exact acc. | Final train loss | Directional loss | Mean cosine | Mean gate | Interventions | Duration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.489082 | 0.000000 | 1.900726 | 0.381398 | 0.082537 | 0.468359 | 7.0 | 1.000000 |
| 0 | 3 | 0.486814 | 0.000000 | 1.894602 | 0.375482 | 0.079567 | 0.463759 | 3.0 | 2.333333 |
| 0 | 4 | 0.481030 | 0.000000 | 1.887635 | 0.425168 | -0.011424 | 0.471663 | 2.0 | 3.500000 |
| 0 | 6 | 0.485266 | 0.000000 | 1.889919 | 0.338444 | 0.087247 | 0.421783 | 2.0 | 3.500000 |
| 0 | inf | 0.471466 | 0.000000 | 1.884493 | 0.301176 | 0.088342 | 0.378151 | 1.0 | 7.000000 |
| 1 | 1 | 0.486239 | 0.000000 | 2.000289 | 0.394153 | 0.196225 | 0.550396 | 7.0 | 1.000000 |
| 1 | 3 | 0.486108 | 0.000000 | 1.999170 | 0.410515 | 0.212418 | 0.591131 | 3.0 | 2.333333 |
| 1 | 4 | 0.485749 | 0.000000 | 2.004154 | 0.387384 | 0.258260 | 0.579374 | 2.0 | 3.500000 |
| 1 | 6 | 0.487619 | 0.000000 | 1.999596 | 0.495804 | 0.094081 | 0.620769 | 2.0 | 3.500000 |
| 1 | inf | 0.491270 | 0.000000 | 1.999589 | 0.556123 | 0.030330 | 0.655547 | 1.0 | 7.000000 |
| 2 | 1 | 0.475044 | 0.000000 | 1.916351 | 0.378778 | 0.035021 | 0.446814 | 7.0 | 1.000000 |
| 2 | 3 | 0.474931 | 0.000000 | 1.902348 | 0.412863 | -0.002741 | 0.470536 | 3.0 | 2.333333 |
| 2 | 4 | 0.476349 | 0.000000 | 1.890811 | 0.408993 | 0.008534 | 0.468766 | 2.0 | 3.500000 |
| 2 | 6 | 0.477681 | 0.000000 | 1.895824 | 0.372418 | 0.096829 | 0.471435 | 2.0 | 3.500000 |
| 2 | inf | 0.479350 | 0.000000 | 1.884036 | 0.392782 | 0.017198 | 0.456045 | 1.0 | 7.000000 |

## Result directories and per-result provenance

Every listed directory contains `result.json`, `runtime_summary.json`,
`stdout.log`, and `stderr.log`. The seed, P, commit, dirty-tree details, exact
command, and full resolved configuration are embedded in its `result.json`.

| Seed | P | Result directory | Commit | Dirty |
|---:|---:|---|---|---:|
| 0 | 1 | `experiments/meta_agents_fixed_p/runs/20260814T103656Z_corrected_fixed_p_p1_seed0/` | `3a0721ce` | yes |
| 0 | 3 | `experiments/meta_agents_fixed_p/runs/20260814T104737Z_corrected_fixed_p_p3_seed0/` | `3a0721ce` | yes |
| 0 | 4 | `experiments/meta_agents_fixed_p/runs/20260814T105944Z_corrected_fixed_p_p4_seed0/` | `3a0721ce` | yes |
| 0 | 6 | `experiments/meta_agents_fixed_p/runs/20260814T111305Z_corrected_fixed_p_p6_seed0/` | `3a0721ce` | yes |
| 0 | inf | `experiments/meta_agents_fixed_p/runs/20260814T112856Z_corrected_fixed_p_pinf_seed0/` | `3a0721ce` | yes |
| 1 | 1 | `experiments/meta_agents_fixed_p/runs/20260814T114413Z_corrected_fixed_p_p1_seed1/` | `3a0721ce` | yes |
| 1 | 3 | `experiments/meta_agents_fixed_p/runs/20260814T120743Z_corrected_fixed_p_p3_seed1/` | `3a0721ce` | yes |
| 1 | 4 | `experiments/meta_agents_fixed_p/runs/20260814T122149Z_corrected_fixed_p_p4_seed1/` | `3a0721ce` | yes |
| 1 | 6 | `experiments/meta_agents_fixed_p/runs/20260814T123543Z_corrected_fixed_p_p6_seed1/` | `3a0721ce` | yes |
| 1 | inf | `experiments/meta_agents_fixed_p/runs/20260814T125007Z_corrected_fixed_p_pinf_seed1/` | `3a0721ce` | yes |
| 2 | 1 | `experiments/meta_agents_fixed_p/runs/20260814T132258Z_corrected_fixed_p_p1_seed2/` | `3a0721ce` | yes |
| 2 | 3 | `experiments/meta_agents_fixed_p/runs/20260814T133746Z_corrected_fixed_p_p3_seed2/` | `3a0721ce` | yes |
| 2 | 4 | `experiments/meta_agents_fixed_p/runs/20260814T135035Z_corrected_fixed_p_p4_seed2/` | `3a0721ce` | yes |
| 2 | 6 | `experiments/meta_agents_fixed_p/runs/20260814T140632Z_corrected_fixed_p_p6_seed2/` | `3a0721ce` | yes |
| 2 | inf | `experiments/meta_agents_fixed_p/runs/20260814T141850Z_corrected_fixed_p_pinf_seed2/` | `3a0721ce` | yes |

## Required comparisons

Differences are absolute held-out-accuracy differences. Paired standard
deviations describe the three seedwise differences; they are not significance
tests.

### Frequent versus moderate persistence

| Comparison | Mean difference | Paired sample std | Seedwise differences | Direction consistency |
|---|---:|---:|---|---|
| P1 - P3 | +0.000837 | 0.001239 | +0.002268, +0.000131, +0.000112 | P1 higher 3/3 |
| P1 - P4 | +0.002412 | 0.004966 | +0.008052, +0.000490, -0.001305 | P1 higher 2/3 |
| P1 - P6 | -0.000067 | 0.003421 | +0.003816, -0.001380, -0.002638 | P1 higher 1/3 |

P=1 is therefore descriptively almost identical to the moderate schedules as a
group. Its advantage over P=3 is consistent in direction but tiny; it does not
generalize across P=4 and P=6.

### Finite persistence versus no replanning

P=6 is the best finite condition by mean held-out accuracy. The comparison is:

```text
Pinf - P6 = -0.002827 mean accuracy
paired sample std = 0.009555
seedwise differences = -0.013800, +0.003651, +0.001669
```

P=inf is lower on average because of seed 0, but it beats P=6 on seeds 1 and 2.
The finite-versus-no-replanning ordering is therefore inconsistent and well
within the observed seed variability.

### Matched intervention count: P=4 versus P=6

Both conditions emit twice per episode, but P=4 emits after passes 1 and 4,
while P=6 emits after passes 1 and 6.

```text
P6 - P4 = +0.002480 mean accuracy
paired sample std = 0.001544
seedwise differences = +0.004236, +0.001871, +0.001333
```

P=6 is higher on all three seeds. This is the clearest descriptive schedule
effect in the matrix and suggests that intervention timing may matter even when
intervention count is matched. Its absolute magnitude is still small and no
statistical claim is made from three seeds.

## Interpretation

1. **Reproducible fixed-P persistence effect:** There is no broad monotonic or
   reproducible performance effect of persistence. Aggregate means span only
   0.002827, smaller than each condition's across-seed standard deviation.
2. **P=1 versus moderate P:** Broadly indistinguishable. It is slightly above
   P=3 and P=4 on mean accuracy, essentially tied with P=6, and not uniformly
   better across all moderate schedules and seeds.
3. **P=inf versus finite P:** Inconsistent. P=inf is lowest on mean accuracy
   but wins seed 1 and exceeds the best finite P on two of three paired seeds.
4. **Highest mean held-out accuracy:** P=6 at 0.483522, only 0.000067 above P=1.
5. **Difference versus seed variability:** Overall P differences are within
   observed seed variability. The P=4/P=6 paired direction is consistent, but
   the absolute difference remains small.
6. **P=4 versus P=6 timing:** Descriptively yes: P=6 beats P=4 on all seeds at
   matched intervention count, suggesting later timing may help. More seeds
   would be needed before treating this as robust.
7. **Directional trends:** Directional loss rises loosely with persistence
   from P=1 toward P=inf, and cosine is lowest for P=inf, but both are
   non-monotonic and have substantial seed variation. Gate means are nearly
   flat (approximately 0.489-0.508). There is no clean systematic trend.
8. **Implementation failure:** None. All schedules, compute invariants,
   terminal-emission checks, evaluations, metrics, and logs are valid.

Exact accuracy is zero in all 15 runs. It is a present, finite scientific
result rather than a missing-evaluation artifact.

## Recommendation

The baseline is sufficient to motivate a cautious learned-replanning phase:
fixed-P performance is largely flat and seed-sensitive, while the consistent
but small P=6-over-P=4 result suggests timing could matter. Retain these fixed-P
conditions as controls and do not interpret three-seed differences as
statistical evidence. No learned replanning was implemented in this task.
