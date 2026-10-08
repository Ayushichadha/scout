# Learned Re-Planning v1 Adaptive Eta=0 Sanity Pilot

## Final verdict

**SANITY PILOT PARTIAL — INSPECT BEFORE FURTHER RUNS**

The adaptive trigger learned from downstream credit and its parameters moved
away from zero. Training remained stochastic and non-degenerate. However, all
held-out beta values finished below the strict `0.5` evaluation threshold, so
deterministic evaluation collapsed to the initial-emission-only schedule in
every episode. The mechanism is alive, but the first short pilot did not
produce a usable deterministic adaptive policy.

This is a mechanism sanity result, not a performance result.

## Run validity

- Label: `SANITY PILOT — NOT FINAL EXPERIMENT`
- Seed: `0`
- Completed optimizer steps: `96`
- Dataset: `ConceptARC-mini`
- Global batch size: `4`
- Device: `cpu`
- Reruns of the 96-step configuration: `0`
- Runtime configuration changes: `0`
- Runtime/invariant failures: `0`
- NaN/Inf: `0`
- Evaluation schema errors: `0`
- Checkpoint/config mismatches: `0`

The earlier 100-step request was rejected before model construction because
100 is not divisible by `M=8`. After explicit authorization, exactly one
96-step attempt was launched. Before launch, the resolved config was diffed
against the rejected config. The only differences were `max_steps: 100 -> 96`
and the permitted checkpoint/output path.

## Exact command

```bash
/Users/ayushi/Documents/hrm_research/.venv/bin/python /Users/ayushi/Documents/hrm_research/scripts/run_adaptive_sanity_pilot.py --execute
```

## Resolved configuration

- `replan_mode=adaptive`
- `fixed_refinement_steps=8`
- `directional_displacement=true`
- `detach_goals=true`
- `trigger_stochastic_train=true`
- evaluation policy: strict `beta > 0.5`
- `trigger_threshold=0.5`
- `intervention_weight=0.0`
- directional/feudal loss weight: `0.05`
- seed/effective process seed/rank: `0 / 0 / 0`
- optimizer steps: `96`
- global batch size: `4`
- dataset: `data/conceptarc-mini`
- learning rate: `1e-4`
- LR warmup: `0`
- optimizer beta values: `0.9, 0.95`
- weight decay: `0.1`
- puzzle embedding LR/weight decay: `0.01 / 0.1`
- hidden size / goal dimension: `32 / 32`
- H/L layers: `1 / 1`
- H/L cycles: `1 / 1`
- heads: `2`
- expansion: `2`
- halt max steps: `4`
- fixed compute M: `8`

The saved `run_name` retains the earlier `100step` label because the pre-launch
resolved-config gate permitted no metadata changes beyond `max_steps` and
output paths. The actual budget and checkpoint are unambiguously 96 steps.

## Training and trigger learning

All trigger parameters initialized at exactly zero.

| Step | w_c | w_d | w_rho | w_dwell | w_q | w_alpha | bias | Mean beta | Hard train rate | Grad norm mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.500000 | — | — |
| 10 | -0.000361 | -0.000317 | -0.000263 | -0.000473 | 0.000000 | -0.000484 | -0.000500 | 0.499939 | 0.4286 | 0.001418 |
| 25 | -0.001269 | -0.001085 | -0.000837 | -0.001121 | 0.0000159 | -0.001014 | -0.001197 | 0.499655 | 0.6136 | 0.002090 |
| 50 | -0.001942 | -0.001622 | -0.001104 | -0.001890 | 0.0000123 | -0.001838 | -0.002293 | 0.499315 | 0.5526 | 0.001496 |
| 96 | -0.002074 | -0.001829 | -0.000579 | -0.001958 | 0.0000151 | -0.002075 | -0.003430 | 0.498858 | 0.4929 | 0.001364 |

Final trigger parameters:

```text
w_c      = -0.0020735294
w_d      = -0.0018290979
w_rho    = -0.0005789109
w_dwell  = -0.0019584675
w_q      =  0.0000150749
w_alpha  = -0.0020746570
bias     = -0.0034299861
```

The trigger clearly moved. Most learned coefficients and the bias moved
negative, with `q` remaining close to zero.

Selected training-window diagnostics:

| Ending step | Total loss/pass | LM loss/pass | Directional loss/pass | Directional cosine | Mean gate |
|---:|---:|---:|---:|---:|---:|
| 10 | 2.729261 | 2.691192 | 0.292438 | 0.056101 | 0.383668 |
| 25 | 2.709102 | 2.668258 | 0.349126 | 0.059720 | 0.418369 |
| 50 | 2.330070 | 2.289026 | 0.355154 | 0.042815 | 0.418057 |
| 96 | 2.275366 | 2.235421 | 0.335979 | 0.084078 | 0.405671 |

## Gradient sanity

- Trigger-gradient observations: `96` steps
- Steps with nonzero trigger gradient: `72`
- Mean trigger gradient norm: `0.00151759`
- Maximum trigger gradient norm: `0.00640696`
- Intervention weight eta: `0`
- Intervention-cost gradient contribution: exactly `0`

The nonzero updates therefore came from downstream task/alignment credit, not
from intervention regularization. Zero-gradient steps are consistent with the
one-consuming-pass causal phase: bootstrap and current decisions do not all
have downstream credit on the same outer call.

## Held-out task diagnostics

Evaluation covered `4,607` completed episodes.

- Token/individual accuracy: `0.43855595`
- Exact accuracy: `0.0`
- Eight-pass accumulated LM loss per episode: `17.36353368`
- Corresponding LM loss per outer pass: `2.17044171`

These values are diagnostic only. They must not be compared as performance
evidence against the 400-step fixed-P reference.

## Held-out beta and intervention behavior

- Eligible beta observations: `27,642` (`4,607 * 6`)
- Mean beta: `0.49870937`
- Population standard deviation: `0.00012200`
- Minimum / maximum: `0.49843776 / 0.49912179`
- Hard adaptive intervention rate at threshold 0.5: `0.0`
- Mean total interventions per episode: `1.0`
- Mean adaptive interventions per episode: `0.0`
- Fraction with one total intervention: `1.0`
- Fraction with seven total interventions: `0.0`

Adaptive intervention positions:

| After pass | Count | Fraction of episodes |
|---:|---:|---:|
| 2 | 0 | 0.0 |
| 3 | 0 | 0.0 |
| 4 | 0 | 0.0 |
| 5 | 0 | 0.0 |
| 6 | 0 | 0.0 |
| 7 | 0 | 0.0 |

The stochastic training policy remained near a 50% hard rate, but the learned
negative bias put every deterministic held-out beta below the strict threshold.
This is an evaluation-policy collapse to always retain.

## Dwell behavior

- Mean completed dwell: `7.0`
- Dwell histogram: `{7: 4607}`
- Inter-intervention interval distribution: `{7: 4607}`

Every episode used only the mandatory initial commitment.

## State-dependence diagnostics

Mean held-out beta by dwell:

| Dwell | Mean beta | Within-dwell std |
|---:|---:|---:|
| 1 | 0.49895470 | 0.00004235 |
| 2 | 0.49867339 | 0.00005166 |
| 3 | 0.49862822 | 0.00005001 |
| 4 | 0.49864793 | 0.00002678 |
| 5 | 0.49868825 | 0.00004854 |
| 6 | 0.49866373 | 0.00006999 |

There is measurable sample-level variation at each dwell, but it is very
small. Dwell 1 is visibly separated from later positions, and every hard
decision is retain, so the pilot does not demonstrate meaningful deterministic
state-dependent scheduling.

Feature means at hard retain points:

| Feature | Retain mean | Intervene mean |
|---|---:|---:|
| c | 0.06346161 | unavailable |
| d | 0.00836096 | unavailable |
| rho | 0.00522599 | unavailable |
| dwell/M | 0.43750000 | unavailable |
| q | -0.00062528 | unavailable |
| alpha | 0.34984146 | unavailable |

No intervene-conditioned means exist because evaluation produced no hard
replacements.

## Goal-change behavior

- Actual held-out replacements: `0`
- Mean cosine(old goal, new goal): unavailable

The goal-change diagnostic cannot distinguish real versus nominal changes in
this pilot because strict-threshold evaluation emitted no replacement.

## Fixed-compute and evaluator invariants

- M for every completed episode: `8`
- Eligible adaptive decisions per episode: `6`
- Terminal emissions: `0`
- Fixed-compute violations: `0`
- Every episode's completed dwell lengths summed to `7`
- Dwell segment count equaled commitment count for every episode
- All recorded metrics finite: `true`
- Evaluator schema errors: `0`

## Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| A — trigger moved | PASS | Six weights and bias moved clearly from zero; final bias was -0.00343. |
| B — beta non-degenerate | WARNING | Beta varied across 27,642 states, but std was only 0.000122 and all values remained below 0.5. |
| C — state dependence | FAIL | Small within-dwell variation exists, but every episode followed the same deterministic retain schedule. |
| D — no collapse | FAIL | 100% of episodes had exactly one total intervention. |
| E — downstream credit active | PASS | 72/96 steps had nonzero trigger gradient with eta exactly zero. |
| F — causal/evaluation invariants | PASS | M=8, six eligible decisions, zero terminal emissions/violations, finite metrics, valid dwell sums. |

## Interpretation and next decision

The implementation passed its causal, numerical, gradient, and accounting
sanity checks. It did not pass the behavioral part of the sanity pilot under
strict threshold 0.5: the learned policy shifted only slightly below 0.5 and
therefore became deterministically always-retain.

Do not launch an eta sweep or full adaptive experiment automatically. Inspect
the threshold/initialization interaction and decide whether a threshold-only
sensitivity analysis (no retraining) is warranted before designing further
runs. No such sensitivity analysis was performed here.

## Provenance and artifacts

- Git commit: `3a0721ce21250376edee24e4e5aaad228db39de3`
- Branch: `subgoal_ablations`
- Worktree: dirty; exact pre-run status is in `provenance.json`
- Result directory:
  `experiments/meta_agents_adaptive_sanity/20260816T144122Z_adaptive_eta0_seed0_96step_sanity/`
- Result: `pilot_result.json`
- Events/training snapshots: `events.jsonl`
- Resolved configuration: `resolved_config.yaml`
- Provenance: `provenance.json`
- Checkpoint: `checkpoints/step_96`
