# Scout: Subgoal Persistence and Adaptive Revision in Hierarchical Reasoning

[![CI](https://github.com/Ayushichadha/scout/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayushichadha/scout/actions/workflows/ci.yml)

Scout extends [Sapient's Hierarchical Reasoning Model (HRM)](https://github.com/sapientinc/HRM)
with a manager–worker interface for persistent directional commitments in latent
space. This repository contains research code for two papers by **Ayushi Chadha**:

| Paper | arXiv | Research question |
|---|---|---|
| **When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning** | [2606.03741](https://arxiv.org/abs/2606.03741) | How long should a latent subgoal persist? |
| **Beyond the Clock: Measuring the Value of Adaptive Revision** | [2609.00874](https://arxiv.org/abs/2609.00874) | Does learned revision timing add value beyond a strong fixed schedule? |

The arXiv papers are the reference for published claims. Historical experiment
scripts, corrected timing controls, and later diagnostics represent different
protocols; their metrics should be compared only under matched settings.

## Paper 1: persistence and alignment

The first study reports benefits from intermediate persistence periods, roughly
`P=3–6`, and a directional-loss weight near `0.05`. Its five-seed best-configuration
replication reports mean LM loss `1.595` with standard deviation `0.045`.
Past-sweet-spot ablations examine interference from directional content.
See [the paper](https://arxiv.org/abs/2606.03741) for datasets, regimes, and metrics.

Historical experiment entry points include
[`run_hyperparameter_sweep.py`](scripts/run_hyperparameter_sweep.py),
[`run_baseline_comparison.py`](scripts/run_baseline_comparison.py), and
[`run_confirmation_experiments.py`](scripts/run_confirmation_experiments.py).
These historical results are distinct from the corrected, fixed-compute protocol
used in the second paper.

## Paper 2: adaptive revision and decision value

The second study evaluates three precommitted adaptive seeds with eight outer
refinement passes and intervention-count calibration. Their policy behaviors
differ, but none exceeds its best tested forced schedule on the same frozen
checkpoint. Early-clock collapse is a seed-0 result, not a finding across every seed.

| Seed | Adaptive accuracy | Best tested forced schedule | Forced accuracy |
|---|---:|---|---:|
| 0 | 46.70% | `[1,4]` | 48.75% |
| 1 | 48.61% | `[1,6]` | 48.95% |
| 2 | 47.49% | `[1,2]` | 47.55% |

These are episode-averaged token accuracies. Seed 0 additionally has an exhaustive
six-clock analysis: `[1,5]` reaches `52.12%` micro token accuracy, with only
`0.135` percentage points of held-out-label oracle headroom. The oracle uses
unavailable outcome information and is a diagnostic bound.
The counterfactual PERSIST/REPLAN analysis is also a seed-0 diagnostic.
See [the published paper](https://arxiv.org/abs/2609.00874).

Start with the [release guide](papers/beyond-the-clock/README.md) and
[result-to-code map](papers/beyond-the-clock/MAIN_RESULTS.md).

```bash
# Recompute seed-0 statistics and bootstrap intervals from published records
python scripts/reproduce_paper_main_results.py

# Regenerate the seed-0 timing/oracle figures
pip install matplotlib pandas
python experiments/schedule_sweep_k2/figures/generate_figures.py
```

The analysis command reads archived matched/frozen comparison aggregates and
recomputes the seed-0 clock variance and six-clock analysis. It does not rerun
checkpoint inference or claim a fresh three-seed training replication.

## Current implementation

The fixed and adaptive managers share a causal commitment interface:

1. The worker consumes the commitment cached in recurrent carry.
2. After that pass, the manager observes the new state and may emit a replacement.
3. A replacement after pass `m` first affects pass `m+1`.
4. Goal, gate, and worker anchor persist until replacement; episode resets are per sample.
5. Alignment measures worker displacement from the commitment anchor.

In the corrected fixed-clock implementation, `manager_period` counts outer ACT
refinement calls. With `M=8`, `[1,4]` means emissions after passes 1 and 4; pass 1
is initially unsteered, and no terminal emission is allowed after pass 8.

Cached state is detached between calls. The adaptive implementation reconstructs
a value-preserving local Jacobian on the first consuming pass to give manager and
trigger decisions local credit. This is not unrestricted backpropagation through
the entire episode. See the gradient and causal-timing tests.

## Repository layout

```text
HRM/models/subgoal_head.py             goals, gates, fixed/adaptive replanning
HRM/models/hrm/hrm_act_v1.py            recurrence, injection, local credit
HRM/models/losses.py                   task, halting, alignment, trigger losses
HRM/pretrain.py                        training, final evaluation, runtime audits
scripts/run_meta_agents_fixed_p.py     corrected fixed-clock controls
scripts/run_adaptive_400step.py        adaptive training and configuration preflight
scripts/run_adaptive_replication_audit.py  matched-budget audits for new seeds
scripts/run_frozen_schedule_control.py frozen-weight timing intervention
scripts/run_schedule_sweep_k2.py        exhaustive seed-0 clocks and oracle analysis
scripts/run_counterfactual_*           persistence diagnostics
papers/beyond-the-clock/               release guide, code map, selected results
experiments/schedule_sweep_k2/         published episode matrix, analysis, figures
tests/                                mechanism and numerical replay checks
```

HRM source is tracked directly in Scout despite the historical `.gitmodules`
entry; the modified implementation must be retained.

## Setup and validation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HRM/requirements.txt
pip install pytest ruff black
python -m pytest tests/ -q -rs
python scripts/run_meta_agents_fixed_p.py smoke --fixed-refinement-steps 8
python scripts/check_param_count.py
python scripts/verify_ablation_flags.py
```

GitHub CI runs CPU tests, lint/format checks, and ablation probes. Tests requiring
undistributed checkpoint archives report explicit skips when those files are absent.

## Training and reproducibility

The second paper's compact configuration uses hidden/goal dimension 32, two heads,
one layer and one cycle at each level, batch size 4, 400 optimizer steps, `M=8`,
feudal weight `0.05`, and intervention weight `0`. The generic Hydra defaults are
larger and are not the paper configuration. The dedicated runners supply the overrides.

```bash
# Preview commands and validate configuration; no training is launched
python scripts/run_meta_agents_fixed_p.py matrix \
  --manager-periods 4 6 --seeds 0 1 2 \
  --fixed-refinement-steps 8 --training-steps 400 --device cpu --final-eval
python scripts/run_adaptive_400step.py --seed 0
```

Add `--execute` to a runner after preparing the dataset. Full training and exact
checkpoint inference require the preprocessed data and appropriate checkpoints;
these are not bundled in the code release. Published analysis records support
statistical recomputation and figure generation. The second paper's seed-0
fixed-P checkpoints were deterministic reconstructions because the original
fixed-P runs did not save final weights.

## Cite

```bibtex
@article{chadha2026when,
  title={When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning},
  author={Chadha, Ayushi},
  journal={arXiv preprint arXiv:2606.03741},
  year={2026}
}

@article{chadha2026beyond,
  title={Beyond the Clock: Measuring the Value of Adaptive Revision},
  author={Chadha, Ayushi},
  journal={arXiv preprint arXiv:2609.00874},
  year={2026}
}
```

Scout builds on Sapient's HRM; see [`HRM/README.md`](HRM/README.md) and
[`HRM/LICENSE`](HRM/LICENSE) for upstream documentation and licensing.
