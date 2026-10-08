#!/usr/bin/env python3
"""One-shot Learned Re-Planning v1 sanity pilot (not a performance run)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
OUTPUT_ROOT = ROOT / "experiments" / "meta_agents_adaptive_sanity"
sys.path.insert(0, str(HRM_ROOT))

from models.losses import accumulate_episode_metrics  # noqa: E402
from pretrain import (  # noqa: E402
    PretrainConfig,
    create_dataloader,
    init_train_state,
    save_train_state,
    train_batch,
    validate_training_config,
)
from utils.seeding import seed_everything  # noqa: E402


SNAPSHOT_STEPS = (0, 10, 25, 50, 96)
FEATURE_NAMES = ("c", "d", "rho", "dwell", "q", "alpha")


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    status = run("status", "--short").splitlines()
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "dirty_paths": status,
    }


def build_overrides(run_dir: Path) -> list[str]:
    return [
        "device=cpu",
        "seed=0",
        "max_steps=96",
        "enable_wandb=false",
        "data_path=data/conceptarc-mini",
        "global_batch_size=4",
        "epochs=1",
        "eval_interval=null",
        "+final_eval=true",
        "lr_warmup_steps=0",
        f"+checkpoint_path={run_dir / 'checkpoints'}",
        "+project_name=meta_agents_adaptive_sanity",
        "+run_name=adaptive_eta0_seed0_100step_sanity",
        "arch.fixed_refinement_steps=8",
        "arch.halt_max_steps=4",
        "arch.hidden_size=32",
        "arch.subgoal_head.hidden_size=32",
        "arch.subgoal_head.goal_dim=32",
        "arch.H_layers=1",
        "arch.L_layers=1",
        "arch.H_cycles=1",
        "arch.L_cycles=1",
        "arch.num_heads=2",
        "arch.expansion=2",
        "arch.puzzle_emb_ndim=32",
        "arch.loss.feudal_loss_weight=0.05",
        "arch.loss.intervention_weight=0.0",
        "arch.subgoal_head.replan_mode=adaptive",
        "arch.subgoal_head.directional_displacement=true",
        "arch.subgoal_head.detach_goals=true",
        "arch.subgoal_head.initial_goal_only=false",
        "arch.subgoal_head.trigger_stochastic_train=true",
        "arch.subgoal_head.trigger_threshold=0.5",
    ]


def resolve_config(overrides: list[str]) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(HRM_ROOT / "config"), version_base=None):
        config = compose(config_name="cfg_pretrain", overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    GlobalHydra.instance().clear()
    assert isinstance(resolved, dict)
    return resolved


def unwrap_model(loss_model):
    model = loss_model.model
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def trigger_parameters(loss_model) -> dict[str, float]:
    trigger = unwrap_model(loss_model).adaptive_trigger
    if trigger is None:
        raise RuntimeError("sanity pilot did not instantiate the adaptive trigger")
    weights = trigger.linear.weight.detach().cpu().squeeze(0)
    return {
        **{
            f"w_{name}": float(weights[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "bias": float(trigger.linear.bias.detach().cpu().squeeze(0)),
    }


def metric_delta(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float]:
    return {key: value - previous.get(key, 0.0) for key, value in current.items()}


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def training_snapshot(
    *,
    step: int,
    parameters: dict[str, float],
    interval: dict[str, float] | None,
    gradient_norms: list[float],
) -> dict[str, Any]:
    if interval is None:
        return {
            "step": step,
            "parameters": parameters,
            "mean_beta": 0.5,
            "hard_adaptive_intervention_rate": None,
            "mean_gate": None,
            "mean_directional_cosine": None,
            "lm_loss_per_pass": None,
            "directional_loss_per_pass": None,
            "total_loss_per_pass": None,
            "gradient_norm_mean": None,
            "gradient_norm_max": None,
        }
    executed = interval.get("executed_refinement_passes", 0.0)
    beta_count = interval.get("trigger_probability_count", 0.0)
    gate_count = interval.get("active_gate_count", 0.0)
    cosine_count = interval.get("directional_cosine_count", 0.0)
    lm = safe_ratio(interval.get("lm_loss", 0.0), executed)
    q = safe_ratio(
        interval.get("q_halt_loss", 0.0) + interval.get("q_continue_loss", 0.0),
        executed,
    )
    directional = safe_ratio(interval.get("feudal_loss", 0.0), executed)
    intervention = safe_ratio(interval.get("intervention_loss", 0.0), executed)
    total = None
    if lm is not None and q is not None and directional is not None:
        total = lm + 0.5 * q + 0.05 * directional + 0.0 * (intervention or 0.0)
    return {
        "step": step,
        "parameters": parameters,
        "mean_beta": safe_ratio(
            interval.get("trigger_probability_sum", 0.0), beta_count
        ),
        "hard_adaptive_intervention_rate": safe_ratio(
            interval.get("adaptive_interventions", 0.0), beta_count
        ),
        "mean_gate": safe_ratio(interval.get("active_gate_sum", 0.0), gate_count),
        "mean_directional_cosine": safe_ratio(
            interval.get("directional_cosine_sum", 0.0), cosine_count
        ),
        "lm_loss_per_pass": lm,
        "directional_loss_per_pass": directional,
        "total_loss_per_pass": total,
        "gradient_norm_mean": (
            sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        ),
        "gradient_norm_max": max(gradient_norms, default=0.0),
    }


def deterministic_evaluate(loss_model, eval_loader) -> dict[str, Any]:
    loss_model.eval()
    aggregate_metrics: dict[str, float] = {}
    episode_total_interventions: list[int] = []
    episode_adaptive_interventions: list[int] = []
    position_counts = Counter()
    dwell_counts = Counter()
    beta_values: list[float] = []
    beta_by_dwell: dict[int, list[float]] = defaultdict(list)
    features_by_decision: dict[str, list[list[float]]] = {
        "intervene": [],
        "retain": [],
    }
    goal_cosines: list[float] = []
    episode_count = 0
    return_keys = [
        "subgoal_updated",
        "adaptive_trigger_probability",
        "adaptive_trigger_hard",
        "adaptive_trigger_eligible",
        "adaptive_trigger_feature",
        "adaptive_dwell_after_pass",
        "adaptive_completed_dwell",
        "adaptive_completed_dwell_mask",
        "adaptive_old_new_goal_cosine",
        "subgoal_has_next_worker_pass",
    ]

    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            valid = (batch["labels"] != -100).any(dim=-1)
            batch_size = batch["labels"].shape[0]
            total_per_sample = torch.zeros(batch_size, dtype=torch.long)
            adaptive_per_sample = torch.zeros(batch_size, dtype=torch.long)
            dwell_sum_per_sample = torch.zeros(batch_size, dtype=torch.long)
            dwell_segments_per_sample = torch.zeros(batch_size, dtype=torch.long)
            carry = loss_model.initial_carry(batch)
            episode_metrics = None

            for pass_index in range(1, 9):
                carry, _loss, metrics, outputs, all_finish = loss_model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                if not all(torch.isfinite(value).all() for value in metrics.values()):
                    raise FloatingPointError(
                        f"non-finite evaluation metric on pass {pass_index}"
                    )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
                if bool(all_finish) != (pass_index == 8):
                    raise AssertionError(
                        f"evaluation M invariant failed on pass {pass_index}"
                    )

                updated = outputs["subgoal_updated"].to(torch.bool) & valid
                total_per_sample += updated.to(torch.long)
                eligible = outputs["adaptive_trigger_eligible"].to(torch.bool) & valid
                hard = outputs["adaptive_trigger_hard"].to(torch.bool) & eligible
                adaptive_per_sample += hard.to(torch.long)
                if eligible.any():
                    probabilities = outputs["adaptive_trigger_probability"]
                    dwell = outputs["adaptive_dwell_after_pass"]
                    features = outputs["adaptive_trigger_feature"]
                    for index in eligible.nonzero(as_tuple=False).flatten().tolist():
                        beta = float(probabilities[index])
                        beta_values.append(beta)
                        beta_by_dwell[int(dwell[index])].append(beta)
                        condition = "intervene" if bool(hard[index]) else "retain"
                        features_by_decision[condition].append(
                            [float(value) for value in features[index]]
                        )
                    if pass_index not in range(2, 8):
                        raise AssertionError(
                            f"adaptive decision was eligible on pass {pass_index}"
                        )
                    position_counts[pass_index] += int(hard.sum())
                    cosines = outputs["adaptive_old_new_goal_cosine"]
                    goal_cosines.extend(float(value) for value in cosines[hard])

                completed = (
                    outputs["adaptive_completed_dwell_mask"].to(torch.bool) & valid
                )
                completed_dwell = outputs["adaptive_completed_dwell"]
                dwell_sum_per_sample += torch.where(
                    completed, completed_dwell, torch.zeros_like(completed_dwell)
                )
                dwell_segments_per_sample += completed.to(torch.long)
                for value in completed_dwell[completed].tolist():
                    dwell_counts[int(value)] += 1

                terminal = updated & ~outputs["subgoal_has_next_worker_pass"].to(
                    torch.bool
                )
                if terminal.any():
                    raise AssertionError("terminal manager emission detected")

            assert episode_metrics is not None
            for key, value in episode_metrics.items():
                aggregate_metrics[key] = aggregate_metrics.get(key, 0.0) + float(value)

            valid_indices = valid.nonzero(as_tuple=False).flatten()
            for index in valid_indices.tolist():
                total = int(total_per_sample[index])
                adaptive = int(adaptive_per_sample[index])
                dwell_sum = int(dwell_sum_per_sample[index])
                segments = int(dwell_segments_per_sample[index])
                if dwell_sum != 7:
                    raise AssertionError(
                        f"episode dwell sum was {dwell_sum}, expected 7"
                    )
                if segments != total:
                    raise AssertionError(
                        f"episode had {segments} dwell segments but {total} commitments"
                    )
                episode_total_interventions.append(total)
                episode_adaptive_interventions.append(adaptive)
            episode_count += int(valid.sum())

    eligible_expected = 6 * episode_count
    if len(beta_values) != eligible_expected:
        raise AssertionError(
            f"eligible decision count was {len(beta_values)}, expected {eligible_expected}"
        )
    count = aggregate_metrics["count"]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def population_std(values: list[float]) -> float | None:
        if not values:
            return None
        value_mean = mean(values)
        assert value_mean is not None
        return math.sqrt(
            sum((value - value_mean) ** 2 for value in values) / len(values)
        )

    conditional_features = {}
    for condition, rows in features_by_decision.items():
        conditional_features[condition] = {
            name: (mean([row[index] for row in rows]) if rows else None)
            for index, name in enumerate(FEATURE_NAMES)
        }

    result = {
        "episodes": episode_count,
        "task": {
            "token_accuracy": aggregate_metrics["accuracy"] / count,
            "exact_accuracy": aggregate_metrics["exact_accuracy"] / count,
            "lm_loss": aggregate_metrics["lm_loss"] / count,
        },
        "interventions": {
            "mean_total_per_episode": mean(episode_total_interventions),
            "mean_adaptive_per_episode": mean(episode_adaptive_interventions),
            "fraction_one_total": episode_total_interventions.count(1) / episode_count,
            "fraction_seven_total": episode_total_interventions.count(7)
            / episode_count,
            "position_counts": {str(pos): position_counts[pos] for pos in range(2, 8)},
            "position_fractions": {
                str(pos): position_counts[pos] / episode_count for pos in range(2, 8)
            },
        },
        "beta": {
            "count": len(beta_values),
            "mean": mean(beta_values),
            "std": population_std(beta_values),
            "min": min(beta_values),
            "max": max(beta_values),
            "hard_rate": sum(episode_adaptive_interventions) / len(beta_values),
            "by_dwell": {
                str(dwell): {
                    "count": len(values),
                    "mean": mean(values),
                    "std": population_std(values),
                }
                for dwell, values in sorted(beta_by_dwell.items())
            },
        },
        "dwell": {
            "mean": safe_ratio(
                sum(length * frequency for length, frequency in dwell_counts.items()),
                sum(dwell_counts.values()),
            ),
            "histogram": {
                str(key): value for key, value in sorted(dwell_counts.items())
            },
        },
        "features": conditional_features,
        "goal_change": {
            "count": len(goal_cosines),
            "mean_old_new_cosine": mean(goal_cosines),
        },
        "invariants": {
            "M": 8,
            "eligible_per_episode": len(beta_values) / episode_count,
            "terminal_emissions": aggregate_metrics.get(
                "unconsumed_terminal_emissions", 0.0
            ),
            "fixed_compute_violations": aggregate_metrics.get(
                "fixed_compute_violations", 0.0
            ),
            "all_finite": True,
        },
    }
    return result


def run_pilot(run_dir: Path, resolved: dict[str, Any], command: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    provenance = git_provenance()
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    events_path = run_dir / "events.jsonl"

    def event(payload: dict[str, Any]) -> None:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    event({"event": "start", "label": "SANITY PILOT — NOT FINAL EXPERIMENT"})
    os.chdir(HRM_ROOT)
    config = PretrainConfig(**resolved)
    effective_seed = seed_everything(config.seed, rank=0)
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        rank=0,
        world_size=1,
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, _eval_metadata = create_dataloader(
        config,
        "test",
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    validate_training_config(config, train_metadata, rank=0)
    train_state = init_train_state(config, train_metadata, world_size=1)
    if train_state.total_steps != 96:
        raise AssertionError(
            f"resolved training budget was {train_state.total_steps}, not 96"
        )

    core_model = unwrap_model(train_state.model)
    trigger = core_model.adaptive_trigger
    if trigger is None:
        raise AssertionError("adaptive trigger missing")
    gradient_parts: dict[int, dict[str, float]] = defaultdict(dict)

    def capture(name: str):
        def hook(gradient: torch.Tensor):
            gradient_parts[train_state.step][name] = float(gradient.float().norm())

        return hook

    hooks = [
        trigger.linear.weight.register_hook(capture("weight")),
        trigger.linear.bias.register_hook(capture("bias")),
    ]
    snapshots = [
        training_snapshot(
            step=0,
            parameters=trigger_parameters(train_state.model),
            interval=None,
            gradient_norms=[],
        )
    ]
    event({"event": "snapshot", **snapshots[0]})
    previous_totals: dict[str, float] = {}
    previous_snapshot_step = 0

    try:
        for _set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=0,
                world_size=1,
            )
            if metrics is None:
                raise FloatingPointError(
                    f"training step {train_state.step} returned no metrics"
                )
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError(
                    f"non-finite training metric at step {train_state.step}"
                )
            if train_state.step in SNAPSHOT_STEPS[1:]:
                totals = dict(train_state.metric_totals)
                interval = metric_delta(totals, previous_totals)
                norms = []
                for step in range(previous_snapshot_step + 1, train_state.step + 1):
                    parts = gradient_parts.get(step, {})
                    norms.append(
                        math.sqrt(sum(value * value for value in parts.values()))
                    )
                snapshot = training_snapshot(
                    step=train_state.step,
                    parameters=trigger_parameters(train_state.model),
                    interval=interval,
                    gradient_norms=norms,
                )
                snapshots.append(snapshot)
                event({"event": "snapshot", **snapshot})
                previous_totals = totals
                previous_snapshot_step = train_state.step
            if train_state.step >= train_state.total_steps:
                break

        if train_state.step != 96:
            raise AssertionError(
                f"pilot stopped at step {train_state.step}, expected 96"
            )
        for hook in hooks:
            hook.remove()

        all_gradient_norms = []
        for parts in gradient_parts.values():
            all_gradient_norms.append(
                math.sqrt(sum(value * value for value in parts.values()))
            )
        nonzero_gradient_norms = [value for value in all_gradient_norms if value > 0]
        if not nonzero_gradient_norms:
            raise AssertionError("zero downstream adaptive-trigger gradient")

        save_train_state(config, train_state)
        checkpoint = Path(config.checkpoint_path) / f"step_{train_state.step}"
        evaluation = deterministic_evaluate(train_state.model, eval_loader)
        final_parameters = trigger_parameters(train_state.model)
        payload = {
            "label": "SANITY PILOT — NOT FINAL EXPERIMENT",
            "status": "completed",
            "command": command,
            "resolved_configuration": resolved,
            "rng": {
                "requested_seed": config.seed,
                "effective_process_seed": effective_seed,
                "rank": 0,
                "torch_initial_seed": torch.initial_seed(),
                "training_policy": "eligible-only Bernoulli(beta)",
                "evaluation_policy": "beta > 0.5",
            },
            "training": {
                "completed_optimizer_steps": train_state.step,
                "snapshots": snapshots,
                "trigger_gradient": {
                    "observed_steps": len(all_gradient_norms),
                    "nonzero_steps": len(nonzero_gradient_norms),
                    "mean_norm": sum(all_gradient_norms) / len(all_gradient_norms),
                    "max_norm": max(all_gradient_norms),
                    "intervention_weight": 0.0,
                    "intervention_cost_contribution": 0.0,
                },
            },
            "final_trigger_parameters": final_parameters,
            "evaluation": evaluation,
            "checkpoint_path": str(checkpoint),
            "provenance": provenance,
        }
        (run_dir / "pilot_result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        event(
            {
                "event": "completed",
                "checkpoint": str(checkpoint),
                "result": str(run_dir / "pilot_result.json"),
            }
        )
    except Exception as error:
        failure = {
            "label": "SANITY PILOT — NOT FINAL EXPERIMENT",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_optimizer_steps": train_state.step,
            "command": command,
            "resolved_configuration": resolved,
            "provenance": provenance,
        }
        (run_dir / "pilot_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        event({"event": "failed", **failure})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"{timestamp}_adaptive_eta0_seed0_96step_sanity"
    overrides = build_overrides(run_dir)
    resolved = resolve_config(overrides)
    command = [sys.executable, str(Path(__file__).resolve()), "--execute"]
    if not args.execute:
        print("DRY RUN — no training")
        print("command:", " ".join(command))
        print("result_directory:", run_dir)
        print(OmegaConf.to_yaml(resolved))
        return
    run_pilot(run_dir, resolved, command)


if __name__ == "__main__":
    main()
