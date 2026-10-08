#!/usr/bin/env python3
"""Read-only oracle diagnostic for counterfactual learned persistence v2.

The frozen reasoning checkpoint is never written.  A neutral, newly
initialized two-signal critic is added in memory, matched persist/replan
futures are collected, and a fresh critic is fit on an episode-level training
split and evaluated on held-out episodes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HRM_ROOT))

from models.losses import counterfactual_local_objective  # noqa: E402
from models.subgoal_head import CounterfactualPersistenceCritic  # noqa: E402
from scripts.eval_checkpoints_local import (  # noqa: E402
    build_model,
    dataset_path_from_config,
    load_config,
)
from scripts.run_adaptive_matched_budget import (  # noqa: E402
    _batch_episodes,
    load_held_out_episodes,
    logical_state_sha256,
    sha256_file,
)
from utils.seeding import seed_everything  # noqa: E402

DEFAULT_RUN = ROOT / (
    "experiments/meta_agents_adaptive_400step/"
    "20260820T101100Z_adaptive_eta0_seed0_400step"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN / "checkpoints/step_400"
DEFAULT_CONFIG = DEFAULT_RUN / "resolved_config.yaml"
DEFAULT_OUTPUT = ROOT / "analysis/counterfactual_persistence_v2_diagnostic"
FIELDNAMES = (
    "episode_id",
    "pass_position",
    "progress",
    "goal_disagreement",
    "goal_similarity",
    "l_persist",
    "l_replan",
    "advantage",
    "oracle_preference",
    "critic_prediction",
    "effective_advantage",
    "split",
)


def _safe_association(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _objective(loss_model, outputs, branch: str, labels: torch.Tensor) -> torch.Tensor:
    prefix = f"counterfactual_{branch}"
    return counterfactual_local_objective(
        logits=outputs[f"{prefix}_logits"],
        q_halt_logits=outputs[f"{prefix}_q_halt_logits"],
        q_continue_logits=outputs[f"{prefix}_q_continue_logits"],
        continue_target=outputs[f"{prefix}_continue_target"],
        worker_hidden=outputs[f"{prefix}_worker_hidden"],
        goal=outputs[f"{prefix}_goal"],
        gate=outputs[f"{prefix}_gate"],
        anchor=outputs[f"{prefix}_anchor"],
        labels=labels,
        loss_fn=loss_model.loss_fn,
        feudal_loss_weight=loss_model.feudal_loss_weight,
    )


def collect_rows(
    loss_model,
    episodes,
    metadata,
    *,
    batch_size: int,
    device: str,
) -> list[dict[str, object]]:
    core = loss_model.model
    core.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch, valid_episodes in _batch_episodes(
            episodes, batch_size=batch_size, metadata=metadata
        ):
            batch = {key: value.to(device) for key, value in batch.items()}
            carry = core.initial_carry(batch)
            for pass_position in range(1, 9):
                carry, outputs = core(carry, batch)
                eligible = outputs["counterfactual_eligible"].to(torch.bool)
                if eligible.any():
                    persist = _objective(
                        loss_model, outputs, "persist", batch["labels"]
                    )
                    replan = _objective(loss_model, outputs, "replan", batch["labels"])
                    advantage = persist - replan
                    features = outputs["counterfactual_critic_feature"]
                    for index, episode in enumerate(valid_episodes):
                        if not bool(eligible[index]):
                            continue
                        value = float(advantage[index])
                        disagreement = float(features[index, 1])
                        rows.append(
                            {
                                "episode_id": episode.episode_id,
                                "pass_position": pass_position,
                                "progress": float(features[index, 0]),
                                "goal_disagreement": disagreement,
                                "goal_similarity": 1.0 - disagreement,
                                "l_persist": float(persist[index]),
                                "l_replan": float(replan[index]),
                                "advantage": value,
                                "oracle_preference": (
                                    "REPLAN" if value > 0 else "PERSIST"
                                ),
                            }
                        )
            if not carry.halted.all() or not torch.all(carry.steps == 8):
                raise RuntimeError("diagnostic violated the fixed M=8 invariant")
    return rows


def split_rows(rows, *, seed: int, held_out_fraction: float):
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    rng = np.random.Generator(np.random.Philox(seed))
    held_out_count = max(1, int(round(len(episode_ids) * held_out_fraction)))
    permutation = rng.permutation(len(episode_ids))
    held_out_ids = {episode_ids[int(index)] for index in permutation[:held_out_count]}
    train = [row for row in rows if row["episode_id"] not in held_out_ids]
    held_out = [row for row in rows if row["episode_id"] in held_out_ids]
    return train, held_out


def fit_critic(train_rows, *, seed: int, epochs: int):
    seed_everything(seed)
    critic = CounterfactualPersistenceCritic()
    x = torch.tensor(
        [[row["progress"], row["goal_disagreement"]] for row in train_rows],
        dtype=torch.float32,
    )
    y = torch.tensor([row["advantage"] for row in train_rows], dtype=torch.float32)
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.02)
    for _ in range(epochs):
        prediction = critic(x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return critic.eval()


def evaluate_predictions(critic, rows, *, cost: float) -> dict[str, float]:
    x = torch.tensor(
        [[row["progress"], row["goal_disagreement"]] for row in rows],
        dtype=torch.float32,
    )
    y = np.asarray([row["advantage"] for row in rows], dtype=np.float64)
    with torch.no_grad():
        prediction = critic(x).numpy().astype(np.float64)
    for row, score in zip(rows, prediction):
        row["critic_prediction"] = float(score)
        row["effective_advantage"] = float(score - cost)
    residual = y - prediction
    total = np.sum((y - np.mean(y)) ** 2)
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else 0.0,
        "pearson": _safe_association(prediction, y)["pearson"],
        "spearman": _safe_association(prediction, y)["spearman"],
        "sign_accuracy": float(np.mean((prediction > 0) == (y > 0))),
        "costed_decision_accuracy": float(np.mean((prediction > cost) == (y > 0))),
    }


def build_report(rows, train_rows, held_out_rows, train_metrics, held_metrics):
    advantage = np.asarray([row["advantage"] for row in rows])
    progress = np.asarray([row["progress"] for row in rows])
    disagreement = np.asarray([row["goal_disagreement"] for row in rows])
    held_advantage = np.asarray([row["advantage"] for row in held_out_rows])
    held_prediction = np.asarray([row["critic_prediction"] for row in held_out_rows])
    position = np.asarray([row["pass_position"] for row in held_out_rows], dtype=float)
    train_mean = float(np.mean([row["advantage"] for row in train_rows]))
    baseline_mse = float(np.mean((held_advantage - train_mean) ** 2))
    train_replan_rate = float(np.mean([row["advantage"] > 0 for row in train_rows]))
    baseline_class = train_replan_rate >= 0.5
    baseline_accuracy = float(np.mean((held_advantage > 0) == baseline_class))

    by_pass = {}
    for pass_value in range(2, 8):
        selected = [row for row in held_out_rows if row["pass_position"] == pass_value]
        by_pass[str(pass_value)] = {
            "count": len(selected),
            "advantage": _summary(np.asarray([row["advantage"] for row in selected])),
            "critic_prediction": _summary(
                np.asarray([row["critic_prediction"] for row in selected])
            ),
        }
    grand_mean = float(np.mean(held_prediction))
    between = sum(
        len([row for row in held_out_rows if row["pass_position"] == pass_value])
        * (
            np.mean(
                [
                    row["critic_prediction"]
                    for row in held_out_rows
                    if row["pass_position"] == pass_value
                ]
            )
            - grand_mean
        )
        ** 2
        for pass_value in range(2, 8)
    )
    total = float(np.sum((held_prediction - grand_mean) ** 2))

    return {
        "counterfactual_behavior": {
            "decision_states": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "persist_win_fraction": float(np.mean(advantage <= 0)),
            "replan_win_fraction": float(np.mean(advantage > 0)),
            "advantage_distribution": _summary(advantage),
        },
        "signal_usefulness": {
            "progress_vs_advantage": _safe_association(progress, advantage),
            "goal_disagreement_vs_advantage": _safe_association(
                disagreement, advantage
            ),
            "train_metrics": train_metrics,
            "held_out_metrics": held_metrics,
            "held_out_trivial_mean_mse": baseline_mse,
            "held_out_trivial_majority_accuracy": baseline_accuracy,
            "held_out_mse_improvement_fraction": (
                float(1.0 - held_metrics["mse"] / baseline_mse)
                if baseline_mse > 0
                else 0.0
            ),
        },
        "clock_shortcut_audit": {
            "held_out_by_pass": by_pass,
            "prediction_variance_explained_by_pass_fraction": (
                float(between / total) if total > 0 else 0.0
            ),
            "prediction_vs_pass_position": _safe_association(held_prediction, position),
        },
    }


def render_markdown(report, provenance) -> str:
    behavior = report["counterfactual_behavior"]
    signal = report["signal_usefulness"]
    held = signal["held_out_metrics"]
    clock = report["clock_shortcut_audit"]
    distribution = behavior["advantage_distribution"]
    predictive = held["r2"] > 0 and (
        held["sign_accuracy"] > signal["held_out_trivial_majority_accuracy"]
    )
    clock_concern = (
        clock["prediction_variance_explained_by_pass_fraction"] >= 0.5
        or abs(clock["prediction_vs_pass_position"]["spearman"]) >= 0.5
    )
    justified = predictive and not clock_concern
    by_pass_lines = [
        "| Pass | Mean A | Median A | Mean prediction |",
        "|---:|---:|---:|---:|",
    ]
    for pass_position, values in clock["held_out_by_pass"].items():
        by_pass_lines.append(
            f"| {pass_position} | {values['advantage']['mean']:.6g} | "
            f"{values['advantage']['median']:.6g} | "
            f"{values['critic_prediction']['mean']:.6g} |"
        )
    by_pass_table = "\n".join(by_pass_lines)
    return f"""# Counterfactual Learned Subgoal Persistence v2 Diagnostic

## Provenance and invariants

- Frozen checkpoint: `{provenance['checkpoint']}`
- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`
- Model state unchanged during collection: `{provenance['model_state_stable']}`
- Episodes / decision states: {behavior['episodes']} / {behavior['decision_states']}
- Fixed reasoning budget: exactly M=8; eligible decisions are passes 2-7.
- Critic inputs: progress and goal disagreement only. Pass position is logged only.
- Counterfactual branches start from the same latent carry; only the selected directional commitment (goal, its gate, and its emission anchor) differs.
- Branch objectives and critic target are detached. Only critic MSE updates the critic; there is no straight-through estimator.

## Counterfactual behavior

- PERSIST wins: {behavior['persist_win_fraction']:.3%}
- REPLAN wins: {behavior['replan_win_fraction']:.3%}
- Advantage mean / median: {distribution['mean']:.6g} / {distribution['median']:.6g}
- Advantage std and [5%, 95%]: {distribution['std']:.6g}, [{distribution['q05']:.6g}, {distribution['q95']:.6g}]

## Signal usefulness (held out by episode)

- Progress vs A: Pearson {signal['progress_vs_advantage']['pearson']:.4f}, Spearman {signal['progress_vs_advantage']['spearman']:.4f}
- Goal disagreement vs A: Pearson {signal['goal_disagreement_vs_advantage']['pearson']:.4f}, Spearman {signal['goal_disagreement_vs_advantage']['spearman']:.4f}
- Tiny critic: MSE {held['mse']:.6g}, R² {held['r2']:.4f}, Pearson {held['pearson']:.4f}, Spearman {held['spearman']:.4f}
- Sign accuracy: {held['sign_accuracy']:.3%}; trivial majority baseline: {signal['held_out_trivial_majority_accuracy']:.3%}
- MSE improvement over train-mean baseline: {signal['held_out_mse_improvement_fraction']:.3%}

## Clock-shortcut audit

- Prediction variance explained by pass: {clock['prediction_variance_explained_by_pass_fraction']:.3%}
- Prediction vs position: Pearson {clock['prediction_vs_pass_position']['pearson']:.4f}, Spearman {clock['prediction_vs_pass_position']['spearman']:.4f}

{by_pass_table}

## Scientific stop decision

The two signals are **{'predictive' if predictive else 'not predictively adequate'}** on this split. The clock audit is **{'concerning' if clock_concern else 'not concerning'}**. A full run is therefore **{'justified' if justified else 'not yet justified'}**. This is a small single-checkpoint diagnostic, not a training result.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--held-out-fraction", type=float, default=0.30)
    parser.add_argument("--critic-epochs", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.episodes < 4:
        raise ValueError("at least four episodes are required")

    seed_everything(args.seed)
    config = copy.deepcopy(load_config(args.config))
    config["arch"]["subgoal_head"]["replan_mode"] = "counterfactual_v2"
    config["arch"]["subgoal_head"]["counterfactual_replan_cost"] = 0.01
    config["arch"]["loss"]["counterfactual_critic_weight"] = 1.0
    dataset_path = dataset_path_from_config(config)
    metadata, all_episodes = load_held_out_episodes(dataset_path)
    episodes = all_episodes[: args.episodes]
    loss_model = build_model(config, metadata, args.batch_size, args.device)

    state = torch.load(args.checkpoint, map_location=args.device)
    if all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    incompatible = loss_model.load_state_dict(state, strict=False)
    expected_missing = {
        key
        for key in loss_model.state_dict()
        if key.startswith("model.counterfactual_critic.")
    }
    expected_unexpected = {
        "model.adaptive_trigger.linear.weight",
        "model.adaptive_trigger.linear.bias",
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or set(incompatible.unexpected_keys) != expected_unexpected
    ):
        raise RuntimeError(
            "checkpoint mismatch beyond the new critic: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    before = logical_state_sha256(loss_model)
    rows = collect_rows(
        loss_model,
        episodes,
        metadata,
        batch_size=args.batch_size,
        device=args.device,
    )
    after = logical_state_sha256(loss_model)
    if before != after:
        raise RuntimeError("model state changed during diagnostic collection")
    if not rows or not all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "progress",
            "goal_disagreement",
            "l_persist",
            "l_replan",
            "advantage",
        )
    ):
        raise RuntimeError("diagnostic produced empty or non-finite observations")

    train_rows, held_out_rows = split_rows(
        rows, seed=args.seed, held_out_fraction=args.held_out_fraction
    )
    critic = fit_critic(train_rows, seed=args.seed, epochs=args.critic_epochs)
    cost = config["arch"]["subgoal_head"]["counterfactual_replan_cost"]
    train_metrics = evaluate_predictions(critic, train_rows, cost=cost)
    held_metrics = evaluate_predictions(critic, held_out_rows, cost=cost)
    for row in train_rows:
        row["split"] = "train"
    for row in held_out_rows:
        row["split"] = "held_out"
    ordered_rows = sorted(
        train_rows + held_out_rows,
        key=lambda row: (str(row["episode_id"]), int(row["pass_position"])),
    )
    report = build_report(
        ordered_rows, train_rows, held_out_rows, train_metrics, held_metrics
    )
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config.resolve()),
        "seed": args.seed,
        "episode_selection": f"first {args.episodes} held-out episodes",
        "held_out_split_unit": "episode_id",
        "held_out_fraction": args.held_out_fraction,
        "critic_architecture": "Linear(2,4)-Tanh-Linear(4,1)",
        "critic_inputs": ["progress", "goal_disagreement"],
        "audit_only_fields": ["pass_position"],
        "model_state_before": before,
        "model_state_after": after,
        "model_state_stable": before == after,
        "missing_checkpoint_keys": sorted(expected_missing),
        "unused_adaptive_v1_checkpoint_keys": sorted(expected_unexpected),
    }
    report["provenance"] = provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "decision_states.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered_rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    markdown = render_markdown(report, provenance)
    (args.output_dir / "REPORT.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
