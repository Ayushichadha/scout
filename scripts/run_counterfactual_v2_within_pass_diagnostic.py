#!/usr/bin/env python3
"""Leakage-safe within-pass analysis for persistence-v2 counterfactual states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
sys.path.insert(0, str(HRM_ROOT))

from models.subgoal_head import CounterfactualPersistenceCritic  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

DEFAULT_INPUT = (
    ROOT / "analysis/counterfactual_persistence_v2_diagnostic/decision_states.csv"
)
DEFAULT_OUTPUT = ROOT / "analysis/counterfactual_persistence_v2_within_pass"
PASSES = tuple(range(2, 8))
NUMERIC_FIELDS = (
    "pass_position",
    "progress",
    "goal_disagreement",
    "goal_similarity",
    "l_persist",
    "l_replan",
    "advantage",
    "critic_prediction",
    "effective_advantage",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_association(x, y) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def regression_metrics(target, prediction) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    total = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mse": float(np.mean(error**2)),
        "r2": float(1.0 - np.sum(error**2) / total) if total > 0 else 0.0,
        **safe_association(prediction, target),
    }


def classification_metrics(target_positive, score) -> dict[str, float | None]:
    target = np.asarray(target_positive, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    prediction = score > 0
    has_both_classes = len(np.unique(target)) == 2
    result = {
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction))
            if has_both_classes
            else None
        ),
        "replan_precision": float(precision_score(target, prediction, zero_division=0)),
        "replan_recall": float(recall_score(target, prediction, zero_division=0)),
    }
    result["auroc"] = float(roc_auc_score(target, score)) if has_both_classes else None
    return result


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])
        row["pass_position"] = int(row["pass_position"])
    if len(rows) != 384:
        raise RuntimeError(f"expected 384 decision states, found {len(rows)}")
    episode_ids = {row["episode_id"] for row in rows}
    if len(episode_ids) != 64:
        raise RuntimeError(f"expected 64 episodes, found {len(episode_ids)}")
    for pass_position in PASSES:
        count = sum(row["pass_position"] == pass_position for row in rows)
        if count != 64:
            raise RuntimeError(
                f"pass {pass_position}: expected 64 states, found {count}"
            )
    if {row["split"] for row in rows} != {"train", "held_out"}:
        raise RuntimeError("expected the original episode-level train/held-out split")
    split_by_episode: dict[object, set[object]] = {}
    for row in rows:
        split_by_episode.setdefault(row["episode_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in split_by_episode.values()):
        raise RuntimeError("an episode crosses the train/held-out boundary")
    if not all(
        math.isfinite(float(row[field])) for row in rows for field in NUMERIC_FIELDS
    ):
        raise RuntimeError("input contains non-finite values")
    return rows


def within_pass_statistics(rows) -> dict[str, object]:
    result = {}
    for pass_position in PASSES:
        selected = [row for row in rows if row["pass_position"] == pass_position]
        advantage = np.asarray([row["advantage"] for row in selected])
        progress = np.asarray([row["progress"] for row in selected])
        disagreement = np.asarray([row["goal_disagreement"] for row in selected])
        result[str(pass_position)] = {
            "states": len(selected),
            "replan_win_fraction": float(np.mean(advantage > 0)),
            "advantage_mean": float(np.mean(advantage)),
            "advantage_median": float(np.median(advantage)),
            "progress_vs_advantage": safe_association(progress, advantage),
            "goal_disagreement_vs_advantage": safe_association(disagreement, advantage),
        }
    return result


def training_pass_means(train_rows) -> dict[int, dict[str, float]]:
    means = {}
    for pass_position in PASSES:
        selected = [row for row in train_rows if row["pass_position"] == pass_position]
        if not selected:
            raise RuntimeError(f"training split has no states at pass {pass_position}")
        means[pass_position] = {
            field: float(np.mean([row[field] for row in selected]))
            for field in ("advantage", "progress", "goal_disagreement")
        }
    return means


def residualize(rows, pass_means) -> None:
    """Center using training-split means only, including for held-out rows."""

    for row in rows:
        means = pass_means[int(row["pass_position"])]
        row["advantage_residual"] = float(row["advantage"] - means["advantage"])
        row["progress_residual"] = float(row["progress"] - means["progress"])
        row["disagreement_residual"] = float(
            row["goal_disagreement"] - means["goal_disagreement"]
        )
        row["pass_only_prediction"] = means["advantage"]


def fit_residual_critic(train_rows, *, seed: int, epochs: int):
    seed_everything(seed)
    critic = CounterfactualPersistenceCritic()
    features = torch.tensor(
        [
            [row["progress_residual"], row["disagreement_residual"]]
            for row in train_rows
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [row["advantage_residual"] for row in train_rows], dtype=torch.float32
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.02)
    for _ in range(epochs):
        prediction = critic(features)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return critic.eval()


def apply_residual_critic(critic, rows) -> None:
    features = torch.tensor(
        [[row["progress_residual"], row["disagreement_residual"]] for row in rows],
        dtype=torch.float32,
    )
    with torch.no_grad():
        predictions = critic(features).numpy().astype(np.float64)
    for row, prediction in zip(rows, predictions):
        row["residual_critic_prediction"] = float(prediction)
        row["combined_prediction"] = float(row["pass_only_prediction"] + prediction)


def variance_explained_by_pass(rows, field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    grand_mean = float(np.mean(values))
    between = 0.0
    for pass_position in PASSES:
        selected = [row[field] for row in rows if row["pass_position"] == pass_position]
        between += len(selected) * (float(np.mean(selected)) - grand_mean) ** 2
    total = float(np.sum((values - grand_mean) ** 2))
    return float(between / total) if total > 0 else 0.0


def group_summary(rows) -> dict[str, object]:
    fields = (
        "advantage",
        "progress",
        "goal_disagreement",
        "progress_residual",
        "disagreement_residual",
        "residual_critic_prediction",
        "combined_prediction",
    )
    return {
        "states": len(rows),
        **{
            field: {
                "mean": float(np.mean([row[field] for row in rows])),
                "median": float(np.median([row[field] for row in rows])),
                "std": float(np.std([row[field] for row in rows])),
            }
            for field in fields
        },
    }


def build_report(rows, train_rows, held_rows, pass_means) -> dict[str, object]:
    held_a = np.asarray([row["advantage"] for row in held_rows])
    held_a_resid = np.asarray([row["advantage_residual"] for row in held_rows])
    held_resid_prediction = np.asarray(
        [row["residual_critic_prediction"] for row in held_rows]
    )
    held_pass_prediction = np.asarray(
        [row["pass_only_prediction"] for row in held_rows]
    )
    held_combined_prediction = np.asarray(
        [row["combined_prediction"] for row in held_rows]
    )
    held_global_prediction = np.asarray([row["critic_prediction"] for row in held_rows])
    residual_zero_mse = float(np.mean(held_a_resid**2))
    residual_metrics = regression_metrics(held_a_resid, held_resid_prediction)
    residual_metrics["train_mean_baseline_mse"] = residual_zero_mse
    residual_metrics["mse_improvement_over_train_mean_fraction"] = (
        float(1.0 - residual_metrics["mse"] / residual_zero_mse)
        if residual_zero_mse > 0
        else 0.0
    )

    held_residual_associations = {
        "progress_residual_vs_advantage_residual": safe_association(
            [row["progress_residual"] for row in held_rows], held_a_resid
        ),
        "disagreement_residual_vs_advantage_residual": safe_association(
            [row["disagreement_residual"] for row in held_rows], held_a_resid
        ),
    }
    comparison = {
        "pass_only": regression_metrics(held_a, held_pass_prediction),
        "pass_plus_residual_critic": regression_metrics(
            held_a, held_combined_prediction
        ),
        "previous_global_two_signal_critic": regression_metrics(
            held_a, held_global_prediction
        ),
    }
    positive = held_a > 0
    train_majority_positive = bool(
        np.mean([row["advantage"] > 0 for row in train_rows]) >= 0.5
    )
    majority_score = np.full(len(held_rows), 1.0 if train_majority_positive else -1.0)
    classification = {
        "held_out_positive_states": int(positive.sum()),
        "held_out_negative_states": int((~positive).sum()),
        "trivial_majority": classification_metrics(positive, majority_score),
        "pass_only": classification_metrics(positive, held_pass_prediction),
        "pass_plus_residual_critic": classification_metrics(
            positive, held_combined_prediction
        ),
        "previous_global_two_signal_critic": classification_metrics(
            positive, held_global_prediction
        ),
    }
    classification_by_pass = {}
    for pass_position in PASSES:
        held_at_pass = [
            row for row in held_rows if row["pass_position"] == pass_position
        ]
        train_at_pass = [
            row for row in train_rows if row["pass_position"] == pass_position
        ]
        pass_target = np.asarray(
            [row["advantage"] > 0 for row in held_at_pass], dtype=bool
        )
        pass_specific_majority_positive = bool(
            np.mean([row["advantage"] > 0 for row in train_at_pass]) >= 0.5
        )
        classification_by_pass[str(pass_position)] = {
            "positive_states": int(pass_target.sum()),
            "negative_states": int((~pass_target).sum()),
            "pass_specific_majority": classification_metrics(
                pass_target,
                np.full(
                    len(held_at_pass),
                    1.0 if pass_specific_majority_positive else -1.0,
                ),
            ),
            "pass_plus_residual_critic": classification_metrics(
                pass_target, [row["combined_prediction"] for row in held_at_pass]
            ),
            "residual_score_ranking": classification_metrics(
                pass_target,
                [row["residual_critic_prediction"] for row in held_at_pass],
            ),
        }

    positions = np.asarray([row["pass_position"] for row in held_rows])
    clock = {
        "prediction_variance_explained_by_pass_fraction": variance_explained_by_pass(
            held_rows, "residual_critic_prediction"
        ),
        "prediction_vs_pass": safe_association(held_resid_prediction, positions),
        "mean_residual_prediction_by_pass": {
            str(pass_position): float(
                np.mean(
                    [
                        row["residual_critic_prediction"]
                        for row in held_rows
                        if row["pass_position"] == pass_position
                    ]
                )
            )
            for pass_position in PASSES
        },
    }
    positive_rows = [row for row in held_rows if row["advantage"] > 0]
    persist_rows = [row for row in held_rows if row["advantage"] <= 0]
    state_summaries_by_pass = {}
    for pass_position in PASSES:
        replan_at_pass = [
            row for row in positive_rows if row["pass_position"] == pass_position
        ]
        persist_at_pass = [
            row for row in persist_rows if row["pass_position"] == pass_position
        ]
        state_summaries_by_pass[str(pass_position)] = {
            "replan": group_summary(replan_at_pass) if replan_at_pass else None,
            "persist": group_summary(persist_at_pass) if persist_at_pass else None,
        }
    return {
        "invariants": {
            "episodes": len({row["episode_id"] for row in rows}),
            "decision_states": len(rows),
            "train_states": len(train_rows),
            "held_out_states": len(held_rows),
            "passes": list(PASSES),
            "critic_inputs": ["progress_residual", "disagreement_residual"],
            "pass_is_critic_input": False,
        },
        "within_pass_statistics_all_states": within_pass_statistics(rows),
        "training_pass_means_used_for_all_residualization": {
            str(key): value for key, value in pass_means.items()
        },
        "held_out_residual_associations": held_residual_associations,
        "held_out_residual_critic": residual_metrics,
        "held_out_advantage_prediction_comparison": comparison,
        "held_out_replan_classification": classification,
        "held_out_replan_classification_by_pass": classification_by_pass,
        "held_out_positive_replan_states": positive_rows,
        "held_out_persist_state_summary": group_summary(persist_rows),
        "held_out_replan_state_summary": group_summary(positive_rows),
        "held_out_state_summaries_by_pass": state_summaries_by_pass,
        "residual_clock_audit": clock,
    }


def markdown_report(report, provenance) -> str:
    within = report["within_pass_statistics_all_states"]
    residual_assoc = report["held_out_residual_associations"]
    residual = report["held_out_residual_critic"]
    comparison = report["held_out_advantage_prediction_comparison"]
    classification = report["held_out_replan_classification"]
    classification_by_pass = report["held_out_replan_classification_by_pass"]
    clock = report["residual_clock_audit"]
    positives = report["held_out_positive_replan_states"]
    persist = report["held_out_persist_state_summary"]
    pass_two_summary = report["held_out_state_summaries_by_pass"]["2"]

    within_lines = [
        "| Pass | N | REPLAN wins | Mean A | Median A | Progress Pearson/Spearman | Disagreement Pearson/Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pass_position, values in within.items():
        p = values["progress_vs_advantage"]
        d = values["goal_disagreement_vs_advantage"]
        within_lines.append(
            f"| {pass_position} | {values['states']} | "
            f"{values['replan_win_fraction']:.1%} | "
            f"{values['advantage_mean']:.6g} | "
            f"{values['advantage_median']:.6g} | "
            f"{p['pearson']:.3f} / {p['spearman']:.3f} | "
            f"{d['pearson']:.3f} / {d['spearman']:.3f} |"
        )

    compare_lines = [
        "| Predictor | MSE | R² | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in comparison.items():
        compare_lines.append(
            f"| {name.replace('_', ' ')} | {values['mse']:.6g} | "
            f"{values['r2']:.3f} | {values['pearson']:.3f} | "
            f"{values['spearman']:.3f} |"
        )

    class_lines = [
        "| Classifier | Accuracy | Balanced accuracy | REPLAN precision | REPLAN recall | AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in classification.items():
        if not isinstance(values, dict):
            continue
        auroc = "n/a" if values["auroc"] is None else f"{values['auroc']:.3f}"
        class_lines.append(
            f"| {name.replace('_', ' ')} | {values['accuracy']:.1%} | "
            f"{values['balanced_accuracy']:.1%} | "
            f"{values['replan_precision']:.1%} | "
            f"{values['replan_recall']:.1%} | {auroc} |"
        )

    pass_class_lines = [
        "| Pass | Pos/Neg | Pass-majority balanced accuracy | Residual-combined balanced accuracy | Residual-score AUROC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for pass_position, values in classification_by_pass.items():
        auroc = values["residual_score_ranking"]["auroc"]
        majority_balanced = values["pass_specific_majority"]["balanced_accuracy"]
        residual_balanced = values["pass_plus_residual_critic"]["balanced_accuracy"]
        pass_class_lines.append(
            f"| {pass_position} | {values['positive_states']}/"
            f"{values['negative_states']} | "
            f"{'n/a' if majority_balanced is None else f'{majority_balanced:.1%}'} | "
            f"{'n/a' if residual_balanced is None else f'{residual_balanced:.1%}'} | "
            f"{'n/a' if auroc is None else f'{auroc:.3f}'} |"
        )

    positive_lines = [
        "| Episode | Pass | A | Progress | Disagreement | Progress residual | Disagreement residual | Residual prediction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in positives:
        positive_lines.append(
            f"| {row['episode_id']} | {row['pass_position']} | "
            f"{row['advantage']:.6g} | {row['progress']:.5f} | "
            f"{row['goal_disagreement']:.5f} | "
            f"{row['progress_residual']:.5f} | "
            f"{row['disagreement_residual']:.5f} | "
            f"{row['residual_critic_prediction']:.6g} |"
        )

    pass_two_lines = [
        "| Oracle class | N | Mean A | Mean progress | Mean disagreement | Mean progress residual | Mean disagreement residual | Mean residual prediction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("replan", "persist"):
        values = pass_two_summary[label]
        pass_two_lines.append(
            f"| {label.upper()} | {values['states']} | "
            f"{values['advantage']['mean']:.6g} | "
            f"{values['progress']['mean']:.5f} | "
            f"{values['goal_disagreement']['mean']:.5f} | "
            f"{values['progress_residual']['mean']:.5f} | "
            f"{values['disagreement_residual']['mean']:.5f} | "
            f"{values['residual_critic_prediction']['mean']:.6g} |"
        )

    pass_mse = comparison["pass_only"]["mse"]
    combined_mse = comparison["pass_plus_residual_critic"]["mse"]
    beyond_pass_improvement = 1.0 - combined_mse / pass_mse
    useful = residual["r2"] > 0 and beyond_pass_improvement > 0
    rare_detection = (
        classification["pass_plus_residual_critic"]["balanced_accuracy"]
        > classification["trivial_majority"]["balanced_accuracy"]
        and classification["pass_plus_residual_critic"]["replan_recall"] > 0
    )
    strong_enough = useful and rare_detection and residual["r2"] >= 0.2

    return f"""# Counterfactual Persistence v2: Within-Pass Diagnostic

## Provenance

- Source states: `{provenance['input_csv']}`
- Source SHA-256: `{provenance['input_sha256']}`
- Frozen checkpoint represented by source: `{provenance['checkpoint']}`
- 64 episodes, 384 eligible states, fixed M=8, passes 2-7.
- Original episode-level train/held-out split retained (270/114 states).
- Pass means come from training states only. Critic inputs are exactly residual progress and residual disagreement; pass and dwell/time features are excluded.

## Within-pass statistics

{chr(10).join(within_lines)}

## Residual signal and critic (held out)

- Residual progress vs residual A: Pearson {residual_assoc['progress_residual_vs_advantage_residual']['pearson']:.3f}, Spearman {residual_assoc['progress_residual_vs_advantage_residual']['spearman']:.3f}.
- Residual disagreement vs residual A: Pearson {residual_assoc['disagreement_residual_vs_advantage_residual']['pearson']:.3f}, Spearman {residual_assoc['disagreement_residual_vs_advantage_residual']['spearman']:.3f}.
- Residual critic: MSE {residual['mse']:.6g}, R² {residual['r2']:.3f}, Pearson {residual['pearson']:.3f}, Spearman {residual['spearman']:.3f}.
- MSE improvement over the zero/train-mean residual baseline: {residual['mse_improvement_over_train_mean_fraction']:.1%}.

## Advantage prediction comparison (held out)

{chr(10).join(compare_lines)}

Adding the residual critic changes MSE versus pass-only by {beyond_pass_improvement:+.1%}.

## Rare REPLAN classification (held out)

Held-out positives/negatives: {classification['held_out_positive_states']}/{classification['held_out_negative_states']}.

{chr(10).join(class_lines)}

The pass-only AUROC above is a cross-pass ranking statistic: it assigns one constant score per pass and therefore has no within-pass ranking ability. Per-pass results are:

{chr(10).join(pass_class_lines)}

## Held-out REPLAN-positive states

{chr(10).join(positive_lines)}

Corresponding PERSIST-state summary (N={persist['states']}): mean progress {persist['progress']['mean']:.5f}, disagreement {persist['goal_disagreement']['mean']:.5f}, residual progress {persist['progress_residual']['mean']:.5f}, residual disagreement {persist['disagreement_residual']['mean']:.5f}, and residual prediction {persist['residual_critic_prediction']['mean']:.6g}.

Because all held-out positive states occur at pass 2, the matched pass-2 comparison is:

{chr(10).join(pass_two_lines)}

## Residual clock audit

- Prediction variance explained by pass: {clock['prediction_variance_explained_by_pass_fraction']:.1%}.
- Prediction vs pass: Pearson {clock['prediction_vs_pass']['pearson']:.3f}, Spearman {clock['prediction_vs_pass']['spearman']:.3f}.
- Mean residual prediction by pass: {json.dumps(clock['mean_residual_prediction_by_pass'], sort_keys=True)}.

## Scientific verdict

Within-pass residual signal is **{'useful' if useful else 'weak'}**. Rare-state discrimination is **{'better than trivial' if rare_detection else 'not better than trivial'}**. The remaining evidence is **{'strong enough to justify full v2 training' if strong_enough else 'not strong enough to justify full v2 training; another state feature should be evaluated first'}**. No training was launched.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--critic-epochs", type=int, default=1000)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    train_rows = [row for row in rows if row["split"] == "train"]
    held_rows = [row for row in rows if row["split"] == "held_out"]
    means = training_pass_means(train_rows)
    residualize(rows, means)
    critic = fit_residual_critic(train_rows, seed=args.seed, epochs=args.critic_epochs)
    apply_residual_critic(critic, rows)
    report = build_report(rows, train_rows, held_rows, means)
    provenance = {
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": sha256_file(args.input_csv),
        "checkpoint": str(
            ROOT / "experiments/meta_agents_adaptive_400step/"
            "20260820T101100Z_adaptive_eta0_seed0_400step/checkpoints/step_400"
        ),
        "seed": args.seed,
        "critic_epochs": args.critic_epochs,
        "critic_architecture": "Linear(2,4)-Tanh-Linear(4,1)",
    }
    report["provenance"] = provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = markdown_report(report, provenance)
    (args.output_dir / "REPORT.md").write_text(markdown)
    positive_fields = (
        "episode_id",
        "pass_position",
        "advantage",
        "progress",
        "goal_disagreement",
        "progress_residual",
        "disagreement_residual",
        "residual_critic_prediction",
        "combined_prediction",
    )
    with (args.output_dir / "held_out_replan_positive_states.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=positive_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in positive_fields}
            for row in report["held_out_positive_replan_states"]
        )
    print(markdown)


if __name__ == "__main__":
    main()
