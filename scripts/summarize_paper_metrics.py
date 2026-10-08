#!/usr/bin/env python3
"""Create compact paper metrics from replicated experiment JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    "experiments/baseline_extended_20251229_142140.json",
    "experiments/feudal_p3_w0.05_extended_20251229_143835.json",
    "experiments/feudal_p4_w0.05_extended_20251229_150204.json",
]

METRICS = ["final_lm_loss", "final_accuracy", "final_feudal_loss"]


def load_runs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = []
    for row in data.get("replications", []):
        if not row.get("success"):
            continue
        metrics = row.get("metrics", {})
        if not metrics:
            continue
        rows.append(
            {
                "source_file": path.name,
                "run_name": row.get("run_name", ""),
                "feudal_loss_weight": row.get("feudal_loss_weight"),
                "manager_period": row.get("manager_period"),
                **metrics,
            }
        )
    return rows


def config_name(row: dict[str, Any]) -> str:
    weight = row.get("feudal_loss_weight")
    period = row.get("manager_period")
    if weight == 0 or weight == 0.0:
        return "baseline_w0.0"
    return f"p{period}_w{weight}"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def stderr(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if values else float("nan")


def ci95(values: list[float]) -> float:
    # Normal approximation is fine for a compact internal paper table.
    return 1.96 * stderr(values)


def cohen_d(values: list[float], baseline: list[float]) -> float:
    if len(values) < 2 or len(baseline) < 2:
        return float("nan")
    pooled_var = (
        (len(values) - 1) * stdev(values) ** 2
        + (len(baseline) - 1) * stdev(baseline) ** 2
    ) / (len(values) + len(baseline) - 2)
    if pooled_var <= 0:
        return float("nan")
    return (mean(values) - mean(baseline)) / math.sqrt(pooled_var)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(config_name(row), []).append(row)

    baseline_rows = grouped.get("baseline_w0.0", [])
    summaries = []
    for name, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "config": name,
            "n": len(group),
            "feudal_loss_weight": group[0].get("feudal_loss_weight"),
            "manager_period": group[0].get("manager_period"),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if metric in row]
            baseline_values = [
                float(row[metric]) for row in baseline_rows if metric in row
            ]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
            summary[f"{metric}_ci95"] = ci95(values)
            summary[f"{metric}_delta_vs_baseline"] = (
                mean(values) - mean(baseline_values)
                if baseline_values
                else float("nan")
            )
            summary[f"{metric}_cohen_d_vs_baseline"] = cohen_d(values, baseline_values)
        summaries.append(summary)
    return summaries


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "config",
        "n",
        "final_lm_loss_mean",
        "final_lm_loss_std",
        "final_accuracy_mean",
        "final_accuracy_std",
        "final_feudal_loss_mean",
        "final_feudal_loss_std",
        "final_accuracy_delta_vs_baseline",
        "final_feudal_loss_delta_vs_baseline",
    ]
    labels = [
        "Config",
        "n",
        "LM Loss",
        "LM SD",
        "Acc",
        "Acc SD",
        "Intrinsic Align Loss",
        "Align SD",
        "Acc Delta",
        "Align Loss Delta",
    ]
    lines = [
        "# Replicated Paper Metrics",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    lines.append("")
    lines.append(
        "Notes: intrinsic alignment loss is the logged `final_feudal_loss`; lower is better. "
        "Deltas are relative to `baseline_w0.0`."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--csv", default="outputs/paper_replicated_metrics_summary.csv")
    parser.add_argument(
        "--markdown", default="outputs/paper_replicated_metrics_summary.md"
    )
    args = parser.parse_args()

    rows = []
    for raw_path in args.inputs:
        rows.extend(load_runs(Path(raw_path)))
    if not rows:
        raise SystemExit("No successful runs found.")

    summaries = summarize(rows)
    write_csv(summaries, Path(args.csv))
    write_markdown(summaries, Path(args.markdown))

    print(f"Loaded {len(rows)} successful replicated runs.")
    print(f"Saved CSV: {Path(args.csv).resolve()}")
    print(f"Saved Markdown: {Path(args.markdown).resolve()}")
    for row in summaries:
        print(
            f"{row['config']}: n={row['n']}, "
            f"loss={row['final_lm_loss_mean']:.4f}, "
            f"acc={row['final_accuracy_mean']:.4f}, "
            f"align_loss={row['final_feudal_loss_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
