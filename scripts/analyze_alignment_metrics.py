#!/usr/bin/env python3
"""Summarize worker-goal alignment metrics from hidden-state dumps.

The input files are produced by scripts/dump_hidden_states.py and contain
worker_hidden, subgoal_goal, and solved tensors. This script is read-only with
respect to dumps and writes compact CSV/Markdown summaries for paper tables.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


STAGE_ORDER = {"early": 0, "mid": 1, "late": 2}


def mean_std(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return mean, std


def point_biserial(values: torch.Tensor, labels: torch.Tensor) -> float:
    """Correlation between continuous values and bool labels."""
    x = values.to(torch.float64)
    y = labels.to(torch.float64)
    if x.numel() < 2 or y.unique().numel() < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum()))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum().div(denom).item())


def summarize_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    worker = payload["worker_hidden"].to(torch.float32)
    goal = payload["subgoal_goal"].to(torch.float32)
    solved = payload["solved"].bool()

    if worker.dim() == 3:
        worker = worker.mean(dim=1)
    if goal.dim() == 3:
        goal = goal.mean(dim=1)

    alignment = (
        F.normalize(worker, dim=-1, eps=1e-8) * F.normalize(goal, dim=-1, eps=1e-8)
    ).sum(dim=-1)
    feudal_distance = 1.0 - alignment
    worker_norm = worker.norm(dim=-1)
    goal_norm = goal.norm(dim=-1)

    solved_alignment = alignment[solved]
    unsolved_alignment = alignment[~solved]
    align_mean, align_std = mean_std(alignment)
    solved_mean, solved_std = mean_std(solved_alignment)
    unsolved_mean, unsolved_std = mean_std(unsolved_alignment)
    worker_norm_mean, worker_norm_std = mean_std(worker_norm)
    goal_norm_mean, goal_norm_std = mean_std(goal_norm)

    cell = path.parent.name
    stage = path.stem
    n = int(alignment.numel())
    n_solved = int(solved.sum().item())
    n_unsolved = n - n_solved

    return {
        "cell": cell,
        "stage": stage,
        "dump_path": str(path),
        "n": n,
        "n_solved": n_solved,
        "solve_rate": n_solved / n if n else float("nan"),
        "alignment_mean": align_mean,
        "alignment_std": align_std,
        "alignment_solved_mean": solved_mean,
        "alignment_solved_std": solved_std,
        "alignment_unsolved_mean": unsolved_mean,
        "alignment_unsolved_std": unsolved_std,
        "alignment_solved_minus_unsolved": (
            solved_mean - unsolved_mean if n_solved and n_unsolved else float("nan")
        ),
        "alignment_solved_correlation": point_biserial(alignment, solved),
        "feudal_distance_mean": float(feudal_distance.mean().item()),
        "worker_norm_mean": worker_norm_mean,
        "worker_norm_std": worker_norm_std,
        "goal_norm_mean": goal_norm_mean,
        "goal_norm_std": goal_norm_std,
    }


def sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["cell"]), STAGE_ORDER.get(str(row["stage"]), 99), str(row["stage"]))


def format_float(value: Any) -> str:
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
        "cell",
        "stage",
        "solve_rate",
        "alignment_mean",
        "alignment_solved_mean",
        "alignment_unsolved_mean",
        "alignment_solved_minus_unsolved",
        "alignment_solved_correlation",
    ]
    labels = {
        "cell": "Cell",
        "stage": "Stage",
        "solve_rate": "Solve Rate",
        "alignment_mean": "Align Mean",
        "alignment_solved_mean": "Solved Align",
        "alignment_unsolved_mean": "Unsolved Align",
        "alignment_solved_minus_unsolved": "Delta",
        "alignment_solved_correlation": "Corr",
    }
    lines = [
        "# Worker-Goal Alignment Summary",
        "",
        "| " + " | ".join(labels[col] for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(format_float(row[col]) for col in columns) + " |"
        )
    lines.append("")
    lines.append(
        "Notes: alignment is cosine(worker_hidden, subgoal_goal). Delta is solved minus unsolved alignment."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dumps-dir",
        default="dumps",
        help="Directory containing cell/stage .pt dumps.",
    )
    parser.add_argument(
        "--csv",
        default="outputs/alignment_metrics_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--markdown",
        default="outputs/alignment_metrics_summary.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    dumps_dir = Path(args.dumps_dir)
    dump_paths = sorted(dumps_dir.glob("*/*.pt"))
    if not dump_paths:
        raise SystemExit(f"No .pt dump files found under {dumps_dir}")

    rows = sorted((summarize_dump(path) for path in dump_paths), key=sort_key)
    write_csv(rows, Path(args.csv))
    write_markdown(rows, Path(args.markdown))

    print(f"Analyzed {len(rows)} dump files from {dumps_dir}")
    print(f"Saved CSV: {Path(args.csv).resolve()}")
    print(f"Saved Markdown: {Path(args.markdown).resolve()}")
    for row in rows:
        print(
            f"{row['cell']} {row['stage']}: solve={row['solve_rate']:.3f}, "
            f"align={row['alignment_mean']:.3f}, "
            f"delta={format_float(row['alignment_solved_minus_unsolved'])}"
        )


if __name__ == "__main__":
    main()
