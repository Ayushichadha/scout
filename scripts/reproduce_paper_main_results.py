#!/usr/bin/env python3
"""Recompute paper analysis from published records, without training or checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_schedule_sweep_k2 import analyze, read_matrix  # noqa: E402

RESULTS = ROOT / "papers/beyond-the-clock/results"
SWEEP = ROOT / "experiments/schedule_sweep_k2"


def verify_summary(actual, expected, path="summary"):
    """Reject changed reported values, allowing numerical roundoff only."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            verify_summary(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: length mismatch")
        for index, value in enumerate(expected):
            verify_summary(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, (int, float)):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{path}: recomputed {actual}, archived {expected}")
    elif actual != expected:
        raise ValueError(f"{path}: value mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/paper-main-results"
    )
    args = parser.parse_args()

    for filename in ("final_comparison.csv", "schedule_comparison.csv"):
        print(f"\n{filename} (archived evaluation aggregates):")
        with (RESULTS / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                print(
                    f"  {row['condition']}: macro accuracy={float(row['token_accuracy']):.9f}"
                )

    with (RESULTS / "decision_level_metrics.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    beta = np.asarray([float(row["beta"]) for row in decisions])
    passes = np.asarray([int(row["eligible_pass"]) for row in decisions])
    means = np.asarray([beta[passes == index].mean() for index in passes])
    explained = float(100 * np.mean((means - beta.mean()) ** 2) / np.var(beta))
    statistics = json.loads((RESULTS / "feature_beta_statistics.json").read_text())
    verify_summary(
        explained, statistics["variance_decomposition"]["percent_explained_by_position"]
    )
    print(f"\nRecomputed variance explained by pass position: {explained:.6f}%")

    print(
        "Recomputing six-clock aggregates and 10,000 paired bootstrap replicates...",
        flush=True,
    )
    report, summary = analyze(read_matrix(SWEEP / "per_episode_matrix.csv"))
    summary = json.loads(json.dumps(summary))
    verify_summary(summary, json.loads((SWEEP / "analysis_summary.json").read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ceiling_analysis.md").write_text(report)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Verified best clock [1,{summary['best_fixed_k']}], micro={summary['best_fixed_micro']:.9f}"
    )
    print(
        f"Verified oracle headroom={summary['headroom_micro']:.9f}, CI={summary['headroom_ci95']}"
    )
    print(f"Analysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
