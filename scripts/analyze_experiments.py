#!/usr/bin/env python3
"""Analyze and compare training experiment results."""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_experiment_results(result_files: List[Path]) -> pd.DataFrame:
    """Load experiment results from JSON files."""
    all_results = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)

    return pd.DataFrame(all_results)


def analyze_hyperparameter_sweep(df: pd.DataFrame) -> Dict:
    """Analyze hyperparameter sweep results."""
    analysis = {}

    # Group by hyperparameters
    if "feudal_loss_weight" in df.columns and "manager_period" in df.columns:
        # Grid search results
        pivot = df.pivot_table(
            values="success",
            index="feudal_loss_weight",
            columns="manager_period",
            aggfunc="sum",
        )
        analysis["grid_summary"] = pivot.to_dict()

        # Best hyperparameters
        best = df[df["success"]]
        if len(best) > 0:
            analysis["best_configs"] = best.to_dict("records")

    elif "feudal_loss_weight" in df.columns:
        # Feudal weight sweep
        analysis["feudal_weight_analysis"] = (
            df.groupby("feudal_loss_weight")["success"].sum().to_dict()
        )
        best_weight = df[df["success"]]["feudal_loss_weight"].values
        if len(best_weight) > 0:
            analysis["best_feudal_weight"] = float(best_weight[0])

    elif "manager_period" in df.columns:
        # Manager period sweep
        analysis["manager_period_analysis"] = (
            df.groupby("manager_period")["success"].sum().to_dict()
        )
        best_period = df[df["success"]]["manager_period"].values
        if len(best_period) > 0:
            analysis["best_manager_period"] = int(best_period[0])

    return analysis


def compare_baseline_vs_feudal(df: pd.DataFrame) -> Dict:
    """Compare baseline vs feudal loss experiments."""
    comparison = {}

    baseline = df[~df.get("use_feudal", False)]
    feudal = df[df.get("use_feudal", False)]

    comparison["baseline_success"] = (
        baseline["success"].sum() if len(baseline) > 0 else 0
    )
    comparison["feudal_success"] = feudal["success"].sum() if len(feudal) > 0 else 0

    comparison["baseline_total"] = len(baseline)
    comparison["feudal_total"] = len(feudal)

    if len(baseline) > 0 and len(feudal) > 0:
        comparison["improvement"] = (
            comparison["feudal_success"] / comparison["feudal_total"]
            - comparison["baseline_success"] / comparison["baseline_total"]
        )

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze training experiment results")
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="JSON files containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for analysis",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "table"],
        default="table",
        help="Output format",
    )

    args = parser.parse_args()

    # Load results
    df = load_experiment_results(args.result_files)

    # Analyze
    analysis = {}
    if "feudal_loss_weight" in df.columns or "manager_period" in df.columns:
        analysis["hyperparameter_sweep"] = analyze_hyperparameter_sweep(df)

    if "use_feudal" in df.columns:
        analysis["baseline_comparison"] = compare_baseline_vs_feudal(df)

    # Output
    if args.format == "json":
        output = json.dumps(analysis, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(output)
        else:
            print(output)
    else:
        # Table format
        print("\n" + "=" * 60)
        print("Experiment Results Summary")
        print("=" * 60)
        print(f"\nTotal experiments: {len(df)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")

        if "feudal_loss_weight" in df.columns:
            print("\nBy Feudal Loss Weight:")
            print(df.groupby("feudal_loss_weight")["success"].sum())

        if "manager_period" in df.columns:
            print("\nBy Manager Period:")
            print(df.groupby("manager_period")["success"].sum())

        if "use_feudal" in df.columns:
            print("\nBaseline vs Feudal:")
            baseline = df[~df["use_feudal"]]
            feudal = df[df["use_feudal"]]
            print(f"  Baseline: {baseline['success'].sum()}/{len(baseline)}")
            print(f"  Feudal: {feudal['success'].sum()}/{len(feudal)}")


if __name__ == "__main__":
    main()
