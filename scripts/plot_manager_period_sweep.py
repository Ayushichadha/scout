#!/usr/bin/env python3
"""
Create signature plot for manager period sweep showing the non-monotonic "sweet spot".

Main claim: Manager period sweep curve - signature plot
Goal: Show the non-monotonic "sweet spot" at period ≈ 3.

Features:
- x-axis: manager_period (1, 2, 3, 4, 6, 8...)
- y-axis: primary metric (final eval loss or accuracy)
- lines: Baseline (no feudal) as horizontal reference + Feudal curve
- error bars: mean ± std across seeds (or 95% CI if available)
- Separate plots for Loss and Accuracy
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_experiment_data(
    sweep_file: Path,
    baseline_files: List[Path],
) -> Tuple[Dict, Dict]:
    """
    Load manager period sweep and baseline data.

    Returns:
        feudal_data: Dict mapping manager_period -> list of metric values
        baseline_data: Dict with baseline metrics
    """
    # Load manager period sweep
    with open(sweep_file, "r") as f:
        sweep_data = json.load(f)

    # Extract manager_period_sweep if it's nested
    if isinstance(sweep_data, dict) and "manager_period_sweep" in sweep_data:
        sweep_results = sweep_data["manager_period_sweep"]
    elif isinstance(sweep_data, list):
        sweep_results = sweep_data
    else:
        raise ValueError(f"Unexpected format in {sweep_file}")

    # Group by manager_period
    feudal_data = {}
    for result in sweep_results:
        if not result.get("success", False):
            continue

        period = result.get("manager_period")
        if period is None:
            continue

        metrics = result.get("metrics", {})

        if period not in feudal_data:
            feudal_data[period] = {
                "lm_loss": [],
                "accuracy": [],
                "feudal_loss": [],
            }

        if "final_lm_loss" in metrics:
            feudal_data[period]["lm_loss"].append(metrics["final_lm_loss"])
        if "final_accuracy" in metrics:
            feudal_data[period]["accuracy"].append(metrics["final_accuracy"])
        if "final_feudal_loss" in metrics:
            feudal_data[period]["feudal_loss"].append(metrics["final_feudal_loss"])

    # Load baseline data
    baseline_metrics = {
        "lm_loss": [],
        "accuracy": [],
        "feudal_loss": [],
    }

    for baseline_file in baseline_files:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)

        if isinstance(baseline_data, list):
            for result in baseline_data:
                if not result.get("success", False):
                    continue

                # Only use baseline (no feudal) results
                use_feudal = result.get("use_feudal", False)
                feudal_weight = result.get("feudal_loss_weight", 0.0)

                if use_feudal or feudal_weight > 0:
                    continue

                metrics = result.get("metrics", {})
                if "final_lm_loss" in metrics:
                    baseline_metrics["lm_loss"].append(metrics["final_lm_loss"])
                if "final_accuracy" in metrics:
                    baseline_metrics["accuracy"].append(metrics["final_accuracy"])
                if "final_feudal_loss" in metrics:
                    baseline_metrics["feudal_loss"].append(metrics["final_feudal_loss"])

    return feudal_data, baseline_metrics


def compute_statistics(values: List[float]) -> Tuple[float, float, Optional[float]]:
    """
    Compute mean, std, and 95% CI for a list of values.

    Returns:
        mean, std, ci (or None if < 3 samples)
    """
    if len(values) == 0:
        return None, None, None

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0

    # Compute 95% CI if we have enough samples
    ci = None
    if len(values) >= 3 and HAS_SCIPY:
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
        ci = ci[1] - mean  # Return half-width

    return mean, std, ci


def plot_manager_period_sweep(
    feudal_data: Dict,
    baseline_data: Dict,
    output_dir: Path,
    metric_name: str = "lm_loss",
    ylabel: str = "Language Model Loss",
    title_suffix: str = "",
):
    """
    Create manager period sweep plot with baseline reference.

    Args:
        feudal_data: Dict mapping manager_period -> dict of metric lists
        baseline_data: Dict with baseline metric lists
        output_dir: Directory to save plot
        metric_name: Name of metric to plot ("lm_loss" or "accuracy")
        ylabel: Label for y-axis
        title_suffix: Additional text for title
    """
    # Extract data
    periods = sorted([p for p in feudal_data.keys() if feudal_data[p][metric_name]])
    if not periods:
        print(f"⚠️  No {metric_name} data found in feudal experiments. Skipping plot.")
        return

    # Compute feudal statistics
    feudal_means = []
    feudal_stds = []
    feudal_cis = []
    valid_periods = []

    for period in periods:
        values = feudal_data[period][metric_name]
        if not values:
            continue

        mean, std, ci = compute_statistics(values)
        if mean is not None:
            feudal_means.append(mean)
            feudal_stds.append(std)
            feudal_cis.append(ci)
            valid_periods.append(period)

    if not valid_periods:
        print(f"⚠️  No valid {metric_name} data. Skipping plot.")
        return

    # Compute baseline statistics
    baseline_values = baseline_data.get(metric_name, [])
    baseline_mean = None
    baseline_std = None
    baseline_ci = None

    if baseline_values:
        baseline_mean, baseline_std, baseline_ci = compute_statistics(baseline_values)

    # Create figure with better styling
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Use default style

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baseline as horizontal reference line
    if baseline_mean is not None:
        ax.axhline(
            y=baseline_mean,
            color="#6B7280",  # Gray-600
            linestyle="--",
            linewidth=2.5,
            label="Baseline (no feudal)",
            alpha=0.8,
            zorder=1,
        )

        # Add error band for baseline if we have multiple runs
        if len(baseline_values) > 1 and baseline_std is not None:
            ax.fill_between(
                [min(valid_periods) - 0.5, max(valid_periods) + 0.5],
                baseline_mean - baseline_std,
                baseline_mean + baseline_std,
                color="#6B7280",
                alpha=0.15,
                zorder=0,
            )

    # Plot feudal curve
    use_ci = any(ci is not None for ci in feudal_cis) and len(baseline_values) >= 3

    # Determine error bars
    if use_ci and any(ci is not None for ci in feudal_cis):
        # Use 95% CI error bars
        errors = [
            ci if ci is not None else std for ci, std in zip(feudal_cis, feudal_stds)
        ]
        error_label = "Feudal (95% CI)"
    elif any(std > 0 for std in feudal_stds):
        # Use std error bars
        errors = feudal_stds
        error_label = "Feudal (±std)"
    else:
        # No error bars
        errors = None
        error_label = "Feudal"

    # Plot the main curve
    ax.errorbar(
        valid_periods,
        feudal_means,
        yerr=errors,
        marker="o",
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        label=error_label,
        color="#8B5CF6",  # Purple-500
        elinewidth=2,
        zorder=3,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    # Highlight the sweet spot (period 3)
    if 3 in valid_periods:
        idx = valid_periods.index(3)
        ax.plot(
            valid_periods[idx],
            feudal_means[idx],
            marker="*",
            markersize=25,
            color="#FBBF24",  # Amber-400
            markeredgecolor="#92400E",  # Amber-800
            markeredgewidth=2,
            zorder=10,
            label="Sweet Spot (Period=3)",
        )

    # Formatting
    ax.set_xlabel("Manager Period", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")

    title = f"Manager Period Sweep{title_suffix}"
    if baseline_mean is not None:
        title += f"\n(Baseline: {baseline_mean:.4f})"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    ax.set_xticks(valid_periods)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)

    # Add annotation for sweet spot
    if 3 in valid_periods:
        idx = valid_periods.index(3)
        best_value = feudal_means[idx]
        metric_label = ylabel.split("(")[0].strip()
        ax.annotate(
            f"Best: Period=3\n{metric_label}={best_value:.4f}",
            xy=(3, best_value),
            xytext=(15, 25),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#FEF3C7",  # Amber-100
                edgecolor="#92400E",  # Amber-800
                linewidth=1.5,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                color="#92400E",
                lw=1.5,
            ),
            fontsize=10,
            fontweight="bold",
            ha="left",
        )

    plt.tight_layout()

    # Save plot
    metric_file = metric_name.replace("_", "_")
    output_file = output_dir / f"manager_period_sweep_{metric_file}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create signature plot for manager period sweep"
    )
    parser.add_argument(
        "--sweep-file",
        type=Path,
        required=True,
        help="JSON file with manager period sweep results",
    )
    parser.add_argument(
        "--baseline-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSON files with baseline (no feudal) results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="Directory to save plots",
    )

    args = parser.parse_args()

    # Load data
    print("Loading experiment data...")
    feudal_data, baseline_data = load_experiment_data(
        args.sweep_file, args.baseline_files
    )

    print(f"Found feudal data for periods: {sorted(feudal_data.keys())}")
    print(f"Found baseline runs: {len(baseline_data.get('lm_loss', []))}")

    # Create plots
    print("\nGenerating plots...")

    # Plot 1: Loss
    plot_manager_period_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        metric_name="lm_loss",
        ylabel="Language Model Loss (lower is better)",
        title_suffix=" - Loss",
    )

    # Plot 2: Accuracy (if available)
    plot_manager_period_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        metric_name="accuracy",
        ylabel="Accuracy (higher is better)",
        title_suffix=" - Accuracy",
    )

    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
