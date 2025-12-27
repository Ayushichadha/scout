#!/usr/bin/env python3
"""
Create feudal weight sweep plot showing "light helps, heavy hurts" pattern.

Purpose: Show that light feudal weight helps, but heavy weight hurts.

Features:
- x-axis: feudal_weight (0, 0.05, 0.1, 0.2...)
- y-axis: Eval LM loss
- Main line: Fixed manager_period (preferably P=3, fallback to P=4)
- Optional: Additional lines for other periods (P=1, P=6) to show interaction
- Baseline: weight=0 is baseline (must be included)
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
    sweep_files: List[Path],
    baseline_files: Optional[List[Path]] = None,
) -> Tuple[Dict, Dict]:
    """
    Load feudal weight sweep data grouped by manager_period.

    Returns:
        feudal_data: Dict mapping manager_period -> dict mapping weight -> list of metric values
        baseline_data: Dict with baseline metrics (weight=0)
    """
    feudal_data = {}  # {period: {weight: [values]}}
    baseline_metrics = {"lm_loss": [], "accuracy": []}

    # Load sweep data
    for sweep_file in sweep_files:
        with open(sweep_file, "r") as f:
            data = json.load(f)

        # Handle nested structure
        if isinstance(data, dict):
            if "manager_period_sweep" in data:
                results = data["manager_period_sweep"]
            elif "replications" in data:
                results = data["replications"]
            else:
                results = []
        elif isinstance(data, list):
            results = data
        else:
            continue

        for result in results:
            if not result.get("success", False):
                continue

            period = result.get("manager_period")
            weight = result.get("feudal_loss_weight", 0.0)

            # Skip if missing required fields
            if period is None:
                continue

            metrics = result.get("metrics", {})

            # Initialize period dict if needed
            if period not in feudal_data:
                feudal_data[period] = {}

            # Initialize weight dict if needed
            if weight not in feudal_data[period]:
                feudal_data[period][weight] = {
                    "lm_loss": [],
                    "accuracy": [],
                }

            # Add metrics
            if "final_lm_loss" in metrics:
                feudal_data[period][weight]["lm_loss"].append(metrics["final_lm_loss"])
            if "final_accuracy" in metrics:
                feudal_data[period][weight]["accuracy"].append(
                    metrics["final_accuracy"]
                )

    # Load baseline data (weight=0 or use_feudal=False)
    if baseline_files:
        for baseline_file in baseline_files:
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)

            if isinstance(baseline_data, list):
                for result in baseline_data:
                    if not result.get("success", False):
                        continue

                    # Only use true baseline (no feudal)
                    use_feudal = result.get("use_feudal", False)
                    weight = result.get("feudal_loss_weight", 0.0)

                    if use_feudal or weight > 0:
                        continue

                    metrics = result.get("metrics", {})
                    if "final_lm_loss" in metrics:
                        baseline_metrics["lm_loss"].append(metrics["final_lm_loss"])
                    if "final_accuracy" in metrics:
                        baseline_metrics["accuracy"].append(metrics["final_accuracy"])

    return feudal_data, baseline_metrics


def compute_statistics(values: List[float]) -> Tuple[float, float, Optional[float]]:
    """Compute mean, std, and 95% CI for a list of values."""
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


def plot_feudal_weight_sweep(
    feudal_data: Dict,
    baseline_data: Dict,
    output_dir: Path,
    primary_period: int = 3,
    additional_periods: Optional[List[int]] = None,
    metric_name: str = "lm_loss",
    ylabel: str = "Language Model Loss (lower is better)",
):
    """
    Create feudal weight sweep plot.

    Args:
        feudal_data: Dict mapping period -> dict mapping weight -> metric lists
        baseline_data: Dict with baseline metric lists
        output_dir: Directory to save plot
        primary_period: Main period to plot (default: 3)
        additional_periods: Optional additional periods to show
        metric_name: Name of metric to plot
        ylabel: Label for y-axis
    """
    # Find available periods
    available_periods = sorted([p for p in feudal_data.keys() if feudal_data[p]])

    if not available_periods:
        print("⚠️  No feudal data found. Skipping plot.")
        return

    # Use primary period if available, otherwise use first available
    if primary_period in available_periods:
        main_period = primary_period
    else:
        main_period = available_periods[0]
        print(
            f"⚠️  Period {primary_period} not found in data. Using period {main_period} instead."
        )
        print(
            f"   To plot period {primary_period}, run a feudal weight sweep at that period."
        )

    # Collect periods to plot
    periods_to_plot = [main_period]
    if additional_periods:
        for p in additional_periods:
            if p in available_periods and p != main_period:
                periods_to_plot.append(p)

    # Extract weights for main period
    main_period_data = feudal_data[main_period]
    weights = sorted(
        [w for w in main_period_data.keys() if main_period_data[w][metric_name]]
    )

    if not weights:
        print(f"⚠️  No {metric_name} data for period {main_period}. Skipping plot.")
        return

    # Compute baseline statistics
    baseline_values = baseline_data.get(metric_name, [])
    baseline_mean = None
    baseline_std = None

    if baseline_values:
        baseline_mean, baseline_std, _ = compute_statistics(baseline_values)

    # Create figure
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baseline as horizontal reference line
    if baseline_mean is not None:
        ax.axhline(
            y=baseline_mean,
            color="#6B7280",  # Gray-600
            linestyle="--",
            linewidth=2.5,
            label="Baseline (weight=0)",
            alpha=0.8,
            zorder=1,
        )

        # Add error band for baseline if we have multiple runs
        if len(baseline_values) > 1 and baseline_std is not None:
            ax.fill_between(
                [min(weights) - 0.01, max(weights) + 0.01],
                baseline_mean - baseline_std,
                baseline_mean + baseline_std,
                color="#6B7280",
                alpha=0.15,
                zorder=0,
            )

    # Color palette for different periods
    colors = {
        main_period: "#8B5CF6",  # Purple (main)
        1: "#EF4444",  # Red
        6: "#10B981",  # Green
        4: "#F59E0B",  # Amber (fallback)
    }

    # Plot curves for each period
    for period in periods_to_plot:
        if period not in feudal_data:
            continue

        period_data = feudal_data[period]
        period_weights = sorted(
            [w for w in period_data.keys() if period_data[w][metric_name]]
        )

        if not period_weights:
            print(
                f"⚠️  No {metric_name} data for period {period}. Skipping this period."
            )
            continue

        # Compute statistics for each weight
        means = []
        stds = []
        valid_weights = []

        for weight in period_weights:
            values = period_data[weight][metric_name]
            if not values:
                continue

            mean, std, _ = compute_statistics(values)
            if mean is not None:
                means.append(mean)
                stds.append(std)
                valid_weights.append(weight)

        if not valid_weights:
            continue

        # Choose color
        color = colors.get(period, "#6366F1")

        # Choose label
        if period == main_period:
            label = f"Feudal (Period={period})"
        else:
            label = f"Period={period}"

        # Plot curve
        ax.errorbar(
            valid_weights,
            means,
            yerr=stds if any(std > 0 for std in stds) else None,
            marker="o",
            markersize=10 if period == main_period else 8,
            linewidth=2.5 if period == main_period else 2,
            capsize=6,
            capthick=2,
            label=label,
            color=color,
            elinewidth=2,
            zorder=3 if period == main_period else 2,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    # Highlight optimal weight (best metric for main period)
    if main_period in feudal_data:
        main_data = feudal_data[main_period]
        best_weight = None
        best_value = None

        # Determine if higher is better (accuracy) or lower is better (loss)
        higher_is_better = metric_name == "accuracy"

        for weight in weights:
            values = main_data.get(weight, {}).get(metric_name, [])
            if values:
                mean = np.mean(values)
                if weight > 0:  # Exclude baseline
                    if best_value is None:
                        best_value = mean
                        best_weight = weight
                    elif higher_is_better and mean > best_value:
                        best_value = mean
                        best_weight = weight
                    elif not higher_is_better and mean < best_value:
                        best_value = mean
                        best_weight = weight

        if best_weight is not None:
            ax.plot(
                best_weight,
                best_value,
                marker="*",
                markersize=25,
                color="#FBBF24",  # Amber-400
                markeredgecolor="#92400E",  # Amber-800
                markeredgewidth=2,
                zorder=10,
                label="Optimal Weight",
            )

            # Add annotation
            metric_label = "Accuracy" if metric_name == "accuracy" else "Loss"
            ax.annotate(
                f"Best: w={best_weight}\n{metric_label}={best_value:.4f}",
                xy=(best_weight, best_value),
                xytext=(15, 25),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="#FEF3C7",
                    edgecolor="#92400E",
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

    # Formatting
    ax.set_xlabel("Feudal Loss Weight", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")

    title = "Feudal Weight Sweep"
    if baseline_mean is not None:
        title += f"\n(Baseline: {baseline_mean:.4f})"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    ax.set_xticks(weights)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"feudal_weight_sweep_{metric_name}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create feudal weight sweep plot")
    parser.add_argument(
        "--sweep-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSON files with feudal weight sweep results",
    )
    parser.add_argument(
        "--baseline-files",
        type=Path,
        nargs="+",
        default=None,
        help="JSON files with baseline (weight=0) results",
    )
    parser.add_argument(
        "--primary-period",
        type=int,
        default=3,
        help="Primary manager period to plot (default: 3)",
    )
    parser.add_argument(
        "--additional-periods",
        type=int,
        nargs="+",
        default=None,
        help="Additional periods to show (e.g., 1 6)",
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
        args.sweep_files, args.baseline_files
    )

    print(f"Found data for periods: {sorted(feudal_data.keys())}")
    print(f"Found baseline runs: {len(baseline_data.get('lm_loss', []))}")

    # Create plots
    print("\nGenerating plots...")

    # Plot Loss
    plot_feudal_weight_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        primary_period=args.primary_period,
        additional_periods=args.additional_periods,
        metric_name="lm_loss",
        ylabel="Language Model Loss (lower is better)",
    )

    # Plot Accuracy (if available)
    plot_feudal_weight_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        primary_period=args.primary_period,
        additional_periods=args.additional_periods,
        metric_name="accuracy",
        ylabel="Accuracy (higher is better)",
    )

    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
