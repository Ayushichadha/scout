#!/usr/bin/env python3
"""
Create feudal weight sweep plot showing "light helps, heavy hurts" pattern.

Purpose: Show that light feudal weight helps, but heavy weight hurts.

Features:
- x-axis: feudal_weight (0, 0.05, 0.1, 0.2...)
- y-axis: Eval LM loss or Accuracy
- Main line: Fixed manager_period (preferably P=3, fallback to P=4)
- Baseline: Dashed reference line for HRM without subgoal head
- λ=0: Special point labeled as "Subgoal head only (λ=0)"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import shared style
from plot_style import (
    apply_style,
    COLORS,
    FIGSIZE,
    LINE_WIDTH,
    LINE_WIDTH_REFERENCE,
    MARKER_SIZE,
    MARKER_SIZE_HIGHLIGHT,
    MARKER_EDGE_WIDTH,
    CAPSIZE,
    CAPTHICK,
    ERROR_LINE_WIDTH,
    FONT_ANNOTATION,
    save_figure,
    add_footnote,
    format_baseline_label,
    format_lambda_zero_label,
)

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

    # Apply consistent style
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot baseline as horizontal reference line
    if baseline_mean is not None:
        ax.axhline(
            y=baseline_mean,
            color=COLORS["baseline"],
            linestyle="--",
            linewidth=LINE_WIDTH_REFERENCE,
            label=format_baseline_label(),
            zorder=1,
        )

        # Add subtle error band for baseline if we have multiple runs
        if len(baseline_values) > 1 and baseline_std is not None:
            ax.fill_between(
                [min(weights) - 0.01, max(weights) + 0.01],
                baseline_mean - baseline_std,
                baseline_mean + baseline_std,
                color=COLORS["baseline_fill"],
                alpha=0.12,
                zorder=0,
            )

    # Compute statistics for each weight
    means = []
    stds = []
    valid_weights = []
    single_run_weights = []

    for weight in weights:
        values = main_period_data[weight][metric_name]
        if not values:
            continue

        mean, std, _ = compute_statistics(values)
        if mean is not None:
            means.append(mean)
            stds.append(std)
            valid_weights.append(weight)
            if len(values) == 1:
                single_run_weights.append(weight)

    if not valid_weights:
        print(f"⚠️  No valid data for period {main_period}. Skipping plot.")
        plt.close()
        return

    # Separate λ=0 from λ>0
    w0_idx = None
    if 0.0 in valid_weights:
        w0_idx = valid_weights.index(0.0)

    # Plot λ>0 curve
    w_positive = [w for w in valid_weights if w > 0]
    means_positive = [means[i] for i, w in enumerate(valid_weights) if w > 0]
    stds_positive = [stds[i] for i, w in enumerate(valid_weights) if w > 0]

    if w_positive:
        ax.errorbar(
            w_positive,
            means_positive,
            yerr=stds_positive if any(std > 0 for std in stds_positive) else None,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            label=f"Feudal HRM (P={main_period})",
            color=COLORS["primary"],
            elinewidth=ERROR_LINE_WIDTH,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
        )

    # Plot λ=0 separately with special label
    if w0_idx is not None:
        ax.plot(
            0.0,
            means[w0_idx],
            marker="s",  # Square marker to distinguish
            markersize=MARKER_SIZE,
            color=COLORS["lambda_zero"],
            markeredgecolor="black",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=4,
            label=format_lambda_zero_label(),
        )

    # Find and highlight optimal weight (best metric for λ>0)
    higher_is_better = metric_name == "accuracy"
    best_weight = None
    best_value = None

    for weight, mean in zip(valid_weights, means):
        if weight <= 0:  # Skip λ=0 for optimal
            continue
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
        # Plot optimal marker (star) - no legend entry
        ax.plot(
            best_weight,
            best_value,
            marker="*",
            markersize=MARKER_SIZE_HIGHLIGHT,
            color=COLORS["optimal"],
            markeredgecolor=COLORS["optimal_edge"],
            markeredgewidth=1.5,
            zorder=10,
            label="_nolegend_",  # Exclude from legend
        )

        # Add annotation - positioned to avoid overlap with star
        metric_label = "Accuracy" if metric_name == "accuracy" else "Loss"
        ax.annotate(
            f"Optimal: λ={best_weight}\n{metric_label}={best_value:.4f}",
            xy=(best_weight, best_value),
            xytext=(25, 35),  # Move further away from star marker
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["annotation_bg"],
                edgecolor=COLORS["annotation_edge"],
                linewidth=1.2,
                alpha=0.98,
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                color=COLORS["annotation_edge"],
                lw=1.2,
            ),
            fontsize=FONT_ANNOTATION,
            fontweight="bold",
            ha="left",
        )

    # Formatting
    ax.set_xlabel("Feudal Loss Weight (λ)")
    ax.set_ylabel(ylabel)

    title = f"Feudal Weight Sweep (Period={main_period})"
    ax.set_title(title)

    ax.set_xticks(valid_weights)

    # Legend in upper right corner
    ax.legend(loc="upper right", framealpha=0.95)

    # Add footnote for sample sizes and single-run info
    baseline_n = len(baseline_values) if baseline_values else 0
    footnote_parts = [f"Baseline: n={baseline_n}"]

    if single_run_weights:
        single_run_positive = [w for w in single_run_weights if w > 0]
        if single_run_positive:
            footnote_parts.append(f"Single-run λ values: {single_run_positive}")

    add_footnote(ax, " | ".join(footnote_parts))

    # Save plot
    output_file = output_dir / f"feudal_weight_sweep_{metric_name}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_file)

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
        metric_name="lm_loss",
        ylabel="Language Model Loss (lower is better)",
    )

    # Plot Accuracy (if available)
    plot_feudal_weight_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        primary_period=args.primary_period,
        metric_name="accuracy",
        ylabel="Accuracy (higher is better)",
    )

    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
