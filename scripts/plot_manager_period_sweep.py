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
)

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
    single_run_periods = []  # Track periods with only 1 run

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
            if len(values) == 1:
                single_run_periods.append(period)

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

        # Removed heavy background shading - just use dashed line for baseline
        # If needed, use very light shading (alpha=0.05) but currently removed

    # Determine error bars
    use_ci = any(ci is not None for ci in feudal_cis) and len(baseline_values) >= 3

    if use_ci and any(ci is not None for ci in feudal_cis):
        errors = [
            ci if ci is not None else std for ci, std in zip(feudal_cis, feudal_stds)
        ]
        curve_label = "Feudal HRM (95% CI)"
    elif any(std > 0 for std in feudal_stds):
        errors = feudal_stds
        curve_label = "Feudal HRM (±std)"
    else:
        errors = None
        curve_label = "Feudal HRM"

    # Plot the main curve
    ax.errorbar(
        valid_periods,
        feudal_means,
        yerr=errors,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        capsize=CAPSIZE,
        capthick=CAPTHICK,
        label=curve_label,
        color=COLORS["primary"],
        elinewidth=ERROR_LINE_WIDTH,
        zorder=3,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
    )

    # Find and highlight the optimal point (best metric)
    higher_is_better = metric_name == "accuracy"
    if higher_is_better:
        best_idx = np.argmax(feudal_means)
    else:
        best_idx = np.argmin(feudal_means)

    best_period = valid_periods[best_idx]
    best_value = feudal_means[best_idx]

    # Plot optimal marker (star) - no legend entry
    ax.plot(
        best_period,
        best_value,
        marker="*",
        markersize=MARKER_SIZE_HIGHLIGHT,
        color=COLORS["optimal"],
        markeredgecolor=COLORS["optimal_edge"],
        markeredgewidth=1.5,
        zorder=10,
        label="_nolegend_",  # Exclude from legend
    )

    # Add annotation for optimal point - positioned to avoid overlap with star
    metric_label = "Accuracy" if metric_name == "accuracy" else "Loss"
    ax.annotate(
        f"Optimal: P={best_period}\n{metric_label}={best_value:.4f}",
        xy=(best_period, best_value),
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
    ax.set_xlabel("Manager Period")
    ax.set_ylabel(ylabel)

    title = f"Manager Period Sweep{title_suffix}"
    ax.set_title(title)

    ax.set_xticks(valid_periods)

    # Legend in upper right corner (out of the way)
    ax.legend(loc="upper right", framealpha=0.95)

    # Add footnote for sample sizes and single-run info
    baseline_n = len(baseline_values) if baseline_values else 0
    footnote_parts = [f"Baseline: n={baseline_n}"]

    if single_run_periods:
        footnote_parts.append(f"Single-run periods: {single_run_periods}")

    add_footnote(ax, " | ".join(footnote_parts))

    # Save plot
    metric_file = metric_name.replace("_", "_")
    output_file = output_dir / f"manager_period_sweep_{metric_file}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_file)

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

    # Create plots - ONLY lm_loss (main metric)
    print("\nGenerating plot...")

    plot_manager_period_sweep(
        feudal_data,
        baseline_data,
        args.output_dir,
        metric_name="lm_loss",
        ylabel="Language Model Loss (lower is better)",
        title_suffix="",
    )

    print("\n✅ Plot generated!")


if __name__ == "__main__":
    main()
