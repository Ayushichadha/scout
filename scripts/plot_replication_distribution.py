#!/usr/bin/env python3
"""
Create replication distribution plot: baseline vs best-feudal.

Purpose: Show that results are "not noise" - demonstrate statistical significance.

Features:
- Box/violin + swarm plot
- Groups: Baseline vs Best-Feudal (P=3, λ=0.05)
- Clean statistics annotation (n, mean±std)
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import shared style
from plot_style import (
    apply_style,
    COLORS,
    MARKER_EDGE_WIDTH,
    FONT_TICK,
    FONT_ANNOTATION,
    save_figure,
    add_footnote,
)

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_replication_data(
    baseline_files: List[Path],
    feudal_replication_file: Path,
    feudal_period: int = 3,
    feudal_weight: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """
    Load replication data for baseline and best-feudal config.

    Returns:
        baseline_values: List of loss values for baseline
        feudal_values: List of loss values for best-feudal
    """
    baseline_values = []
    feudal_values = []

    # Load baseline data
    for baseline_file in baseline_files:
        with open(baseline_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            for result in data:
                if not result.get("success", False):
                    continue

                # Only use true baseline (no feudal)
                use_feudal = result.get("use_feudal", False)
                weight = result.get("feudal_loss_weight", 0.0)

                if use_feudal or weight > 0:
                    continue

                metrics = result.get("metrics", {})
                if "final_lm_loss" in metrics:
                    baseline_values.append(metrics["final_lm_loss"])

    # Load feudal replication data
    with open(feudal_replication_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "replications" in data:
        replications = data["replications"]
    elif isinstance(data, list):
        replications = data
    else:
        replications = []

    for result in replications:
        if not result.get("success", False):
            continue

        period = result.get("manager_period")
        weight = result.get("feudal_loss_weight", 0.0)

        # Only use the best-feudal config
        if period == feudal_period and weight == feudal_weight:
            metrics = result.get("metrics", {})
            if "final_lm_loss" in metrics:
                feudal_values.append(metrics["final_lm_loss"])

    return baseline_values, feudal_values


def plot_replication_distribution(
    baseline_values: List[float],
    feudal_values: List[float],
    output_dir: Path,
    metric_name: str = "lm_loss",
    ylabel: str = "Language Model Loss (lower is better)",
    title: str = "Replication Distribution: Baseline vs Best-Feudal",
    feudal_period: int = 3,
    feudal_weight: float = 0.05,
):
    """
    Create replication distribution plot with box/violin + swarm.

    Args:
        baseline_values: List of baseline metric values
        feudal_values: List of best-feudal metric values
        output_dir: Directory to save plot
        metric_name: Name of metric
        ylabel: Y-axis label
        title: Plot title
    """
    if not baseline_values or not feudal_values:
        print(
            f"⚠️  Missing data. Baseline: {len(baseline_values)}, Feudal: {len(feudal_values)}"
        )
        return

    # Apply consistent style
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute statistics first
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values, ddof=1) if len(baseline_values) > 1 else 0
    feudal_mean = np.mean(feudal_values)
    feudal_std = np.std(feudal_values, ddof=1) if len(feudal_values) > 1 else 0

    # Remove violin/box plots - use jittered dots + mean±std instead
    # Violin is misleading with n=2 baseline

    # Add jittered scatter points
    np.random.seed(42)  # For reproducibility
    x_baseline = np.random.normal(0, 0.08, len(baseline_values))
    x_feudal = np.random.normal(1, 0.08, len(feudal_values))

    ax.scatter(
        x_baseline,
        baseline_values,
        color=COLORS["scatter_baseline"],
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=MARKER_EDGE_WIDTH,
        zorder=10,
    )
    ax.scatter(
        x_feudal,
        feudal_values,
        color=COLORS["scatter_feudal"],
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=MARKER_EDGE_WIDTH,
        zorder=10,
    )

    # Add mean ± std indicators with error bars
    # Baseline
    ax.errorbar(
        [0],
        [baseline_mean],
        yerr=[[baseline_std], [baseline_std]] if baseline_std > 0 else None,
        fmt="D",
        color=COLORS["scatter_baseline"],
        markersize=10,
        capsize=8,
        capthick=2,
        elinewidth=2,
        zorder=5,
        label="_nolegend_",
    )
    # Feudal
    ax.errorbar(
        [1],
        [feudal_mean],
        yerr=[[feudal_std], [feudal_std]] if feudal_std > 0 else None,
        fmt="D",
        color=COLORS["scatter_feudal"],
        markersize=10,
        capsize=8,
        capthick=2,
        elinewidth=2,
        zorder=5,
        label="_nolegend_",
    )

    # Set labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "HRM Baseline\n(no subgoal head)",
            f"Best-Feudal\n(P={feudal_period}, λ={feudal_weight})",
        ],
        fontsize=FONT_TICK,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Compute p-value if scipy is available
    p_value_text = ""
    if HAS_SCIPY and len(baseline_values) > 1 and len(feudal_values) > 1:
        t_stat, p_value = stats.ttest_ind(baseline_values, feudal_values)
        sig_stars = ""
        if p_value < 0.001:
            sig_stars = "***"
        elif p_value < 0.01:
            sig_stars = "**"
        elif p_value < 0.05:
            sig_stars = "*"
        p_value_text = f"  (p={p_value:.3f}{sig_stars})"

    # Add compact statistics annotation in upper left
    stats_text = (
        f"Baseline: n={len(baseline_values)}, {baseline_mean:.4f}±{baseline_std:.4f}\n"
        f"Feudal: n={len(feudal_values)}, {feudal_mean:.4f}±{feudal_std:.4f}"
        f"{p_value_text}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=FONT_ANNOTATION,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#E5E7EB",
            alpha=0.95,
        ),
    )

    # Add footnote if baseline has limited samples
    footnote = ""
    if len(baseline_values) < 3:
        footnote = "Note: Baseline has limited replications; results are indicative."

    if footnote:
        add_footnote(ax, footnote)

    # Save plot
    output_file = output_dir / f"replication_distribution_{metric_name}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_file)

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create replication distribution plot: baseline vs best-feudal"
    )
    parser.add_argument(
        "--baseline-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSON files with baseline (no feudal) results",
    )
    parser.add_argument(
        "--feudal-replication-file",
        type=Path,
        required=True,
        help="JSON file with feudal replication results",
    )
    parser.add_argument(
        "--feudal-period",
        type=int,
        default=3,
        help="Manager period for best-feudal config (default: 3)",
    )
    parser.add_argument(
        "--feudal-weight",
        type=float,
        default=0.05,
        help="Feudal weight for best-feudal config (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="Directory to save plots",
    )

    args = parser.parse_args()

    # Load data
    print("Loading replication data...")
    baseline_values, feudal_values = load_replication_data(
        args.baseline_files,
        args.feudal_replication_file,
        args.feudal_period,
        args.feudal_weight,
    )

    print(f"Found {len(baseline_values)} baseline runs")
    print(
        f"Found {len(feudal_values)} feudal runs (P={args.feudal_period}, λ={args.feudal_weight})"
    )

    # Create plot
    print("\nGenerating plot...")
    plot_replication_distribution(
        baseline_values,
        feudal_values,
        args.output_dir,
        metric_name="lm_loss",
        ylabel="Language Model Loss (lower is better)",
        title="Replication Distribution: Baseline vs Best-Feudal",
        feudal_period=args.feudal_period,
        feudal_weight=args.feudal_weight,
    )

    print("\n✅ Plot generated!")


if __name__ == "__main__":
    main()
