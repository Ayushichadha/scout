#!/usr/bin/env python3
"""
Create replication distribution plot: baseline vs best-feudal.

Purpose: Show that results are "not noise" - demonstrate statistical significance.

Features:
- Box/violin + swarm plot (best option)
- Or two-column beeswarm plot
- Groups: Baseline vs (P=3, λ=0.05)
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

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

    # Prepare data for plotting
    data = []
    labels = []

    # Baseline data
    for val in baseline_values:
        data.append(val)
        labels.append("Baseline")

    # Feudal data
    for val in feudal_values:
        data.append(val)
        labels.append("Best-Feudal\n(P=3, λ=0.05)")

    # Create figure
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create violin plot
    parts = ax.violinplot(
        [baseline_values, feudal_values],
        positions=[0, 1],
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    # Style the violin plot
    for pc in parts["bodies"]:
        pc.set_facecolor("#E5E7EB")
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Style the parts
    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1.5)

    # Add box plot on top
    bp = ax.boxplot(
        [baseline_values, feudal_values],
        positions=[0, 1],
        widths=0.3,
        patch_artist=True,
        showfliers=False,  # We'll show individual points with swarm
    )

    # Style box plot
    colors = ["#6B7280", "#8B5CF6"]  # Gray for baseline, Purple for feudal
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    # Style other box plot elements
    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        if element in bp:
            for item in bp[element]:
                item.set_color("black")
                item.set_linewidth(1.5)

    # Add swarm plot (individual points)
    # Jitter the x positions slightly
    x_baseline = np.random.normal(0, 0.05, len(baseline_values))
    x_feudal = np.random.normal(1, 0.05, len(feudal_values))

    ax.scatter(
        x_baseline,
        baseline_values,
        color="#374151",
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        label="Baseline",
    )
    ax.scatter(
        x_feudal,
        feudal_values,
        color="#7C3AED",
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        label="Best-Feudal",
    )

    # Set labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Baseline\n(n={len(baseline_values)})",
            f"Best-Feudal\n(P=3, λ=0.05)\n(n={len(feudal_values)})",
        ],
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Add statistics text
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values, ddof=1) if len(baseline_values) > 1 else 0
    feudal_mean = np.mean(feudal_values)
    feudal_std = np.std(feudal_values, ddof=1) if len(feudal_values) > 1 else 0

    stats_text = f"Baseline (n={len(baseline_values)}): {baseline_mean:.4f} ± {baseline_std:.4f}\n"
    stats_text += (
        f"Best-Feudal (n={len(feudal_values)}): {feudal_mean:.4f} ± {feudal_std:.4f}"
    )

    # Add statistical test if scipy is available
    if HAS_SCIPY and len(baseline_values) > 1 and len(feudal_values) > 1:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_values, feudal_values)
        stats_text += f"\np-value: {p_value:.4f}"
        if p_value < 0.05:
            stats_text += " *"
        if p_value < 0.01:
            stats_text += "*"
        if p_value < 0.001:
            stats_text += "*"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"replication_distribution_{metric_name}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot: {output_file}")

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
    )

    print("\n✅ Plot generated!")


if __name__ == "__main__":
    main()
