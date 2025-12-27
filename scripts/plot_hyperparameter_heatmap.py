#!/usr/bin/env python3
"""
Create heatmap showing manager_period × feudal_weight grid.

Purpose: Show global structure, avoid cherry-picking.

Features:
- x-axis: manager_period
- y-axis: feudal_weight
- color: loss (or accuracy)
- Handles missing grid points (blank cells are fine)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def load_experiment_data(experiment_files: List[Path]) -> Dict:
    """
    Load all experiment data and create a grid.

    Returns:
        grid_data: Dict mapping (period, weight) -> list of metric values
    """
    grid_data = {}  # {(period, weight): {metric_name: [values]}}

    for exp_file in experiment_files:
        with open(exp_file, "r") as f:
            data = json.load(f)

        # Handle nested structure
        if isinstance(data, dict):
            results = []
            for key in ["manager_period_sweep", "replications", "feudal_weight_sweep"]:
                if key in data:
                    if isinstance(data[key], list):
                        results.extend(data[key])
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

            # Exclude weight=0.0 (baseline) from heatmap to avoid baseline mismatch
            # Baseline should be shown separately in other plots with unified values
            if weight == 0.0:
                continue

            key = (period, weight)
            if key not in grid_data:
                grid_data[key] = {
                    "lm_loss": [],
                    "accuracy": [],
                }

            metrics = result.get("metrics", {})
            if "final_lm_loss" in metrics:
                grid_data[key]["lm_loss"].append(metrics["final_lm_loss"])
            if "final_accuracy" in metrics:
                grid_data[key]["accuracy"].append(metrics["final_accuracy"])

    return grid_data


def create_heatmap(
    grid_data: Dict,
    output_dir: Path,
    metric_name: str = "lm_loss",
    title: str = "Hyperparameter Heatmap",
    ylabel: str = "Feudal Loss Weight",
    xlabel: str = "Manager Period",
    cmap_name: str = "YlOrRd",
):
    """
    Create heatmap from grid data.

    Args:
        grid_data: Dict mapping (period, weight) -> metric lists
        output_dir: Directory to save plot
        metric_name: Name of metric to plot
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        cmap_name: Colormap name
    """
    # Extract all unique periods and weights
    # Exclude weight=0.0 as it's not true baseline (true baseline comes from baseline_comparison files)
    periods = sorted(set(p for p, w in grid_data.keys()))
    weights = sorted(set(w for p, w in grid_data.keys() if w > 0))

    if not periods or not weights:
        print(f"⚠️  No data found. Skipping {metric_name} heatmap.")
        return

    # Create grid matrix
    grid_matrix = np.full((len(weights), len(periods)), np.nan)

    # Fill in the grid
    for i, weight in enumerate(weights):
        for j, period in enumerate(periods):
            key = (period, weight)
            if key in grid_data:
                values = grid_data[key].get(metric_name, [])
                if values:
                    # Use mean if multiple values
                    grid_matrix[i, j] = np.mean(values)

    # Create figure
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(10, 7))

    # Choose colormap
    try:
        # New matplotlib API
        cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError:
        # Fallback for older matplotlib
        cmap = plt.cm.get_cmap(cmap_name)

    # Create heatmap with better colormap scaling
    vmin = np.nanmin(grid_matrix)
    vmax = np.nanmax(grid_matrix)

    im = ax.imshow(
        grid_matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(periods)))
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_yticks(np.arange(len(weights)))
    # Format weights nicely
    weight_labels = []
    for w in weights:
        if w == 0:
            weight_labels.append("0.00")
        elif w == int(w):
            weight_labels.append(f"{int(w)}.00")
        else:
            weight_labels.append(f"{w:.2f}")
    ax.set_yticklabels(weight_labels, fontsize=11)

    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    metric_label = "Loss" if metric_name == "lm_loss" else "Accuracy"
    cbar.set_label(metric_label, fontsize=12, fontweight="bold")

    # Add legend for missing cells
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="lightgray", edgecolor="white", alpha=0.3, label="Not evaluated"
        )
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    # Add text annotations for each cell
    for i, weight in enumerate(weights):
        for j, period in enumerate(periods):
            value = grid_matrix[i, j]
            if not np.isnan(value):
                # Format value
                text = f"{value:.3f}"

                # Choose text color based on background brightness
                # Get the color of the cell from the colormap
                normalized_value = (
                    (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                )
                if metric_name == "lm_loss":
                    # For reversed colormap, low values are bright (yellow)
                    # Use black text for bright backgrounds, white for dark
                    cell_color = cmap(1.0 - normalized_value)  # Reversed
                else:
                    # For normal colormap, high values are bright
                    cell_color = cmap(normalized_value)

                # Calculate brightness (luminance)
                # Using relative luminance formula
                r, g, b = cell_color[0], cell_color[1], cell_color[2]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.5 else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                # Mark missing data with a subtle background
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=True,
                        facecolor="lightgray",
                        edgecolor="white",
                        linewidth=1,
                        alpha=0.3,
                        zorder=0,
                    )
                )
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=14,
                    fontweight="bold",
                    alpha=0.6,
                )

    # Add legend for missing data
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="lightgray", alpha=0.3, edgecolor="white", label="Not evaluated"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"hyperparameter_heatmap_{metric_name}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved heatmap: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create hyperparameter heatmap (manager_period × feudal_weight)"
    )
    parser.add_argument(
        "--experiment-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSON files with experiment results",
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
    grid_data = load_experiment_data(args.experiment_files)

    periods = sorted(set(p for p, w in grid_data.keys()))
    weights = sorted(set(w for p, w in grid_data.keys()))

    print(f"Found periods: {periods}")
    print(f"Found weights: {weights}")
    print(f"Total grid points: {len(grid_data)}")

    # Create heatmaps
    print("\nGenerating heatmaps...")

    # Loss heatmap
    # Note: weight=0.0 entries are from sweep runs, not true baseline
    # True baseline (no feudal) should be excluded or labeled separately
    create_heatmap(
        grid_data,
        args.output_dir,
        metric_name="lm_loss",
        title="Hyperparameter Heatmap: Manager Period × Feudal Weight\n(Language Model Loss - Lower is Better)\nNote: Only subset of grid evaluated (see filled cells)",
        ylabel="Feudal Loss Weight",
        xlabel="Manager Period",
        cmap_name="YlOrRd",  # Yellow-Orange-Red: yellow = low loss (good), red = high loss (bad)
    )

    # Accuracy heatmap - only create if we have data across multiple periods
    periods = sorted(set(p for p, w in grid_data.keys()))
    accuracy_periods = set()
    for (p, w), metrics in grid_data.items():
        if metrics.get("accuracy"):
            accuracy_periods.add(p)

    if len(accuracy_periods) > 1:
        # Multiple periods have accuracy data
        create_heatmap(
            grid_data,
            args.output_dir,
            metric_name="accuracy",
            title="Hyperparameter Heatmap: Manager Period × Feudal Weight\n(Accuracy - Higher is Better)\nNote: Only subset of grid evaluated (see filled cells)",
            ylabel="Feudal Loss Weight",
            xlabel="Manager Period",
            cmap_name="YlGn",  # Yellow-Green: yellow = low accuracy, green = high accuracy (good)
        )
    elif len(accuracy_periods) == 1:
        # Only single period - create with descriptive title
        period = sorted(accuracy_periods)[0]
        create_heatmap(
            grid_data,
            args.output_dir,
            metric_name="accuracy",
            title=f"Accuracy across Feudal Weights (Period={period} only; other cells unmeasured)\nNote: Only subset of grid evaluated (see filled cells)",
            ylabel="Feudal Loss Weight",
            xlabel="Manager Period",
            cmap_name="YlGn",  # Yellow-Green: yellow = low accuracy, green = high accuracy (good)
        )
    else:
        # No accuracy data
        print(
            "⚠️  No accuracy data found. Skipping accuracy heatmap (use accuracy line plot instead)."
        )

    print("\n✅ All heatmaps generated!")


if __name__ == "__main__":
    main()
