#!/usr/bin/env python3
"""
Shared plotting style configuration for consistent, professional figures.

Usage:
    from plot_style import apply_style, COLORS, FIGSIZE, save_figure
"""

import matplotlib.pyplot as plt

# =============================================================================
# Figure dimensions and DPI
# =============================================================================
FIGSIZE = (10, 6)  # inches
DPI = 300

# =============================================================================
# Font sizes (professional publication style)
# =============================================================================
FONT_TITLE = 17
FONT_AXIS_LABEL = 13
FONT_TICK = 11
FONT_LEGEND = 10
FONT_ANNOTATION = 10
FONT_FOOTNOTE = 9

# =============================================================================
# Line and marker styles
# =============================================================================
LINE_WIDTH = 2.2
LINE_WIDTH_REFERENCE = 2.0
MARKER_SIZE = 8
MARKER_SIZE_HIGHLIGHT = 14  # For "optimal" star markers
MARKER_EDGE_WIDTH = 1.2
CAPSIZE = 5
CAPTHICK = 1.5
ERROR_LINE_WIDTH = 1.5

# =============================================================================
# Color palette
# =============================================================================
COLORS = {
    # Primary data colors
    "primary": "#7C3AED",  # Purple-600 for main feudal data
    "secondary": "#059669",  # Emerald-600 for secondary data
    "tertiary": "#DC2626",  # Red-600 for tertiary data
    # Reference/baseline colors
    "baseline": "#4B5563",  # Gray-600 for baseline reference
    "baseline_fill": "#9CA3AF",  # Gray-400 for baseline band
    # Highlight colors
    "optimal": "#F59E0B",  # Amber-500 for optimal point star
    "optimal_edge": "#78350F",  # Amber-900 for star edge
    # Annotation colors
    "annotation_bg": "#FEF3C7",  # Amber-100 background
    "annotation_edge": "#92400E",  # Amber-800 edge
    # Lambda=0 special point
    "lambda_zero": "#6B7280",  # Gray-500 for λ=0 point
    # Box/violin plot colors
    "box_baseline": "#6B7280",
    "box_feudal": "#8B5CF6",
    "scatter_baseline": "#374151",
    "scatter_feudal": "#7C3AED",
}

# =============================================================================
# Style application
# =============================================================================


def apply_style():
    """
    Apply consistent matplotlib style for professional figures.
    Call this at the start of each plotting function.
    """
    # Reset to default first
    plt.rcdefaults()

    # Use a clean white background style - no seaborn styles
    plt.rcParams.update(
        {
            # Figure
            "figure.figsize": FIGSIZE,
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            # Axes - pure white background
            "axes.facecolor": "white",
            "axes.edgecolor": "#9CA3AF",  # Gray-400 for axis borders
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": FONT_TITLE,
            "axes.titleweight": "bold",
            "axes.titlepad": 15,
            "axes.labelsize": FONT_AXIS_LABEL,
            "axes.labelweight": "bold",
            "axes.labelpad": 8,
            # Grid - very subtle, just horizontal lines
            "grid.color": "#D1D5DB",  # Gray-300
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            # Ticks
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "xtick.color": "#374151",  # Gray-700
            "ytick.color": "#374151",
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend
            "legend.fontsize": FONT_LEGEND,
            "legend.framealpha": 0.98,
            "legend.edgecolor": "#D1D5DB",
            "legend.facecolor": "white",
            "legend.frameon": True,
            "legend.borderpad": 0.5,
            "legend.labelspacing": 0.4,
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": FONT_TICK,
            # Lines
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "lines.markeredgewidth": MARKER_EDGE_WIDTH,
            # Savefig
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def save_figure(fig, output_path, tight=True):
    """
    Save figure with consistent settings.

    Args:
        fig: matplotlib figure
        output_path: Path to save the figure
        tight: Whether to use tight_layout
    """
    if tight:
        fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    print(f"✅ Saved: {output_path}")


def add_footnote(ax, text, y_offset=-0.14):
    """
    Add a small footnote below the plot.

    Args:
        ax: matplotlib axes
        text: Footnote text
        y_offset: Vertical offset from axes bottom (negative = below)
    """
    ax.text(
        0.5,
        y_offset,
        text,
        transform=ax.transAxes,
        ha="center",
        fontsize=FONT_FOOTNOTE,
        style="italic",
        color="#4B5563",  # Gray-600 - readable but subtle
    )


def format_baseline_label():
    """Return the standard baseline label."""
    return "HRM baseline (no subgoal head)"


def format_lambda_zero_label():
    """Return the standard λ=0 label."""
    return "Subgoal head only (λ=0)"
