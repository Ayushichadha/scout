#!/usr/bin/env python3
"""Generate the Scout K=2 paper figures from preserved experiment artifacts.

Run from anywhere with:
    .venv/bin/python experiments/schedule_sweep_k2/figures/generate_figures.py

All plotted experiment values are read from ``per_episode_matrix.csv`` or
``ceiling_analysis.md``.  No experiment is rerun and no CI is recomputed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scout-mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent
MATRIX_PATH = SOURCE_DIR / "per_episode_matrix.csv"
ANALYSIS_PATH = SOURCE_DIR / "ceiling_analysis.md"
PROVENANCE_PATH = SOURCE_DIR / "provenance.json"

# Cohesive paper palette.  Semantic use is fixed across figures:
# teal=fixed/retained, gold=oracle/new goal, coral=short/myopic, navy=structure.
INK = "#172B3A"
MUTED = "#647785"
LIGHT = "#E7EDF0"
PALE = "#F5F7F8"
TEAL = "#0B7A75"
TEAL_LIGHT = "#D6ECE9"
GOLD = "#E1A72B"
GOLD_LIGHT = "#F7EBCB"
CORAL = "#D95D4F"
CORAL_LIGHT = "#F7DEDA"
WHITE = "#FFFFFF"


def configure_style() -> None:
    """Set deterministic, compact typography for NeurIPS-sized figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": LIGHT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.grid": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "scout-k2-paper-figures",
            "path.simplify": False,
        }
    )


def load_sources() -> tuple[pd.DataFrame, str, dict]:
    df = pd.read_csv(MATRIX_PATH)
    analysis = ANALYSIS_PATH.read_text(encoding="utf-8")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    required = {
        "episode_id",
        "schedule_k",
        "dwell_first",
        "dwell_second",
        "tokens_correct",
        "tokens_valid",
        "token_acc_episode",
        "passes_executed",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"matrix is missing columns: {sorted(missing)}")
    if len(df) != 22_116 or df["episode_id"].nunique() != 3_686:
        raise ValueError("unexpected matrix dimensions")
    return df, analysis, provenance


def save_pair(fig: plt.Figure, stem: str) -> None:
    """Write reproducible vector and 300 dpi raster versions."""
    common_metadata = {"Creator": "Scout figure generator"}
    fig.savefig(
        HERE / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={**common_metadata, "Date": None},
    )
    fig.savefig(
        HERE / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Software": "Scout figure generator"},
    )
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LIGHT)
    ax.tick_params(length=3, width=0.7, color=LIGHT)


def add_source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.004, text, fontsize=5.3, color=MUTED, ha="left", va="bottom")


def figure_persistence(df: pd.DataFrame) -> None:
    # DATA TRACE: every point is the requested macro per-episode accuracy:
    # df.groupby(["schedule_k", "dwell_first", "dwell_second"], as_index=False)
    #   .agg(macro_accuracy=("token_acc_episode", "mean"))
    summary = (
        df.groupby(["schedule_k", "dwell_first", "dwell_second"], as_index=False)
        .agg(macro_accuracy=("token_acc_episode", "mean"))
        .sort_values("dwell_first")
    )
    peak = summary.loc[summary["macro_accuracy"].idxmax()]

    fig, ax = plt.subplots(figsize=(3.30, 2.72))
    # Regime boundaries are midpoints between observed dwells adjacent to the
    # observed peak; shading does not add or interpolate measurements.
    ax.axvspan(0.7, peak.dwell_first - 0.5, color=CORAL_LIGHT, zorder=0)
    ax.axvspan(peak.dwell_first + 0.5, 6.3, color=GOLD_LIGHT, alpha=0.55, zorder=0)
    ax.plot(
        summary["dwell_first"],
        summary["macro_accuracy"],
        color=TEAL,
        lw=2.4,
        solid_capstyle="round",
        zorder=2,
    )
    ax.scatter(
        summary["dwell_first"],
        summary["macro_accuracy"],
        s=31,
        color=WHITE,
        edgecolor=TEAL,
        linewidth=1.6,
        zorder=3,
    )
    ax.scatter(
        [peak.dwell_first],
        [peak.macro_accuracy],
        s=58,
        color=GOLD,
        edgecolor=WHITE,
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        f"peak  {peak.macro_accuracy:.1%}\n{int(peak.dwell_first)} passes",
        xy=(peak.dwell_first, peak.macro_accuracy),
        xytext=(peak.dwell_first + 0.28, peak.macro_accuracy + 0.0043),
        fontsize=7,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.0),
    )
    ax.text(1.15, 0.4978, "too brief", color=CORAL, fontsize=6.5, fontweight="bold")
    ax.text(5.02, 0.4978, "mild decay", color="#9A6B08", fontsize=6.5, ha="center")
    ax.set(
        xlim=(0.7, 6.3),
        ylim=(0.457, 0.501),
        xticks=summary["dwell_first"].astype(int).tolist(),
        yticks=[0.46, 0.48, 0.50],
        xlabel="First commitment dwell (worker passes)",
        ylabel="Macro token accuracy",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Persistence has a narrow sweet spot", loc="left", pad=7)
    schedule_count = int(summary["schedule_k"].nunique())
    ax.text(
        1.0,
        1.01,
        f"Zoomed y-axis · {schedule_count} measured schedules",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )
    clean_axis(ax)
    episodes_per_schedule = int(
        summary.shape[0] and df.groupby("schedule_k").size().iloc[0]
    )
    add_source_note(
        fig,
        f"Source: per_episode_matrix.csv · macro mean over {episodes_per_schedule:,} episodes/schedule",
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.84)
    save_pair(fig, "fig1_persistence_curve")


def parse_ceiling_values(analysis: str) -> dict[str, float]:
    """Copy the published point estimates and bootstrap CI from the markdown."""
    # DATA TRACE: these are copied directly from ceiling_analysis.md's
    # "Fixed policy, floor, and held-out-label upper bound" bullets.
    patterns = {
        "best": r"Best fixed clock: .* micro token accuracy `([0-9.]+)`",
        "floor": r"Floor .*: `([0-9.]+)`",
        "ceiling": r"Oracle ceiling[^\n]*:.*?`([0-9.]+)`",
        "headroom": r"headroom:.*?`([0-9.]+)` \(95% episode-bootstrap CI `\[([0-9.]+), ([0-9.]+)\]`\)",
    }
    best = float(re.search(patterns["best"], analysis).group(1))
    floor = float(re.search(patterns["floor"], analysis).group(1))
    ceiling = float(re.search(patterns["ceiling"], analysis).group(1))
    match = re.search(patterns["headroom"], analysis)
    return {
        "best": best,
        "floor": floor,
        "ceiling": ceiling,
        "headroom": float(match.group(1)),
        "ci_low": float(match.group(2)),
        "ci_high": float(match.group(3)),
    }


def figure_headroom(analysis: str) -> None:
    values = parse_ceiling_values(analysis)
    # DATA TRACE: captured gain and capture fraction are exact arithmetic on
    # the three point estimates copied above; no new estimate is introduced.
    captured = values["best"] - values["floor"]
    total = values["ceiling"] - values["floor"]
    captured_fraction = captured / total

    fig = plt.figure(figsize=(3.30, 3.18))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.72], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    y = 0.52
    ax.plot(
        [values["floor"], values["ceiling"]],
        [y, y],
        color=LIGHT,
        lw=15,
        solid_capstyle="round",
    )
    ax.plot(
        [values["floor"], values["best"]],
        [y, y],
        color=TEAL,
        lw=15,
        solid_capstyle="round",
    )
    ax.plot(
        [values["best"], values["ceiling"]],
        [y, y],
        color=GOLD,
        lw=15,
        solid_capstyle="butt",
    )
    for x, color in [
        (values["floor"], INK),
        (values["best"], TEAL),
        (values["ceiling"], GOLD),
    ]:
        ax.scatter(
            [x], [y], s=27, color=color, edgecolor=WHITE, linewidth=0.8, zorder=3
        )
    ax.text(
        values["floor"],
        0.20,
        f"random clock\n{values['floor']:.2%}",
        ha="left",
        va="top",
        fontsize=6.6,
    )
    ax.text(
        values["best"],
        0.20,
        f"best fixed\n{values['best']:.2%}",
        ha="center",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(
        values["ceiling"],
        0.84,
        f"oracle ceiling\n{values['ceiling']:.2%}",
        ha="right",
        va="bottom",
        fontsize=6.6,
        color="#9A6B08",
    )
    ax.text(
        (values["floor"] + values["best"]) / 2,
        0.55,
        f"{captured_fraction:.0%} captured",
        ha="center",
        va="center",
        fontsize=7,
        color=WHITE,
        fontweight="bold",
    )
    ax.set_xlim(values["floor"] - total * 0.05, values["ceiling"] + total * 0.05)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title("A fixed clock captures most timing value", loc="left", pad=8)
    ax.text(
        1,
        1.02,
        "accuracy scale is zoomed",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.8,
        color=MUTED,
    )

    ax2 = fig.add_subplot(gs[1])
    # DATA TRACE: headroom and whiskers are copied verbatim from the published
    # 10,000-replicate episode-bootstrap result in ceiling_analysis.md.
    estimate_pp = values["headroom"] * 100
    ci_low_pp = values["ci_low"] * 100
    ci_high_pp = values["ci_high"] * 100
    ax2.axvline(0, color=LIGHT, lw=1)
    ax2.errorbar(
        estimate_pp,
        0,
        xerr=[[estimate_pp - ci_low_pp], [ci_high_pp - estimate_pp]],
        fmt="o",
        color=GOLD,
        ecolor=GOLD,
        elinewidth=2,
        capsize=3,
        markersize=5,
    )
    ax2.text(
        estimate_pp,
        0.27,
        f"{estimate_pp:.2f} pp",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#9A6B08",
    )
    ax2.text(
        estimate_pp,
        -0.27,
        f"95% CI  {ci_low_pp:.2f}–{ci_high_pp:.2f}",
        ha="center",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    ax2.set_xlim(-0.025, max(0.22, ci_high_pp * 1.15))
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_yticks([])
    ax2.set_xlabel("Remaining oracle headroom (accuracy points)")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines["bottom"].set_color(LIGHT)
    ax2.tick_params(axis="x", length=3, color=LIGHT)
    add_source_note(
        fig, "Source: ceiling_analysis.md · published point estimates and bootstrap CI"
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88)
    save_pair(fig, "fig2_clock_headroom")


def parse_reversed_pairs(analysis: str) -> pd.DataFrame:
    """Read the three published reversed-order effects from markdown."""
    section = analysis.split("## Paired reversed-order dwell test", 1)[1].split(
        "## Preregistered", 1
    )[0]
    pattern = re.compile(
        r"\| `\[1,(\d)\] - \[1,(\d)\]` \| ([+-]?[0-9.]+) \| "
        r"\[([+-]?[0-9.]+), ([+-]?[0-9.]+)\] \|"
    )
    rows = []
    # DATA TRACE: estimates and CIs are copied directly from the paired table.
    # We multiply all three by -1 so the displayed orientation is consistently
    # "longer first dwell minus shorter first dwell"; CI endpoints are reversed.
    for short_k, long_k, estimate, low, high in pattern.findall(section):
        short_k, long_k = int(short_k), int(long_k)
        rows.append(
            {
                "short_k": short_k,
                "long_k": long_k,
                "estimate": -float(estimate),
                "low": -float(high),
                "high": -float(low),
            }
        )
    if len(rows) != 3:
        raise ValueError("could not parse all three reversed-order comparisons")
    return (
        pd.DataFrame(rows)
        .sort_values("estimate", ascending=True)
        .reset_index(drop=True)
    )


def figure_reversed_order(analysis: str, df: pd.DataFrame) -> None:
    pairs = parse_reversed_pairs(analysis)
    # DATA TRACE: dwell labels come from the unique schedule metadata in the
    # matrix, not from an inferred total-pass count.
    dwell_lookup = (
        df.groupby("schedule_k", as_index=True)[["dwell_first", "dwell_second"]]
        .first()
        .astype(int)
    )
    pairs["label"] = pairs.apply(
        lambda row: f"[1,{int(row.long_k)}] − [1,{int(row.short_k)}]", axis=1
    )
    pairs["dwell"] = pairs.apply(
        lambda row: (
            f"({dwell_lookup.loc[int(row.long_k), 'dwell_first']},"
            f"{dwell_lookup.loc[int(row.long_k), 'dwell_second']}) vs "
            f"({dwell_lookup.loc[int(row.short_k), 'dwell_first']},"
            f"{dwell_lookup.loc[int(row.short_k), 'dwell_second']})"
        ),
        axis=1,
    )
    y = np.arange(len(pairs))
    estimate = pairs["estimate"].to_numpy() * 100
    low = pairs["low"].to_numpy() * 100
    high = pairs["high"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(3.30, 2.55))
    ax.axvline(0, color=INK, lw=1.0, alpha=0.6, zorder=0)
    ax.errorbar(
        estimate,
        y,
        xerr=[estimate - low, high - estimate],
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        elinewidth=2.1,
        capsize=3,
        markersize=5.5,
        zorder=2,
    )
    for xi, yi, row in zip(estimate, y, pairs.itertuples()):
        ax.text(
            xi + 0.035,
            yi,
            f"+{xi:.2f}",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=TEAL,
        )
    labels = [f"{row.label}\n{row.dwell}" for row in pairs.itertuples()]
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.08, max(high) + 0.19)
    ax.set_ylim(-0.55, len(pairs) - 0.45)
    ax.set_xlabel("Longer-first advantage (accuracy points)")
    ax.set_title("Order matters in every matched pair", loc="left", pad=7)
    ax.text(
        0.0,
        1.01,
        "shorter first",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )
    ax.text(
        1.0,
        1.01,
        "longer first  →",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=TEAL,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=LIGHT)
    add_source_note(
        fig,
        "Source: ceiling_analysis.md · paired episode-bootstrap 95% CIs; sign reoriented",
    )
    fig.subplots_adjust(left=0.33, right=0.96, bottom=0.24, top=0.83)
    save_pair(fig, "fig3_reversed_order")


def figure_schedule_distributions(df: pd.DataFrame) -> None:
    # DATA TRACE: each panel is an exact empirical CDF computed with no bins,
    # smoothing, interpolation, or subsampling:
    # x = np.sort(group["token_acc_episode"].to_numpy())
    # y = np.arange(1, len(x)+1) / len(x)
    grouped = list(df.sort_values("schedule_k").groupby("schedule_k", sort=True))
    fig, axes = plt.subplots(
        len(grouped), 1, figsize=(3.30, 4.45), sharex=True, sharey=True
    )
    for ax, (k, group) in zip(axes, grouped):
        x = np.sort(group["token_acc_episode"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        dwell_first = int(group["dwell_first"].iloc[0])
        dwell_second = int(group["dwell_second"].iloc[0])
        color = (
            GOLD
            if group["token_acc_episode"].mean()
            == df.groupby("schedule_k")["token_acc_episode"].mean().max()
            else TEAL
        )
        ax.step(x, y, where="post", color=color, lw=1.35)
        ax.axhline(0.5, color=LIGHT, lw=0.7, zorder=0)
        ax.text(
            0.02,
            0.78,
            f"[1,{int(k)}]   dwell {dwell_first}:{dwell_second}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=INK,
        )
        ax.set_ylim(0, 1)
        # A single shared-scale midpoint label avoids collisions between the
        # 0% and 100% endpoints of vertically adjacent panels.
        ax.set_yticks([0.5])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(LIGHT)
        ax.tick_params(length=2, width=0.6, color=LIGHT)
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xticks([0, 0.5, 1.0])
    axes[-1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[-1].set_xlabel("Per-episode token accuracy")
    fig.supylabel("Episodes at or below accuracy", x=0.02, fontsize=7.5)
    axes[0].set_title(
        "The full schedule distributions shift together", loc="left", pad=9
    )
    axes[0].text(
        1,
        1.05,
        "exact ECDF · no smoothing",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=5.8,
        color=MUTED,
    )
    episodes_per_schedule = int(df.groupby("schedule_k").size().iloc[0])
    add_source_note(
        fig,
        f"Source: per_episode_matrix.csv · all {episodes_per_schedule:,} episode accuracies per schedule",
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.13, top=0.90, hspace=0.08)
    save_pair(fig, "fig4_schedule_distributions")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.0,
    radius: float = 0.02,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.0,
    style: str = "-|>",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def figure_architecture(df: pd.DataFrame) -> None:
    # DATA TRACE: timeline length comes directly from passes_executed; the two
    # example schedules and dwell spans are selected from observed matrix rows.
    n_passes = int(df["passes_executed"].max())
    schedule_rows = (
        df.loc[
            df["schedule_k"].isin([2, 5]), ["schedule_k", "dwell_first", "dwell_second"]
        ]
        .drop_duplicates()
        .sort_values("schedule_k")
    )
    if n_passes != 8 or len(schedule_rows) != 2:
        raise ValueError(
            "architecture timeline expects the preserved eight-pass [1,2]/[1,5] schedules"
        )

    fig, ax = plt.subplots(figsize=(6.90, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02, 0.965, "Manager–worker timing", fontsize=12, fontweight="bold", va="top"
    )
    ax.text(
        0.02,
        0.915,
        "A decision made after pass m changes pass m+1 — never pass m itself.",
        fontsize=7.5,
        color=MUTED,
        va="top",
    )

    # Panel A: causal mechanism.  Pass labels are symbolic, not measured values.
    ax.text(0.02, 0.82, "A  Coupled loops", fontsize=7.4, fontweight="bold")
    rounded_box(ax, (0.04, 0.57), 0.17, 0.14, fc=TEAL_LIGHT, ec=TEAL, lw=1.2)
    ax.text(
        0.125,
        0.655,
        "SLOW MANAGER",
        ha="center",
        va="center",
        fontsize=7.2,
        color=TEAL,
        fontweight="bold",
    )
    ax.text(
        0.125,
        0.608,
        "observe · retain / replace",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
    )

    pass_x = [0.29, 0.41, 0.53, 0.70]
    pass_labels = ["pass m−2", "pass m−1", "pass m", "pass m+1"]
    for x, label in zip(pass_x, pass_labels):
        rounded_box(ax, (x, 0.34), 0.095, 0.12, fc=PALE, ec=LIGHT, lw=0.9)
        ax.text(
            x + 0.0475,
            0.40,
            label,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="bold",
        )
    for x1, x2 in zip(pass_x[:-1], pass_x[1:]):
        arrow(ax, (x1 + 0.096, 0.40), (x2 - 0.006, 0.40), color=MUTED, lw=0.8)
    ax.text(
        0.23,
        0.39,
        "FAST\nWORKER",
        ha="center",
        va="center",
        fontsize=7,
        color=INK,
        fontweight="bold",
    )

    # Literal persistence span: the same active goal covers three worker passes.
    ax.plot([0.302, 0.613], [0.505, 0.505], color=TEAL, lw=7, solid_capstyle="round")
    ax.text(
        0.457,
        0.525,
        "same subgoal persists",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=TEAL,
        fontweight="bold",
    )
    ax.text(
        0.302,
        0.49,
        "gₜ",
        ha="center",
        va="top",
        fontsize=7,
        color=TEAL,
        fontweight="bold",
    )

    decision_x = 0.647
    diamond = Polygon(
        [
            [decision_x, 0.67],
            [decision_x + 0.045, 0.62],
            [decision_x, 0.57],
            [decision_x - 0.045, 0.62],
        ],
        closed=True,
        facecolor=WHITE,
        edgecolor=GOLD,
        linewidth=1.25,
    )
    ax.add_patch(diamond)
    ax.text(
        decision_x,
        0.62,
        "retain?",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
    )
    arrow(ax, (0.578, 0.46), (decision_x - 0.018, 0.574), color=INK, lw=0.9)
    ax.text(
        0.602,
        0.515,
        "observe state\nafter pass m",
        ha="center",
        va="center",
        fontsize=5.8,
        color=MUTED,
    )
    arrow(ax, (0.21, 0.64), (decision_x - 0.05, 0.625), color=TEAL, lw=1.0)
    arrow(ax, (decision_x + 0.047, 0.62), (0.747, 0.47), color=GOLD, lw=1.2)
    ax.plot([0.714, 0.783], [0.505, 0.505], color=GOLD, lw=7, solid_capstyle="round")
    ax.text(
        0.749,
        0.525,
        "gₜ or gₜ₊₁",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#9A6B08",
        fontweight="bold",
    )
    ax.text(0.82, 0.62, "replace", fontsize=6.2, color="#9A6B08", fontweight="bold")
    arrow(
        ax,
        (decision_x + 0.016, 0.67),
        (0.18, 0.72),
        color=TEAL,
        lw=0.8,
        connectionstyle="arc3,rad=0.20",
    )
    ax.text(
        0.40,
        0.755,
        "manager loop runs after worker computation",
        ha="center",
        fontsize=5.9,
        color=MUTED,
    )
    ax.plot([0.655, 0.655], [0.30, 0.72], color=CORAL, lw=0.9, ls=(0, (2, 2)))
    ax.text(
        0.655,
        0.285,
        "decision boundary",
        ha="center",
        va="top",
        fontsize=5.8,
        color=CORAL,
    )

    # Panel B: observed forced schedules connect mechanism to persistence study.
    ax.text(0.02, 0.23, "B  Forced K=2 schedules", fontsize=7.4, fontweight="bold")
    x0, x1 = 0.21, 0.96
    step = (x1 - x0) / n_passes
    for i in range(n_passes):
        ax.text(
            x0 + (i + 0.5) * step,
            0.215,
            str(i + 1),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=MUTED,
        )
    ax.text(
        x0 - 0.018, 0.215, "pass", ha="right", va="bottom", fontsize=5.8, color=MUTED
    )

    timeline_y = [0.145, 0.065]
    for (_, row), y in zip(schedule_rows.iterrows(), timeline_y):
        k = int(row.schedule_k)
        d1, d2 = int(row.dwell_first), int(row.dwell_second)
        ax.text(0.02, y + 0.018, f"[1,{k}]", fontsize=7, fontweight="bold", va="center")
        ax.text(
            0.09, y + 0.018, f"dwell {d1}:{d2}", fontsize=6.1, color=MUTED, va="center"
        )
        # Pass 1 computes before the initial goal exists; measured dwell starts at pass 2.
        rounded_box(
            ax, (x0, y), step - 0.004, 0.036, fc=PALE, ec=LIGHT, lw=0.6, radius=0.007
        )
        ax.text(
            x0 + step / 2,
            y + 0.018,
            "init",
            ha="center",
            va="center",
            fontsize=5.2,
            color=MUTED,
        )
        rounded_box(
            ax,
            (x0 + step, y),
            d1 * step - 0.004,
            0.036,
            fc=TEAL_LIGHT,
            ec=TEAL,
            lw=0.8,
            radius=0.007,
        )
        ax.text(
            x0 + step + d1 * step / 2,
            y + 0.018,
            f"g₁ · {d1}",
            ha="center",
            va="center",
            fontsize=5.8,
            color=TEAL,
            fontweight="bold",
        )
        rounded_box(
            ax,
            (x0 + (1 + d1) * step, y),
            d2 * step - 0.004,
            0.036,
            fc=GOLD_LIGHT,
            ec=GOLD,
            lw=0.8,
            radius=0.007,
        )
        ax.text(
            x0 + (1 + d1) * step + d2 * step / 2,
            y + 0.018,
            f"g₂ · {d2}",
            ha="center",
            va="center",
            fontsize=5.8,
            color="#9A6B08",
            fontweight="bold",
        )
        boundary = x0 + (1 + d1) * step
        ax.plot([boundary, boundary], [y - 0.008, y + 0.052], color=CORAL, lw=0.8)
        ax.text(
            boundary,
            y + 0.057,
            f"after {k}",
            ha="center",
            va="bottom",
            fontsize=5.1,
            color=CORAL,
        )

    add_source_note(
        fig,
        "Timing: HRM/models/hrm/hrm_act_v1.py + subgoal_head.py · schedule spans: per_episode_matrix.csv",
    )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.045, top=0.99)
    save_pair(fig, "fig5_manager_worker_architecture")


def figure_sparse_schedule_grid(df: pd.DataFrame) -> None:
    # DATA TRACE: every colored cell is the requested macro per-episode mean:
    # df.groupby(["dwell_first", "dwell_second"], as_index=False)
    #   .agg(macro_accuracy=("token_acc_episode", "mean"))
    cells = (
        df.groupby(["dwell_first", "dwell_second"], as_index=False)
        .agg(macro_accuracy=("token_acc_episode", "mean"))
        .sort_values("dwell_first")
    )
    first_dwells = np.sort(df["dwell_first"].unique()).astype(int)
    second_dwells = np.sort(df["dwell_second"].unique()).astype(int)
    grid = np.full((len(first_dwells), len(second_dwells)), np.nan)
    first_index = {value: index for index, value in enumerate(first_dwells)}
    second_index = {value: index for index, value in enumerate(second_dwells)}
    for row in cells.itertuples():
        grid[first_index[int(row.dwell_first)], second_index[int(row.dwell_second)]] = (
            row.macro_accuracy
        )

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "scout_accuracy", [CORAL_LIGHT, TEAL_LIGHT, TEAL]
    )
    cmap.set_bad(PALE)
    # The fixed 46–50% color domain matches Figure 1's explicitly zoomed range;
    # direct labels carry the exact values and unevaluated cells remain neutral.
    norm = mpl.colors.Normalize(vmin=0.46, vmax=0.50)

    fig, ax = plt.subplots(figsize=(3.30, 3.12))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="equal")
    for row in cells.itertuples():
        yi = first_index[int(row.dwell_first)]
        xi = second_index[int(row.dwell_second)]
        is_peak = row.macro_accuracy == cells["macro_accuracy"].max()
        ax.text(
            xi,
            yi,
            f"{row.macro_accuracy:.1%}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=WHITE if row.macro_accuracy >= 0.485 else INK,
        )
        if is_peak:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (xi - 0.46, yi - 0.46),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor=GOLD,
                    linewidth=2.0,
                )
            )
    ax.set_xticks(np.arange(len(second_dwells)), second_dwells)
    ax.set_yticks(np.arange(len(first_dwells)), first_dwells)
    ax.set_xlabel("Second commitment dwell")
    ax.set_ylabel("First commitment dwell")
    ax.set_xticks(np.arange(-0.5, len(second_dwells), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(first_dwells), 1), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    ax.set_title("Only one dwell trade-off was tested", loc="left", pad=16)
    ax.text(
        0,
        1.025,
        "blank = not evaluated",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=MUTED,
    )
    ax.text(
        1,
        1.025,
        "46–50% color scale",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=MUTED,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_source_note(
        fig,
        "Source: per_episode_matrix.csv · macro mean over all episodes in each observed cell",
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.82)
    save_pair(fig, "fig6_sparse_schedule_grid")


def oracle_gain_decomposition(df: pd.DataFrame) -> tuple[np.ndarray, int, float]:
    """Return sorted extra-correct-token gains over the best fixed clock."""
    # DATA TRACE: this reproduces the micro-accuracy oracle decomposition used
    # by scripts/run_schedule_sweep_k2.py. Denominators are first verified equal
    # within episode; best fixed is the schedule with the largest total correct
    # count; oracle gain is max(correct across schedules) minus best-fixed correct.
    correct = df.pivot(
        index="episode_id", columns="schedule_k", values="tokens_correct"
    ).sort_index(axis=1)
    valid = df.pivot(
        index="episode_id", columns="schedule_k", values="tokens_valid"
    ).sort_index(axis=1)
    if not valid.nunique(axis=1).eq(1).all():
        raise ValueError("valid-token denominators differ within episode")
    best_k = int(correct.sum(axis=0).idxmax())
    gain_tokens = (correct.max(axis=1) - correct[best_k]).sort_values(ascending=False)
    denominator = float(valid.iloc[:, 0].sum())
    return gain_tokens.to_numpy(dtype=float), best_k, denominator


def figure_oracle_concentration(df: pd.DataFrame, analysis: str) -> None:
    gain_tokens, best_k, denominator = oracle_gain_decomposition(df)
    total_gain = gain_tokens.sum()
    if total_gain <= 0:
        raise ValueError("oracle headroom must be positive")
    cumulative = np.cumsum(gain_tokens) / total_gain
    episode_fraction = np.arange(1, len(gain_tokens) + 1) / len(gain_tokens)
    positive_count = int(np.count_nonzero(gain_tokens))
    positive_fraction = positive_count / len(gain_tokens)
    halfway_count = int(np.searchsorted(cumulative, 0.5) + 1)
    halfway_fraction = halfway_count / len(gain_tokens)

    # DATA TRACE: total_gain / denominator exactly decomposes micro headroom.
    # It is checked against the published point estimate copied from
    # ceiling_analysis.md rather than introducing a new estimate.
    published_headroom = parse_ceiling_values(analysis)["headroom"]
    decomposed_headroom = total_gain / denominator
    if not np.isclose(decomposed_headroom, published_headroom, atol=5e-10):
        raise ValueError("episode gains do not reproduce published headroom")

    fig, ax = plt.subplots(figsize=(3.30, 2.86))
    x = np.concatenate([[0.0], episode_fraction])
    y = np.concatenate([[0.0], cumulative])
    ax.plot(x, y, color=GOLD, lw=2.2, drawstyle="steps-post")
    ax.fill_between(x, 0, y, step="post", color=GOLD_LIGHT, alpha=0.72)
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1.0, ls=(0, (2, 2)), zorder=0)
    ax.scatter(
        [halfway_fraction, positive_fraction],
        [cumulative[halfway_count - 1], 1.0],
        s=[28, 32],
        color=[CORAL, TEAL],
        edgecolor=WHITE,
        linewidth=0.8,
        zorder=3,
    )
    ax.annotate(
        f"half the headroom\ncomes from {halfway_count} episodes ({halfway_fraction:.1%})",
        xy=(halfway_fraction, cumulative[halfway_count - 1]),
        xytext=(0.18, 0.48),
        textcoords="axes fraction",
        fontsize=6.5,
        fontweight="bold",
        color=CORAL,
        arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.8),
    )
    ax.annotate(
        f"all gain: {positive_count} episodes ({positive_fraction:.1%})",
        xy=(positive_fraction, 1.0),
        xytext=(0.43, 0.83),
        textcoords="axes fraction",
        fontsize=6.3,
        color=TEAL,
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.025)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Episodes, ranked by oracle gain")
    ax.set_ylabel("Cumulative micro headroom")
    ax.set_title("Headroom lives in a few episodes", loc="left", pad=7)
    ax.text(
        1,
        1.01,
        f"oracle over best fixed [1,{best_k}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )
    clean_axis(ax)
    add_source_note(
        fig,
        "Source: per_episode_matrix.csv · exact extra-correct-token decomposition of micro headroom",
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.23, top=0.84)
    save_pair(fig, "fig7_oracle_concentration")


def figure_oracle_ties(df: pd.DataFrame, analysis: str) -> None:
    # DATA TRACE: tie multiplicity is computed exactly as in the ceiling script:
    # correct.eq(correct.max(axis=1), axis=0).sum(axis=1).value_counts().
    # Per-episode valid denominators are equal, so token-count and accuracy ties
    # are identical.
    correct = df.pivot(
        index="episode_id", columns="schedule_k", values="tokens_correct"
    ).sort_index(axis=1)
    maxima = correct.max(axis=1)
    multiplicity = correct.eq(maxima, axis=0).sum(axis=1)
    counts = multiplicity.value_counts().sort_index().reindex(range(1, 7), fill_value=0)
    fractions = counts / counts.sum()
    tied_count = int(counts.loc[2:].sum())
    tied_fraction = tied_count / counts.sum()

    # DATA TRACE: check the derived tied count against the value copied directly
    # from ceiling_analysis.md's Oracle argmax distribution paragraph.
    match = re.search(r"([0-9,]+) episodes have two or more tied maxima", analysis)
    if match is None or tied_count != int(match.group(1).replace(",", "")):
        raise ValueError("tie count does not match ceiling_analysis.md")

    fig, ax = plt.subplots(figsize=(3.30, 2.92))
    y = np.arange(1, 7)
    colors = [TEAL, "#B6C3CA", "#B6C3CA", "#B6C3CA", "#B6C3CA", GOLD]
    ax.barh(y, fractions.to_numpy(), height=0.58, color=colors)
    for yi, count, fraction in zip(y, counts, fractions):
        ax.text(
            fraction + 0.012,
            yi,
            f"{int(count):,}  ·  {fraction:.1%}",
            va="center",
            ha="left",
            fontsize=6.4,
            color=INK,
            fontweight="bold" if yi in (1, 6) else "normal",
        )
    ax.set_yticks(y, ["1  unique", "2", "3", "4", "5", "6  all clocks"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(fractions.max() * 1.30, 0.62))
    ax.set_xticks([0, 0.25, 0.50])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Fraction of episodes")
    ax.set_ylabel("Schedules tied for best")
    ax.set_title("Most episodes do not choose a clock", loc="left", pad=17)
    ax.text(
        0,
        1.035,
        f"{tied_fraction:.1%} tied across ≥2 schedules ({tied_count:,} episodes)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=LIGHT)
    add_source_note(
        fig,
        "Source: per_episode_matrix.csv · exact multiplicity of per-episode oracle maxima",
    )
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.22, top=0.79)
    save_pair(fig, "fig8_oracle_tie_structure")


def main() -> None:
    configure_style()
    df, analysis, provenance = load_sources()
    # Read provenance as an integrity check and to keep the generator tied to
    # the preserved artifact family; no provenance number is plotted.
    if provenance.get("split_size") != df["episode_id"].nunique():
        raise ValueError("provenance split size does not match matrix")
    figure_persistence(df)
    figure_headroom(analysis)
    figure_reversed_order(analysis, df)
    figure_schedule_distributions(df)
    figure_architecture(df)
    figure_sparse_schedule_grid(df)
    figure_oracle_concentration(df, analysis)
    figure_oracle_ties(df, analysis)
    print("Generated 8 figures as SVG + 300 dpi PNG in", HERE)


if __name__ == "__main__":
    main()
