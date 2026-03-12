"""Shared publication-quality plotting utilities for Bayesian autoresearcher.

Provides consistent style, color palette, results.tsv parsing, and the
ELPD trajectory + waterfall plot that works across all problem directories.
Problem-specific plots (causal forest plots, model diagrams) stay in
problems/<name>/visualize.py and import from here.
"""

import csv
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Color palette — consistent across all problems
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "kept": "#1b7837",
    "kept_light": "#a6dba0",
    "discarded": "#e08214",
    "crashed": "#c2185b",
    "baseline_ref": "#888888",
    "waterfall_gain": "#1b7837",
    "waterfall_minor": "#a6dba0",
    "confounder": "#3274a1",
    "treatment": "#2ca02c",
    "interaction": "#9467bd",
    "noise": "#c44e52",
    "bg_box": "#f7f7f7",
}

# Threshold for "major" contribution in waterfall (ELPD points)
MAJOR_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def set_pub_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Data types & parsing
# ---------------------------------------------------------------------------

class ExperimentRow(NamedTuple):
    """One row from results.tsv."""
    index: int
    commit: str
    elpd: float
    elpd_se: float
    ate_hdi_width: float
    convergence: str
    status: str
    descr: str


def parse_results(tsv_path: Path) -> list[ExperimentRow]:
    """Parse results.tsv into a list of ExperimentRow."""
    rows: list[ExperimentRow] = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=1):
            if not row.get("commit", "").strip():
                continue
            rows.append(ExperimentRow(
                index=i,
                commit=row["commit"].strip(),
                elpd=float(row["elpd"]),
                elpd_se=float(row["elpd_se"]),
                ate_hdi_width=float(row["ate_hdi_width"]),
                convergence=row["convergence"].strip(),
                status=row["status"].strip(),
                descr=row["descr"].strip(),
            ))
    return rows


def split_by_status(
    rows: list[ExperimentRow],
) -> tuple[list[ExperimentRow], list[ExperimentRow], list[ExperimentRow]]:
    """Split rows into (kept, discarded, crashed) lists."""
    kept = [r for r in rows if r.status == "keep"]
    discarded = [r for r in rows if r.status == "discard"]
    crashed = [r for r in rows if r.status == "crash"]
    return kept, discarded, crashed


# ---------------------------------------------------------------------------
# ELPD trajectory + waterfall plot (works for any problem)
# ---------------------------------------------------------------------------

def plot_iterations(
    rows: list[ExperimentRow],
    output_path: Path,
    kept_labels: dict[int, str] | None = None,
    title: str = "ELPD Optimization Across Bayesian Model Iterations",
) -> None:
    """Create a two-panel figure: ELPD trajectory + contribution waterfall.

    Args:
        rows: All experiment rows from results.tsv.
        output_path: Where to save the figure.
        kept_labels: Optional mapping of experiment index to short label
            for the waterfall panel. Falls back to truncated description.
        title: Title for the trajectory panel.
    """
    if kept_labels is None:
        kept_labels = {}

    kept, discarded, crashed = split_by_status(rows)

    valid_elpds = [r.elpd for r in rows if r.status != "crash"]
    if not valid_elpds:
        valid_elpds = [r.elpd for r in rows]
    crash_floor = min(valid_elpds) - 12

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35},
    )

    # ---- Panel A: ELPD trajectory ----
    ax = ax_top
    ax.set_title(title, fontweight="bold", pad=10)

    if crashed:
        ax.scatter(
            [r.index for r in crashed],
            [crash_floor] * len(crashed),
            marker="x", s=40, c=COLORS["crashed"], zorder=4,
            linewidths=1.2, label="Convergence failure",
        )

    if discarded:
        ax.scatter(
            [r.index for r in discarded],
            [r.elpd for r in discarded],
            marker="v", s=45, c=COLORS["discarded"], zorder=4,
            linewidths=0.6, edgecolors=COLORS["discarded"],
            label="Discarded (no improvement)",
        )

    kept_x = [r.index for r in kept]
    kept_y = [r.elpd for r in kept]
    kept_se = [r.elpd_se for r in kept]

    ax.fill_between(
        kept_x,
        [y - se for y, se in zip(kept_y, kept_se)],
        [y + se for y, se in zip(kept_y, kept_se)],
        color=COLORS["kept_light"], alpha=0.35, zorder=3,
    )
    ax.plot(
        kept_x, kept_y, "o-",
        color=COLORS["kept"], markerfacecolor="white",
        markeredgecolor=COLORS["kept"], markeredgewidth=1.8,
        zorder=5, label="Kept (improved model)",
    )

    ax.axhline(
        kept_y[0], color=COLORS["baseline_ref"],
        linestyle="--", linewidth=0.9, alpha=0.6,
    )
    ax.text(
        len(rows) + 0.5, kept_y[0], f"baseline\n{kept_y[0]:.0f}",
        fontsize=8, color=COLORS["baseline_ref"], va="center",
    )

    ax.annotate(
        f"Final: ELPD = {kept_y[-1]:.1f}  (+{kept_y[-1] - kept_y[0]:.0f} from baseline)",
        xy=(kept_x[-1], kept_y[-1]),
        xytext=(6, kept_y[-1] + 3),
        fontsize=9.5, fontweight="bold", color=COLORS["kept"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["kept"], lw=1.2,
                        connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["kept"], alpha=0.95),
    )

    ax.axhline(crash_floor + 4, color="#ddd", linestyle=":", linewidth=0.6)
    ax.text(
        0.3, crash_floor, "convergence\nfailures",
        fontsize=7.5, color=COLORS["crashed"], va="center", alpha=0.7,
    )

    ax.set_xlabel("Experiment iteration")
    ax.set_ylabel("ELPD (PSIS-LOO, higher is better)")
    ax.set_xticks(range(0, len(rows) + 2, 5))
    ax.set_xticks(range(1, len(rows) + 1), minor=True)
    ax.set_xlim(0, len(rows) + 2)
    ax.legend(loc="center right", framealpha=0.95, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel B: Contribution waterfall ----
    ax = ax_bot
    ax.set_title(
        "ELPD Improvement Breakdown by Model Change",
        fontweight="bold", pad=10,
    )

    deltas: list[float] = []
    labels: list[str] = []
    for i in range(1, len(kept)):
        delta = kept[i].elpd - kept[i - 1].elpd
        deltas.append(delta)
        label = kept_labels.get(kept[i].index, kept[i].descr[:25])
        labels.append(label)

    cumulative = kept[0].elpd
    bottoms: list[float] = []
    for d in deltas:
        bottoms.append(cumulative)
        cumulative += d

    bar_colors = [
        COLORS["waterfall_gain"] if d >= MAJOR_THRESHOLD
        else COLORS["waterfall_minor"]
        for d in deltas
    ]

    y_positions = np.arange(len(deltas))
    ax.barh(
        y_positions, deltas, left=bottoms, height=0.6,
        color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3,
    )

    for yp, d, b in zip(y_positions, deltas, bottoms):
        is_major = d >= MAJOR_THRESHOLD
        x_label = b + d + 0.4
        ax.text(
            x_label, yp, f"+{d:.1f}",
            fontsize=9.5 if is_major else 8,
            fontweight="bold" if is_major else "normal",
            va="center", ha="left",
            color=COLORS["waterfall_gain"] if is_major else "#888",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative ELPD")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total_gain = kept[-1].elpd - kept[0].elpd
    ax.text(
        0.98, 0.05,
        f"Total: +{total_gain:.1f} ELPD",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        ha="right", va="bottom", color=COLORS["kept"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["kept_light"],
                  alpha=0.3, edgecolor=COLORS["kept"]),
    )

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")
