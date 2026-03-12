"""Generate publication-quality visualizations for IHDP experiment results.

Produces three figures:
  1. iterations.png — Two-panel optimization journey (trajectory + contribution waterfall)
  2. causal_estimates.png — Forest plot of ATE estimates with HDI intervals
  3. model_diagram.png — Final model architecture with equation and results
"""

import csv
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — publication defaults
# ---------------------------------------------------------------------------

COLORS = {
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
MAJOR_THRESHOLD = 2.0


def _set_pub_style() -> None:
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
# Short labels for kept milestones (index → label)
# ---------------------------------------------------------------------------

KEPT_LABELS: dict[int, str] = {
    1:  "Baseline",
    3:  "+ interactions",
    8:  "+ quadratic X",
    13: "+ propensity score",
    15: "Tighter tx prior",
    16: "Tighter all priors",
    18: "Reduce to 6 tx interactions",
    19: "+ tx x quadratic",
    21: "Remove PS (final)",
}


# ---------------------------------------------------------------------------
# Figure 1: Optimization journey (two panels)
# ---------------------------------------------------------------------------

def plot_iterations(rows: list[ExperimentRow], output_path: Path) -> None:
    """Create a two-panel figure: ELPD trajectory + contribution waterfall."""
    kept, discarded, crashed = split_by_status(rows)

    valid_elpds = [r.elpd for r in rows if r.elpd > -998]
    crash_floor = min(valid_elpds) - 12

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35},
    )

    # ---- Panel A: ELPD trajectory ----
    ax = ax_top
    ax.set_title(
        "ELPD Optimization Across 21 Bayesian Model Iterations",
        fontweight="bold", pad=10,
    )

    # Crashed experiments — small marks at the floor
    if crashed:
        ax.scatter(
            [r.index for r in crashed],
            [crash_floor] * len(crashed),
            marker="x", s=40, c=COLORS["crashed"], zorder=4,
            linewidths=1.2, label="Convergence failure",
        )

    # Discarded experiments
    if discarded:
        ax.scatter(
            [r.index for r in discarded],
            [r.elpd for r in discarded],
            marker="v", s=45, c=COLORS["discarded"], zorder=4,
            linewidths=0.6, edgecolors=COLORS["discarded"],
            label="Discarded (no improvement)",
        )

    # Kept experiments — connected line + error bars
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

    # Baseline reference line
    ax.axhline(
        kept_y[0], color=COLORS["baseline_ref"],
        linestyle="--", linewidth=0.9, alpha=0.6,
    )
    ax.text(
        len(rows) + 0.5, kept_y[0], f"baseline\n{kept_y[0]:.0f}",
        fontsize=8, color=COLORS["baseline_ref"], va="center",
    )

    # Annotate only the final result — place in upper-left clear area
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

    # Crash-floor indicator
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

    # Compute deltas between successive kept models
    deltas: list[float] = []
    labels: list[str] = []
    for i in range(1, len(kept)):
        delta = kept[i].elpd - kept[i - 1].elpd
        deltas.append(delta)
        label = KEPT_LABELS.get(kept[i].index, kept[i].descr[:25])
        labels.append(label)

    # Waterfall: bars stacked from left to right
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
    bars = ax.barh(
        y_positions, deltas, left=bottoms, height=0.6,
        color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3,
    )

    # Value labels on each bar
    for yp, d, b in zip(y_positions, deltas, bottoms):
        is_major = d >= MAJOR_THRESHOLD
        x_label = b + d + 0.4
        ax.text(
            x_label, yp, f"+{d:.1f}",
            fontsize=9.5 if is_major else 8,
            fontweight="bold" if is_major else "normal",
            va="center", ha="left",
            color=COLORS["waterfall_gain"] if is_major
            else "#888",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative ELPD")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mark the total
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


# ---------------------------------------------------------------------------
# Causal estimation data (not in results.tsv — recorded from experiment runs)
# ---------------------------------------------------------------------------

# (index, label, ate_estimate, ate_hdi_width)
# HDI bounds approximated as ATE +/- HDI_width/2 (posterior is ~symmetric)
CAUSAL_DATA: list[tuple[int, str, float, float]] = [
    (1,  "Baseline linear",              3.993, 0.517),
    (3,  "+ Interactions (25)",           3.996, 0.564),
    (8,  "+ Quadratic confounders",       3.993, 0.545),
    (13, "+ Propensity score",            3.995, 0.540),
    (15, "Tighter tx prior",             4.017, 0.536),
    (16, "Tighter all priors",           4.003, 0.526),
    (18, "Reduce to 6 tx interactions",  4.066, 0.484),
    (19, "+ Tx x quadratic",             3.940, 0.467),
    (21, "Remove PS (final)",            3.966, 0.457),
]

TRUE_ATE = 4.0


def plot_causal_estimates(output_path: Path) -> None:
    """Forest-plot showing ATE estimates with HDI intervals across model iterations."""
    n = len(CAUSAL_DATA)
    y_pos = np.arange(n)

    labels = [d[1] for d in CAUSAL_DATA]
    ates = np.array([d[2] for d in CAUSAL_DATA])
    hdi_widths = np.array([d[3] for d in CAUSAL_DATA])
    hdi_low = ates - hdi_widths / 2
    hdi_high = ates + hdi_widths / 2

    fig, (ax_forest, ax_width) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.28},
        sharey=True,
    )

    # ---- Left panel: Forest plot (ATE + HDI) ----
    ax = ax_forest
    ax.set_title(
        "Average Treatment Effect: Point Estimates with 94% HDI",
        fontweight="bold", pad=12, loc="left",
    )

    # Truth reference band
    ax.axvline(TRUE_ATE, color=COLORS["crashed"], linewidth=1.5,
               linestyle="-", alpha=0.8, zorder=2)
    ax.axvspan(TRUE_ATE - 0.02, TRUE_ATE + 0.02, color=COLORS["crashed"],
               alpha=0.06, zorder=1)
    ax.text(TRUE_ATE + 0.015, -0.6, f"True ATE = {TRUE_ATE:.1f}",
            fontsize=9.5, color=COLORS["crashed"], va="center", ha="left",
            fontweight="bold")

    # Color gradient: lighter (early) to darker (later)
    greens = plt.cm.Greens(np.linspace(0.3, 0.85, n))

    for i in range(n):
        color = greens[i]

        # HDI bar
        ax.plot(
            [hdi_low[i], hdi_high[i]], [y_pos[i], y_pos[i]],
            color=color, linewidth=3.5, solid_capstyle="round", zorder=3,
        )
        # Point estimate
        ax.plot(
            ates[i], y_pos[i], "o",
            color=color, markersize=8, markeredgecolor="white",
            markeredgewidth=1.5, zorder=4,
        )

    # Mark final model distinctly
    ax.plot(
        ates[-1], y_pos[-1], "D",
        color=COLORS["kept"], markersize=9, markeredgecolor="white",
        markeredgewidth=1.5, zorder=5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("ATE (cognitive test score points)", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Numeric values on the right edge of the forest plot
    for i in range(n):
        ax.text(
            ax.get_xlim()[1] + 0.005, y_pos[i],
            f"{ates[i]:.2f}  [{hdi_low[i]:.2f}, {hdi_high[i]:.2f}]",
            fontsize=8.5, va="center", ha="left", fontfamily="monospace",
            color="#555",
        )

    # Widen x-axis to make room for the numeric labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.05, xlim[1] + 0.02)

    # ---- Right panel: HDI width (precision) ----
    ax2 = ax_width
    ax2.set_title("HDI Width", fontweight="bold", pad=12, loc="left")

    # Horizontal bars
    bar_colors = [greens[i] for i in range(n)]
    ax2.barh(
        y_pos, hdi_widths, height=0.55,
        color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3,
    )
    # Mark final distinctly
    ax2.barh(
        y_pos[-1], hdi_widths[-1], height=0.55,
        color=COLORS["kept"], edgecolor="white", linewidth=0.5, zorder=4,
    )

    # Value labels
    for i in range(n):
        ax2.text(
            hdi_widths[i] + 0.003, y_pos[i],
            f"{hdi_widths[i]:.3f}",
            fontsize=9, va="center", ha="left",
            fontweight="bold" if i == n - 1 else "normal",
            color=COLORS["kept"] if i == n - 1 else "#666",
        )

    # Improvement annotation — draw arrow from first to last bar
    pct_improvement = (1 - hdi_widths[-1] / hdi_widths[0]) * 100
    ax2.annotate(
        f"{pct_improvement:.0f}% narrower",
        xy=(hdi_widths[-1], y_pos[-1]),
        xytext=(hdi_widths[0] + 0.04, y_pos[0]),
        fontsize=10, fontweight="bold", color=COLORS["kept"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["kept"],
                        lw=1.5, connectionstyle="arc3,rad=0.3"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["kept"], alpha=0.9),
    )

    ax2.set_xlabel("Width (narrower = more precise)", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax2.set_xlim(0, max(hdi_widths) * 1.25)

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Final model diagram
# ---------------------------------------------------------------------------

MODEL_COMPONENTS: list[dict[str, str]] = [
    {"name": "alpha",      "dims": "1",  "prior": "N(0, 3)",
     "role": "Intercept",                            "group": "confounder"},
    {"name": "beta_t",     "dims": "1",  "prior": "N(0, 2)",
     "role": "Treatment effect (ATE base term)",     "group": "treatment"},
    {"name": "beta_x",     "dims": "25", "prior": "N(0, 1)",
     "role": "Linear confounder adjustment (x1-x25)","group": "confounder"},
    {"name": "beta_tx",    "dims": "6",  "prior": "N(0, 0.7)",
     "role": "Treatment x continuous (x1-x6)",       "group": "interaction"},
    {"name": "beta_sq",    "dims": "6",  "prior": "N(0, 0.5)",
     "role": "Quadratic confounders (x1-x6)^2",      "group": "confounder"},
    {"name": "beta_tx_sq", "dims": "6",  "prior": "N(0, 0.3)",
     "role": "Treatment x quadratic (x1-x6)^2",      "group": "interaction"},
    {"name": "sigma",      "dims": "1",  "prior": "HN(2)",
     "role": "Observation noise",                     "group": "noise"},
]

GROUP_COLORS = {
    "confounder":  COLORS["confounder"],
    "treatment":   COLORS["treatment"],
    "interaction": COLORS["interaction"],
    "noise":       COLORS["noise"],
}


def plot_model_diagram(output_path: Path) -> None:
    """Create a clean diagram of the final model architecture."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --- Title ---
    ax.text(
        5, 7.55,
        "IHDP Final Model Architecture",
        fontsize=16, fontweight="bold", ha="center", va="center",
        color="#222",
    )
    ax.text(
        5, 7.2,
        "Bayesian linear model with targeted non-linear confounder adjustment",
        fontsize=10, ha="center", va="center", color="#666",
    )

    # --- Equation box ---
    eq_box = mpatches.FancyBboxPatch(
        (0.5, 6.2), 9, 0.75, boxstyle="round,pad=0.15",
        facecolor="#f0f4ff", edgecolor=COLORS["confounder"], linewidth=1.2,
    )
    ax.add_patch(eq_box)
    ax.text(
        5, 6.7,
        (r"$\mu_i = \alpha"
         r" + \beta_t \, t_i"
         r" + \mathbf{x}_i^\top \boldsymbol{\beta}_x"
         r" + \mathbf{x}_{c,i}^{2\top} \boldsymbol{\beta}_{sq}"
         r" + t_i \left(\mathbf{x}_{c,i}^\top \boldsymbol{\beta}_{tx}"
         r" + \mathbf{x}_{c,i}^{2\top} \boldsymbol{\beta}_{tx,sq}\right)$"),
        fontsize=11, ha="center", va="center", color="#333",
    )
    ax.text(
        5, 6.35,
        r"$y_i \sim \mathrm{Normal}(\mu_i,\; \sigma)$",
        fontsize=10.5, ha="center", va="center", color="#555",
    )

    # --- Parameter table ---
    table_top = 5.85
    row_h = 0.48
    col_positions = [0.8, 2.6, 3.4, 4.5]
    headers = ["Parameter", "Dim", "Prior", "Role"]

    # Header background
    hdr_bg = mpatches.FancyBboxPatch(
        (0.5, table_top - 0.08), 9, 0.42,
        boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="#ccc", linewidth=0.6,
    )
    ax.add_patch(hdr_bg)
    for hdr, cx in zip(headers, col_positions):
        ax.text(
            cx, table_top + 0.12, hdr,
            fontsize=10, fontweight="bold", va="center", color="#333",
        )

    # Data rows
    for i, comp in enumerate(MODEL_COMPONENTS):
        y = table_top - (i + 1) * row_h
        gc = GROUP_COLORS[comp["group"]]

        # Alternating subtle background
        if i % 2 == 0:
            row_bg = mpatches.FancyBboxPatch(
                (0.5, y - 0.1), 9, row_h - 0.04,
                boxstyle="round,pad=0.02",
                facecolor="#fafafa", edgecolor="none",
            )
            ax.add_patch(row_bg)

        # Color indicator stripe
        stripe = mpatches.FancyBboxPatch(
            (0.5, y - 0.1), 0.1, row_h - 0.04,
            boxstyle="round,pad=0.01",
            facecolor=gc, edgecolor="none",
        )
        ax.add_patch(stripe)

        ax.text(col_positions[0], y + 0.1, comp["name"],
                fontsize=10, fontfamily="monospace", fontweight="bold",
                va="center", color=gc)
        ax.text(col_positions[1], y + 0.1, comp["dims"],
                fontsize=10, va="center", color="#555")
        ax.text(col_positions[2], y + 0.1, comp["prior"],
                fontsize=10, fontfamily="monospace", va="center", color="#666")
        ax.text(col_positions[3], y + 0.1, comp["role"],
                fontsize=9, va="center", color="#444")

    # --- Results summary box ---
    rx, ry = 0.7, 0.3
    rw, rh = 4.2, 1.35
    results_bg = mpatches.FancyBboxPatch(
        (rx, ry), rw, rh, boxstyle="round,pad=0.15",
        facecolor="#e8f5e9", edgecolor=COLORS["kept"], linewidth=1.2,
    )
    ax.add_patch(results_bg)
    ax.text(
        rx + rw / 2, ry + rh - 0.18, "Key Results",
        fontsize=11, fontweight="bold", ha="center", va="center",
        color=COLORS["kept"],
    )

    result_items = [
        ("ELPD (PSIS-LOO):", "-865.1  (SE = 17.0)"),
        ("ATE estimate:",     "3.97  (true ~4.0, bias = 0.05)"),
        ("ATE 94% HDI:",     "[3.74,  4.20]  (width = 0.46)"),
        ("Convergence:",     "r_hat = 1.00,  ESS = 754,  0 div."),
    ]
    for j, (key, val) in enumerate(result_items):
        yy = ry + rh - 0.48 - j * 0.25
        ax.text(rx + 0.2, yy, key, fontsize=9, fontweight="bold",
                va="center", color="#444")
        ax.text(rx + 1.8, yy, val, fontsize=9, va="center", color="#333")

    # --- Legend box ---
    lx, ly = 5.8, 0.3
    lw, lh = 3.7, 1.35
    legend_bg = mpatches.FancyBboxPatch(
        (lx, ly), lw, lh, boxstyle="round,pad=0.15",
        facecolor=COLORS["bg_box"], edgecolor="#ccc", linewidth=0.8,
    )
    ax.add_patch(legend_bg)
    ax.text(
        lx + lw / 2, ly + lh - 0.18, "Design Insight",
        fontsize=11, fontweight="bold", ha="center", va="center",
        color="#555",
    )
    insight_lines = [
        "Restricting treatment interactions to the",
        "6 continuous confounders (removing 19 noisy",
        "binary interactions) was the key structural",
        "change: fewer parameters, better ELPD.",
    ]
    for j, line in enumerate(insight_lines):
        ax.text(
            lx + 0.2, ly + lh - 0.48 - j * 0.22, line,
            fontsize=8.5, va="center", color="#555", fontstyle="italic",
        )

    # --- Color legend (two rows) ---
    legend_items = [
        (COLORS["confounder"],  "Confounder adjustment"),
        (COLORS["treatment"],   "Treatment effect"),
        (COLORS["interaction"], "Tx x confounder interaction"),
        (COLORS["noise"],       "Noise"),
    ]
    for k, (color, label) in enumerate(legend_items):
        col = k % 2
        row = k // 2
        cx = 0.7 + col * 3.0
        cy = 1.95 - row * 0.25

        dot = mpatches.Circle(
            (cx, cy), 0.06, facecolor=color, edgecolor="none",
        )
        ax.add_patch(dot)
        ax.text(cx + 0.15, cy, label, fontsize=8.5, va="center", color="#555")

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all three figures."""
    _set_pub_style()
    base_dir = Path(__file__).parent

    rows = parse_results(base_dir / "results.tsv")
    print(f"Parsed {len(rows)} experiments")

    plot_iterations(rows, base_dir / "iterations.png")
    plot_causal_estimates(base_dir / "causal_estimates.png")
    plot_model_diagram(base_dir / "model_diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
