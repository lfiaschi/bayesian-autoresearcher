"""Generate publication-quality visualizations for NHEFS experiment results.

Produces three figures:
  1. iterations.png — Two-panel optimization journey (trajectory + contribution waterfall)
  2. causal_estimates.png — Forest plot of ATE estimates with HDI intervals
  3. model_diagram.png — Final model architecture with equation and results
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Shared utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plotting import COLORS, parse_results, plot_iterations, set_pub_style


# ---------------------------------------------------------------------------
# NHEFS-specific: short labels for kept milestones (index → label)
# ---------------------------------------------------------------------------

KEPT_LABELS: dict[int, str] = {
    1:  "Baseline (Normal)",
    2:  "+ Quadratics + interactions",
    3:  "Student-t likelihood",
    5:  "Heteroscedastic sigma",
    7:  "Remove interactions",
    8:  "Remove quadratics",
    18: "Tighter priors",
    20: "wt71² in sigma (final)",
}


# ---------------------------------------------------------------------------
# Causal estimation data (recorded from experiment runs)
# ---------------------------------------------------------------------------

CAUSAL_DATA: list[tuple[int, str, float, float]] = [
    (1,  "Baseline (Normal)",             3.627, 2.266),
    (2,  "+ Quadratics + interactions",   3.603, 2.250),
    (3,  "Student-t likelihood",          2.994, 2.094),
    (5,  "Heteroscedastic sigma",         3.048, 2.040),
    (7,  "Remove interactions",           3.087, 2.010),
    (8,  "Remove quadratics",             2.970, 1.992),
    (18, "Tighter priors",                2.934, 1.990),
    (20, "wt71² in sigma (final)",        2.931, 1.956),
]

# No ground truth — literature range is 3-5 kg
LITERATURE_ATE_LOW = 3.0
LITERATURE_ATE_HIGH = 5.0


# ---------------------------------------------------------------------------
# Figure 2: ATE forest plot
# ---------------------------------------------------------------------------

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
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.28},
        sharey=True,
    )

    # ---- Left panel: Forest plot ----
    ax = ax_forest
    ax.set_title(
        "Average Treatment Effect on Weight Change: Point Estimates with 94% HDI",
        fontweight="bold", pad=12, loc="left",
    )

    # Literature reference band (3-5 kg)
    ax.axvspan(LITERATURE_ATE_LOW, LITERATURE_ATE_HIGH,
               color=COLORS["crashed"], alpha=0.06, zorder=1)
    ax.axvline(LITERATURE_ATE_LOW, color=COLORS["crashed"], linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=2)
    ax.axvline(LITERATURE_ATE_HIGH, color=COLORS["crashed"], linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=2)
    ax.text((LITERATURE_ATE_LOW + LITERATURE_ATE_HIGH) / 2, -0.6,
            "Literature range\n(3–5 kg)",
            fontsize=9, color=COLORS["crashed"], va="center", ha="center",
            fontweight="bold")

    greens = plt.cm.Greens(np.linspace(0.3, 0.85, n))

    for i in range(n):
        color = greens[i]
        ax.plot(
            [hdi_low[i], hdi_high[i]], [y_pos[i], y_pos[i]],
            color=color, linewidth=3.5, solid_capstyle="round", zorder=3,
        )
        ax.plot(
            ates[i], y_pos[i], "o",
            color=color, markersize=8, markeredgecolor="white",
            markeredgewidth=1.5, zorder=4,
        )

    # Final model diamond
    ax.plot(
        ates[-1], y_pos[-1], "D",
        color=COLORS["kept"], markersize=9, markeredgecolor="white",
        markeredgewidth=1.5, zorder=5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("ATE (kg weight change from quitting smoking)", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Numeric values
    for i in range(n):
        ax.text(
            ax.get_xlim()[1] + 0.02, y_pos[i],
            f"{ates[i]:.2f}  [{hdi_low[i]:.2f}, {hdi_high[i]:.2f}]",
            fontsize=8.5, va="center", ha="left", fontfamily="monospace",
            color="#555",
        )

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1, xlim[1] + 0.02)

    # ---- Right panel: HDI width ----
    ax2 = ax_width
    ax2.set_title("HDI Width", fontweight="bold", pad=12, loc="left")

    bar_colors = [greens[i] for i in range(n)]
    ax2.barh(
        y_pos, hdi_widths, height=0.55,
        color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax2.barh(
        y_pos[-1], hdi_widths[-1], height=0.55,
        color=COLORS["kept"], edgecolor="white", linewidth=0.5, zorder=4,
    )

    for i in range(n):
        ax2.text(
            hdi_widths[i] + 0.01, y_pos[i],
            f"{hdi_widths[i]:.3f}",
            fontsize=9, va="center", ha="left",
            fontweight="bold" if i == n - 1 else "normal",
            color=COLORS["kept"] if i == n - 1 else "#666",
        )

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

    ax2.set_xlabel("Width in kg (narrower = more precise)", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax2.set_xlim(0, max(hdi_widths) * 1.2)

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Final model diagram
# ---------------------------------------------------------------------------

MODEL_COMPONENTS: list[dict[str, str]] = [
    {"name": "alpha",              "dims": "1",  "prior": "N(0, 5)",
     "role": "Intercept (weight change in kg)",  "group": "confounder"},
    {"name": "beta_t",             "dims": "1",  "prior": "N(0, 3)",
     "role": "Treatment effect (quitting)",      "group": "treatment"},
    {"name": "beta_x",             "dims": "9",  "prior": "N(0, 2)",
     "role": "Confounder adjustment",            "group": "confounder"},
    {"name": "nu",                 "dims": "1",  "prior": "Ga(2, 0.1)",
     "role": "Student-t degrees of freedom",     "group": "noise"},
    {"name": "log_sigma_int",      "dims": "1",  "prior": "N(2, 0.3)",
     "role": "Baseline log-noise",               "group": "noise"},
    {"name": "log_sigma_coeffs",   "dims": "2",  "prior": "N(0, 0.3)",
     "role": "Sigma ~ smokeintensity, wt71",     "group": "noise"},
    {"name": "log_sigma_wt71sq",   "dims": "1",  "prior": "N(0, 0.2)",
     "role": "Non-linear wt71 effect on sigma",  "group": "noise"},
]

GROUP_COLORS: dict[str, str] = {
    "confounder":  COLORS["confounder"],
    "treatment":   COLORS["treatment"],
    "noise":       COLORS["noise"],
}


def plot_model_diagram(output_path: Path) -> None:
    """Create a clean diagram of the final model architecture."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Title
    ax.text(5, 7.1, "NHEFS Final Model Architecture",
            fontsize=16, fontweight="bold", ha="center", va="center", color="#222")
    ax.text(5, 6.8, "Student-t linear model with confounder-dependent heteroscedasticity",
            fontsize=10, ha="center", va="center", color="#666")

    # Equation box
    eq_box = mpatches.FancyBboxPatch(
        (0.5, 5.7), 9, 0.85, boxstyle="round,pad=0.15",
        facecolor="#f0f4ff", edgecolor=COLORS["confounder"], linewidth=1.2)
    ax.add_patch(eq_box)
    ax.text(5, 6.25,
            r"$\mu_i = \alpha + \beta_t \, t_i + \mathbf{x}_i^\top \boldsymbol{\beta}_x$",
            fontsize=12, ha="center", va="center", color="#333")
    ax.text(5, 5.95,
            r"$\log\sigma_i = \gamma_0 + \gamma_1 \, x_{\mathrm{smoke},i}"
            r" + \gamma_2 \, x_{\mathrm{wt71},i}"
            r" + \gamma_3 \, x_{\mathrm{wt71},i}^2$",
            fontsize=10.5, ha="center", va="center", color="#555")
    ax.text(5, 5.8,
            r"$y_i \sim \mathrm{StudentT}(\nu,\; \mu_i,\; \sigma_i)$",
            fontsize=10.5, ha="center", va="center", color="#555")

    # Parameter table
    table_top = 5.35
    row_h = 0.48
    col_positions = [0.8, 2.8, 3.6, 5.0]
    headers = ["Parameter", "Dim", "Prior", "Role"]

    hdr_bg = mpatches.FancyBboxPatch(
        (0.5, table_top - 0.08), 9, 0.42, boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="#ccc", linewidth=0.6)
    ax.add_patch(hdr_bg)
    for hdr, cx in zip(headers, col_positions):
        ax.text(cx, table_top + 0.12, hdr, fontsize=10, fontweight="bold",
                va="center", color="#333")

    for i, comp in enumerate(MODEL_COMPONENTS):
        y = table_top - (i + 1) * row_h
        gc = GROUP_COLORS[comp["group"]]
        if i % 2 == 0:
            row_bg = mpatches.FancyBboxPatch(
                (0.5, y - 0.1), 9, row_h - 0.04, boxstyle="round,pad=0.02",
                facecolor="#fafafa", edgecolor="none")
            ax.add_patch(row_bg)
        stripe = mpatches.FancyBboxPatch(
            (0.5, y - 0.1), 0.1, row_h - 0.04, boxstyle="round,pad=0.01",
            facecolor=gc, edgecolor="none")
        ax.add_patch(stripe)

        ax.text(col_positions[0], y + 0.1, comp["name"],
                fontsize=9.5, fontfamily="monospace", fontweight="bold",
                va="center", color=gc)
        ax.text(col_positions[1], y + 0.1, comp["dims"],
                fontsize=10, va="center", color="#555")
        ax.text(col_positions[2], y + 0.1, comp["prior"],
                fontsize=10, fontfamily="monospace", va="center", color="#666")
        ax.text(col_positions[3], y + 0.1, comp["role"],
                fontsize=9, va="center", color="#444")

    # Results box
    rx, ry = 0.7, 0.3
    rw, rh = 4.2, 1.5
    results_bg = mpatches.FancyBboxPatch(
        (rx, ry), rw, rh, boxstyle="round,pad=0.15",
        facecolor="#e8f5e9", edgecolor=COLORS["kept"], linewidth=1.2)
    ax.add_patch(results_bg)
    ax.text(rx + rw / 2, ry + rh - 0.18, "Key Results",
            fontsize=11, fontweight="bold", ha="center", va="center",
            color=COLORS["kept"])

    result_items = [
        ("ELPD (PSIS-LOO):", "-4182.4  (SE = 34.8)"),
        ("ATE estimate:",     "2.93 kg  (literature 3-5 kg)"),
        ("ATE 94% HDI:",     "[1.95,  3.90]  (width = 1.96 kg)"),
        ("Convergence:",     "r_hat = 1.00,  ESS > 1500,  0 div."),
        ("ELPD gain:",       "+120 from baseline"),
    ]
    for j, (key, val) in enumerate(result_items):
        yy = ry + rh - 0.45 - j * 0.23
        ax.text(rx + 0.2, yy, key, fontsize=9, fontweight="bold",
                va="center", color="#444")
        ax.text(rx + 1.7, yy, val, fontsize=9, va="center", color="#333")

    # Design insight box
    lx, ly = 5.8, 0.3
    lw, lh = 3.7, 1.5
    legend_bg = mpatches.FancyBboxPatch(
        (lx, ly), lw, lh, boxstyle="round,pad=0.15",
        facecolor=COLORS["bg_box"], edgecolor="#ccc", linewidth=0.8)
    ax.add_patch(legend_bg)
    ax.text(lx + lw / 2, ly + lh - 0.18, "Design Insight",
            fontsize=11, fontweight="bold", ha="center", va="center", color="#555")

    insight_lines = [
        "Student-t likelihood was the key breakthrough",
        "(+88 ELPD), robustly handling outlier weight",
        "changes. Confounder-dependent heteroscedasticity",
        "added another +31 ELPD. Simplification (removing",
        "quadratics and interactions) preserved gains while",
        "narrowing HDI from 2.27 to 1.96 kg.",
    ]
    for j, line in enumerate(insight_lines):
        ax.text(lx + 0.2, ly + lh - 0.45 - j * 0.19, line,
                fontsize=8.2, va="center", color="#555", fontstyle="italic")

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all three figures."""
    set_pub_style()
    base_dir = Path(__file__).parent

    rows = parse_results(base_dir / "results.tsv")
    print(f"Parsed {len(rows)} experiments")

    plot_iterations(
        rows,
        base_dir / "iterations.png",
        kept_labels=KEPT_LABELS,
        title="NHEFS: ELPD Optimization Across 20 Model Iterations",
    )
    plot_causal_estimates(base_dir / "causal_estimates.png")
    plot_model_diagram(base_dir / "model_diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
