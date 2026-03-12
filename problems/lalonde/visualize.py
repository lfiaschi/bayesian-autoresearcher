"""Generate publication-quality visualizations for LaLonde experiment results.

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
# LaLonde-specific: short labels for kept milestones (index → label)
# ---------------------------------------------------------------------------

KEPT_LABELS: dict[int, str] = {
    1:  "Baseline (Normal)",
    2:  "+ Quadratics + interactions",
    3:  "Student-t likelihood",
    9:  "Gamma GLM (log-link)",
    10: "Tighter Gamma priors",
    11: "Remove interactions",
    13: "Tune alpha + beta_t",
    15: "Remove quadratics",
    16: "Tighter beta_x (final)",
}


# ---------------------------------------------------------------------------
# Causal estimation data (recorded from experiment runs)
# ---------------------------------------------------------------------------

CAUSAL_DATA: list[tuple[int, str, float, float]] = [
    (1,  "Baseline (Normal)",          597.5, 2376.7),
    (2,  "+ Quadratics + interactions", 531.8, 1876.4),
    (3,  "Student-t likelihood",        162.1, 1615.9),
    (9,  "Gamma GLM (log-link)",       2982.9, 11168.3),
    (10, "Tighter Gamma priors",       1019.0, 7125.3),
    (11, "Remove interactions",         588.2, 4782.4),
    (13, "Tune alpha + beta_t",         677.4, 4661.7),
    (15, "Remove quadratics",           696.0, 4269.2),
    (16, "Tighter beta_x (final)",      697.0, 4284.0),
]

TRUE_ATE = 886.0  # LaLonde (1986) experimental estimate


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
        "Average Treatment Effect on 1978 Earnings: Point Estimates with 94% HDI",
        fontweight="bold", pad=12, loc="left",
    )

    # Truth reference
    ax.axvline(TRUE_ATE, color=COLORS["crashed"], linewidth=1.5,
               linestyle="-", alpha=0.8, zorder=2)
    ax.axvspan(TRUE_ATE - 20, TRUE_ATE + 20, color=COLORS["crashed"],
               alpha=0.06, zorder=1)
    ax.text(TRUE_ATE + 50, -0.6, f"True ATE = ${TRUE_ATE:.0f}",
            fontsize=9.5, color=COLORS["crashed"], va="center", ha="left",
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
    ax.set_xlabel("ATE (USD, 1978 earnings)", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Numeric values
    for i in range(n):
        ax.text(
            ax.get_xlim()[1] + 50, y_pos[i],
            f"${ates[i]:,.0f}",
            fontsize=9, va="center", ha="left", fontfamily="monospace",
            color="#555",
        )

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 100, xlim[1] + 50)

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
            hdi_widths[i] + 100, y_pos[i],
            f"${hdi_widths[i]:,.0f}",
            fontsize=8.5, va="center", ha="left",
            fontweight="bold" if i == n - 1 else "normal",
            color=COLORS["kept"] if i == n - 1 else "#666",
        )

    ax2.set_xlabel("Width in USD (narrower = more precise)", fontsize=10)
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
    {"name": "alpha",      "dims": "1",  "prior": "N(log 5000, 1)",
     "role": "Log-scale intercept",              "group": "confounder"},
    {"name": "beta_t",     "dims": "1",  "prior": "N(0, 0.4)",
     "role": "Treatment effect (log scale)",     "group": "treatment"},
    {"name": "beta_x",     "dims": "7",  "prior": "N(0, 0.3)",
     "role": "Confounder adjustment (log scale)","group": "confounder"},
    {"name": "phi",        "dims": "1",  "prior": "HN(5)",
     "role": "Gamma shape (dispersion)",         "group": "noise"},
]

GROUP_COLORS: dict[str, str] = {
    "confounder":  COLORS["confounder"],
    "treatment":   COLORS["treatment"],
    "noise":       COLORS["noise"],
}


def plot_model_diagram(output_path: Path) -> None:
    """Create a clean diagram of the final model architecture."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(5, 6.6, "LaLonde Final Model Architecture",
            fontsize=16, fontweight="bold", ha="center", va="center", color="#222")
    ax.text(5, 6.3, "Gamma GLM with log-link for non-negative earnings",
            fontsize=10, ha="center", va="center", color="#666")

    # Equation box
    eq_box = mpatches.FancyBboxPatch(
        (0.5, 5.25), 9, 0.8, boxstyle="round,pad=0.15",
        facecolor="#f0f4ff", edgecolor=COLORS["confounder"], linewidth=1.2)
    ax.add_patch(eq_box)
    ax.text(5, 5.75,
            r"$\log(\mu_i) = \alpha + \beta_t \, t_i + \mathbf{x}_i^\top \boldsymbol{\beta}_x$",
            fontsize=12, ha="center", va="center", color="#333")
    ax.text(5, 5.4,
            r"$y_i \sim \mathrm{Gamma}(\phi,\; \phi / \mu_i)$"
            r"$\qquad$"
            r"$\mathrm{where}\; y_i = \mathrm{re78}_i + 1$",
            fontsize=10.5, ha="center", va="center", color="#555")

    # Parameter table
    table_top = 4.9
    row_h = 0.52
    col_positions = [0.8, 2.6, 3.4, 5.0]
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
                fontsize=10, fontfamily="monospace", fontweight="bold",
                va="center", color=gc)
        ax.text(col_positions[1], y + 0.1, comp["dims"],
                fontsize=10, va="center", color="#555")
        ax.text(col_positions[2], y + 0.1, comp["prior"],
                fontsize=10, fontfamily="monospace", va="center", color="#666")
        ax.text(col_positions[3], y + 0.1, comp["role"],
                fontsize=9, va="center", color="#444")

    # Results box
    rx, ry = 0.7, 0.4
    rw, rh = 4.2, 1.6
    results_bg = mpatches.FancyBboxPatch(
        (rx, ry), rw, rh, boxstyle="round,pad=0.15",
        facecolor="#e8f5e9", edgecolor=COLORS["kept"], linewidth=1.2)
    ax.add_patch(results_bg)
    ax.text(rx + rw / 2, ry + rh - 0.18, "Key Results",
            fontsize=11, fontweight="bold", ha="center", va="center",
            color=COLORS["kept"])

    result_items = [
        ("ELPD (PSIS-LOO):", "-5044.8  (SE = 71.5)"),
        ("ATE estimate:",     "$697  (true ~$886, bias $189)"),
        ("ATE 94% HDI:",     "[-$1,445,  $2,839]"),
        ("Convergence:",     "r_hat = 1.00,  ESS > 1500,  0 div."),
        ("ELPD gain:",       "+822 from baseline"),
    ]
    for j, (key, val) in enumerate(result_items):
        yy = ry + rh - 0.48 - j * 0.25
        ax.text(rx + 0.2, yy, key, fontsize=9, fontweight="bold",
                va="center", color="#444")
        ax.text(rx + 1.7, yy, val, fontsize=9, va="center", color="#333")

    # Design insight box
    lx, ly = 5.8, 0.4
    lw, lh = 3.7, 1.6
    legend_bg = mpatches.FancyBboxPatch(
        (lx, ly), lw, lh, boxstyle="round,pad=0.15",
        facecolor=COLORS["bg_box"], edgecolor="#ccc", linewidth=0.8)
    ax.add_patch(legend_bg)
    ax.text(lx + lw / 2, ly + lh - 0.18, "Design Insight",
            fontsize=11, fontweight="bold", ha="center", va="center", color="#555")

    insight_lines = [
        "The Gamma likelihood with log-link was the",
        "key breakthrough (+822 ELPD). Earnings are",
        "non-negative and right-skewed — the Gamma",
        "handles this naturally. Student-t improved",
        "ELPD but destroyed ATE by down-weighting",
        "high earners that carry the treatment signal.",
    ]
    for j, line in enumerate(insight_lines):
        ax.text(lx + 0.2, ly + lh - 0.48 - j * 0.2, line,
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
        title="LaLonde: ELPD Optimization Across 16 Model Iterations",
    )
    plot_causal_estimates(base_dir / "causal_estimates.png")
    plot_model_diagram(base_dir / "model_diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
