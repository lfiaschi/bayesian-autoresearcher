"""Generate publication-quality visualizations for Twins experiment results.

Produces three figures:
  1. iterations.png — Two-panel optimization journey (trajectory + contribution waterfall)
  2. causal_estimates.png — Forest plot of ATE estimates with HDI intervals
  3. model_diagram.png — Final model architecture with equation and results
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Shared utilities — style, colors, results parsing, trajectory plot
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plotting import COLORS, parse_results, plot_iterations, set_pub_style


# ---------------------------------------------------------------------------
# Twins-specific: short labels for kept milestones (index → label)
# ---------------------------------------------------------------------------

KEPT_LABELS: dict[int, str] = {
    1:  "Baseline",
    3:  "Tighter priors",
    5:  "+ quadratic X",
    8:  "+ cubic X",
    10: "+ pairwise interactions",
    11: "Heteroscedastic sigma",
    16: "+ gestat10 poly(4,5)",
    22: "log(sigma) ~ gestat10",
    26: "+ treatment in sigma",
}


# ---------------------------------------------------------------------------
# Twins-specific: Causal estimation data (recorded from experiment runs)
# ---------------------------------------------------------------------------

# (index, label, ate_estimate, ate_hdi_width)
# HDI bounds approximated as ATE +/- HDI_width/2 (posterior is ~symmetric)
CAUSAL_DATA: list[tuple[int, str, float, float]] = [
    (1,  "Baseline linear",              -0.0056, 0.0242),
    (3,  "Tighter priors",               -0.0058, 0.0187),
    (5,  "+ Quadratic confounders",       -0.0065, 0.0183),
    (8,  "+ Cubic confounders",           -0.0074, 0.0177),
    (10, "+ Pairwise interactions",       -0.0076, 0.0174),
    (11, "Heteroscedastic sigma",         -0.0074, 0.0165),
    (16, "+ gestat10 poly(4,5)",          -0.0078, 0.0169),
    (22, "log(sigma) ~ gestat10",         -0.0062, 0.0116),
    (26, "+ treatment in sigma (final)",  -0.0063, 0.0113),
]

TRUE_ATE = -0.006476


# ---------------------------------------------------------------------------
# Figure 2: ATE forest plot (Twins-specific)
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

    ax.axvline(TRUE_ATE, color=COLORS["crashed"], linewidth=1.5,
               linestyle="-", alpha=0.8, zorder=2)
    ax.axvspan(TRUE_ATE - 0.0005, TRUE_ATE + 0.0005, color=COLORS["crashed"],
               alpha=0.06, zorder=1)
    ax.text(TRUE_ATE + 0.0005, -0.6, f"True ATE = {TRUE_ATE:.4f}",
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

    ax.plot(
        ates[-1], y_pos[-1], "D",
        color=COLORS["kept"], markersize=9, markeredgecolor="white",
        markeredgewidth=1.5, zorder=5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("ATE (mortality probability difference)", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(n):
        ax.text(
            ax.get_xlim()[1] + 0.0002, y_pos[i],
            f"{ates[i]:.4f}  [{hdi_low[i]:.4f}, {hdi_high[i]:.4f}]",
            fontsize=8.5, va="center", ha="left", fontfamily="monospace",
            color="#555",
        )

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.002, xlim[1] + 0.001)

    # ---- Right panel: HDI width (precision) ----
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
            hdi_widths[i] + 0.0003, y_pos[i],
            f"{hdi_widths[i]:.4f}",
            fontsize=9, va="center", ha="left",
            fontweight="bold" if i == n - 1 else "normal",
            color=COLORS["kept"] if i == n - 1 else "#666",
        )

    pct_improvement = (1 - hdi_widths[-1] / hdi_widths[0]) * 100
    ax2.annotate(
        f"{pct_improvement:.0f}% narrower",
        xy=(hdi_widths[-1], y_pos[-1]),
        xytext=(hdi_widths[0] + 0.003, y_pos[0]),
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
# Figure 3: Final model diagram (Twins-specific)
# ---------------------------------------------------------------------------

MODEL_COMPONENTS: list[dict[str, str]] = [
    {"name": "alpha",            "dims": "1",  "prior": "N(0, 0.5)",
     "role": "Intercept",                                "group": "confounder"},
    {"name": "beta_t",           "dims": "1",  "prior": "N(0, 0.3)",
     "role": "Treatment effect (ATE)",                   "group": "treatment"},
    {"name": "beta_x",           "dims": "10", "prior": "N(0, 0.3)",
     "role": "Linear confounders (10 features)",         "group": "confounder"},
    {"name": "beta_sq",          "dims": "3",  "prior": "N(0, 0.2)",
     "role": "Quadratic (mager8, mrace, gestat10)",      "group": "confounder"},
    {"name": "beta_cb",          "dims": "3",  "prior": "N(0, 0.1)",
     "role": "Cubic (mager8, mrace, gestat10)",          "group": "confounder"},
    {"name": "beta_pair",        "dims": "3",  "prior": "N(0, 0.15)",
     "role": "Pairwise confounder interactions",         "group": "interaction"},
    {"name": "beta_g4",          "dims": "1",  "prior": "N(0, 0.05)",
     "role": "gestat10^4 (quartic)",                     "group": "confounder"},
    {"name": "beta_g5",          "dims": "1",  "prior": "N(0, 0.02)",
     "role": "gestat10^5 (quintic)",                     "group": "confounder"},
    {"name": "log_sigma_0",      "dims": "1",  "prior": "N(-1.5, 0.5)",
     "role": "Base log-noise",                           "group": "noise"},
    {"name": "log_sigma_gestat", "dims": "1",  "prior": "N(0, 0.3)",
     "role": "Noise ~ gestational age",                  "group": "noise"},
    {"name": "log_sigma_tx",     "dims": "1",  "prior": "N(0, 0.2)",
     "role": "Noise ~ treatment group",                  "group": "noise"},
]

GROUP_COLORS: dict[str, str] = {
    "confounder":  COLORS["confounder"],
    "treatment":   COLORS["treatment"],
    "interaction": COLORS["interaction"],
    "noise":       COLORS["noise"],
}


def plot_model_diagram(output_path: Path) -> None:
    """Create a clean diagram of the final model architecture."""
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        5, 9.55,
        "Twins Final Model Architecture",
        fontsize=16, fontweight="bold", ha="center", va="center",
        color="#222",
    )
    ax.text(
        5, 9.2,
        "Heteroscedastic linear probability model with polynomial confounder expansion",
        fontsize=10, ha="center", va="center", color="#666",
    )

    # Mean equation box
    eq_box = mpatches.FancyBboxPatch(
        (0.5, 8.0), 9, 0.9, boxstyle="round,pad=0.15",
        facecolor="#f0f4ff", edgecolor=COLORS["confounder"], linewidth=1.2,
    )
    ax.add_patch(eq_box)
    ax.text(
        5, 8.6,
        (r"$\mu_i = \alpha"
         r" + \beta_t \, t_i"
         r" + \mathbf{x}_i^\top \boldsymbol{\beta}_x"
         r" + \mathbf{x}_{c,i}^{2\top} \boldsymbol{\beta}_{sq}"
         r" + \mathbf{x}_{c,i}^{3\top} \boldsymbol{\beta}_{cb}"
         r" + \mathbf{x}_{pair}^\top \boldsymbol{\beta}_{pair}"
         r" + \beta_{g4}\, g_i^4"
         r" + \beta_{g5}\, g_i^5$"),
        fontsize=9.5, ha="center", va="center", color="#333",
    )
    ax.text(
        5, 8.2,
        (r"$\log\sigma_i = \gamma_0 + \gamma_g \, g_i + \gamma_t \, t_i$"
         r"$\qquad\qquad y_i \sim \mathrm{Normal}(\mu_i,\; \exp(\log\sigma_i))$"),
        fontsize=10, ha="center", va="center", color="#555",
    )

    # Parameter table
    table_top = 7.65
    row_h = 0.42
    col_positions = [0.8, 2.8, 3.6, 4.7]
    headers = ["Parameter", "Dim", "Prior", "Role"]

    hdr_bg = mpatches.FancyBboxPatch(
        (0.5, table_top - 0.08), 9, 0.38,
        boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="#ccc", linewidth=0.6,
    )
    ax.add_patch(hdr_bg)
    for hdr, cx in zip(headers, col_positions):
        ax.text(
            cx, table_top + 0.08, hdr,
            fontsize=10, fontweight="bold", va="center", color="#333",
        )

    for i, comp in enumerate(MODEL_COMPONENTS):
        y = table_top - (i + 1) * row_h
        gc = GROUP_COLORS[comp["group"]]

        if i % 2 == 0:
            row_bg = mpatches.FancyBboxPatch(
                (0.5, y - 0.08), 9, row_h - 0.04,
                boxstyle="round,pad=0.02",
                facecolor="#fafafa", edgecolor="none",
            )
            ax.add_patch(row_bg)

        stripe = mpatches.FancyBboxPatch(
            (0.5, y - 0.08), 0.1, row_h - 0.04,
            boxstyle="round,pad=0.01",
            facecolor=gc, edgecolor="none",
        )
        ax.add_patch(stripe)

        ax.text(col_positions[0], y + 0.08, comp["name"],
                fontsize=9, fontfamily="monospace", fontweight="bold",
                va="center", color=gc)
        ax.text(col_positions[1], y + 0.08, comp["dims"],
                fontsize=10, va="center", color="#555")
        ax.text(col_positions[2], y + 0.08, comp["prior"],
                fontsize=9.5, fontfamily="monospace", va="center", color="#666")
        ax.text(col_positions[3], y + 0.08, comp["role"],
                fontsize=8.5, va="center", color="#444")

    # Results box
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
        ("ELPD (PSIS-LOO):", "3139  (SE = 399)"),
        ("ATE estimate:",     "-0.0063  (true -0.0065, bias = 0.0001)"),
        ("ATE 94% HDI:",     "[-0.012,  -0.001]  (width = 0.011)"),
        ("Convergence:",     "r_hat = 1.00,  0 divergences"),
    ]
    for j, (key, val) in enumerate(result_items):
        yy = ry + rh - 0.48 - j * 0.25
        ax.text(rx + 0.2, yy, key, fontsize=9, fontweight="bold",
                va="center", color="#444")
        ax.text(rx + 1.8, yy, val, fontsize=9, va="center", color="#333")

    # Design insight box
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
        "Confounder-dependent heteroscedasticity was",
        "the single biggest improvement (+951 ELPD).",
        "Letting sigma vary with gestational age",
        "captures the biology: premature babies have",
        "fundamentally different outcome variance.",
    ]
    for j, line in enumerate(insight_lines):
        ax.text(
            lx + 0.2, ly + lh - 0.42 - j * 0.20, line,
            fontsize=8.5, va="center", color="#555", fontstyle="italic",
        )

    # Color legend
    legend_items = [
        (COLORS["confounder"],  "Confounder adjustment"),
        (COLORS["treatment"],   "Treatment effect"),
        (COLORS["interaction"], "Confounder interaction"),
        (COLORS["noise"],       "Heteroscedastic noise"),
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
    set_pub_style()
    base_dir = Path(__file__).parent

    rows = parse_results(base_dir / "results.tsv")
    print(f"Parsed {len(rows)} experiments")

    plot_iterations(
        rows,
        base_dir / "iterations.png",
        kept_labels=KEPT_LABELS,
        title="ELPD Optimization Across 29 Bayesian Model Iterations (Twins)",
    )
    plot_causal_estimates(base_dir / "causal_estimates.png")
    plot_model_diagram(base_dir / "model_diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
