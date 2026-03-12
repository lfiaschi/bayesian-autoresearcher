"""Generate publication-quality visualizations for IHDP experiment results.

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
# IHDP-specific: short labels for kept milestones (index → label)
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
# IHDP-specific: Causal estimation data (recorded from experiment runs)
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


# ---------------------------------------------------------------------------
# Figure 2: ATE forest plot (IHDP-specific)
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
    ax.axvspan(TRUE_ATE - 0.02, TRUE_ATE + 0.02, color=COLORS["crashed"],
               alpha=0.06, zorder=1)
    ax.text(TRUE_ATE + 0.015, -0.6, f"True ATE = {TRUE_ATE:.1f}",
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
    ax.set_xlabel("ATE (cognitive test score points)", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(n):
        ax.text(
            ax.get_xlim()[1] + 0.005, y_pos[i],
            f"{ates[i]:.2f}  [{hdi_low[i]:.2f}, {hdi_high[i]:.2f}]",
            fontsize=8.5, va="center", ha="left", fontfamily="monospace",
            color="#555",
        )

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.05, xlim[1] + 0.02)

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
            hdi_widths[i] + 0.003, y_pos[i],
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
# Figure 3: Final model diagram (IHDP-specific)
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

GROUP_COLORS: dict[str, str] = {
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

    table_top = 5.85
    row_h = 0.48
    col_positions = [0.8, 2.6, 3.4, 4.5]
    headers = ["Parameter", "Dim", "Prior", "Role"]

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

    for i, comp in enumerate(MODEL_COMPONENTS):
        y = table_top - (i + 1) * row_h
        gc = GROUP_COLORS[comp["group"]]

        if i % 2 == 0:
            row_bg = mpatches.FancyBboxPatch(
                (0.5, y - 0.1), 9, row_h - 0.04,
                boxstyle="round,pad=0.02",
                facecolor="#fafafa", edgecolor="none",
            )
            ax.add_patch(row_bg)

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
    set_pub_style()
    base_dir = Path(__file__).parent

    rows = parse_results(base_dir / "results.tsv")
    print(f"Parsed {len(rows)} experiments")

    plot_iterations(
        rows,
        base_dir / "iterations.png",
        kept_labels=KEPT_LABELS,
        title="ELPD Optimization Across 21 Bayesian Model Iterations",
    )
    plot_causal_estimates(base_dir / "causal_estimates.png")
    plot_model_diagram(base_dir / "model_diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
