#!/usr/bin/env python3
"""Supplementary Figure S1: top-k accuracy of A×1 / P / MDT across the datasets
and across the two ER-Reason input conditions (primary judge, GLM-5.3-flash ×
v3). Four panels: CPC, MCR, ER-Reason presentation-only, ER-Reason workup-
informed. Values recomputed from the jsonl results + judge cache; hard-coded
here from that verified output (CPC/MCR 2026-09-17; ER-Reason five-seed
2026-09-18, routing_study/results/topn_erreason/seed_summary.json; workup
condition 2026-09-19, routing_study/results/workup_vs_baseline.json).
Okabe-Ito colorblind-safe palette. Output: paper/supplementary_figure_s1.png
(300 dpi). Error bars are SD across the five seeds in every panel.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (mean %, SD %) per strategy; mean ± SD across five seeds for every panel
DATA = {
    "CPC (n = 87)": {
        "A×1": [(57.9, 1.9), (76.1, 4.1), (81.1, 2.6)],
        "P":   [(53.8, 4.0), (72.2, 2.6), (76.8, 2.1)],
        "MDT": [(60.5, 3.1), (81.6, 2.4), (85.5, 2.1)],
    },
    "MCR (n = 406)": {
        "A×1": [(57.0, 0.9), (69.9, 0.6), (75.6, 0.8)],
        "P":   [(56.4, 0.3), (69.2, 1.1), (75.0, 1.0)],
        "MDT": [(56.0, 0.8), (73.0, 0.8), (79.6, 0.8)],
    },
    "ER-Reason, presentation only (n = 364)": {
        "A×1": [(40.1, 0.9), (54.6, 1.1), (61.6, 0.8)],
        "P":   [(37.7, 0.6), (48.5, 1.5), (55.2, 0.8)],
        "MDT": [(34.3, 1.0), (45.5, 1.0), (51.9, 1.1)],
    },
    "ER-Reason, plus objective results (n = 364)": {
        "A×1": [(42.9, 0.8), (57.2, 0.8), (66.4, 0.6)],
        "P":   [(41.9, 0.8), (53.7, 2.1), (59.7, 1.9)],
        "MDT": [(36.2, 0.5), (48.7, 1.3), (56.1, 1.3)],
    },
}
COLORS = {"A×1": "#0072B2", "P": "#E69F00", "MDT": "#009E73"}  # Okabe-Ito
KS = ["Top-1", "Top-3", "Top-5"]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharey=True)
for ax, (ds, strat) in zip(axes.ravel(), DATA.items()):
    x = np.arange(3)
    w = 0.26
    for i, (name, vals) in enumerate(strat.items()):
        means = [v[0] for v in vals]
        sds = [v[1] for v in vals]
        ax.bar(x + (i - 1) * w, means, w, label=name, color=COLORS[name],
               yerr=sds, capsize=3, error_kw=dict(lw=1, ecolor="#555555"))
        for xi, m, sd in zip(x + (i - 1) * w, means, sds):
            ax.text(xi, m + sd + 1.2, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_title(ds, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(KS)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy / recall (%)")
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Top-k performance of the three strategies across datasets and the "
             "two emergency input conditions\n(primary judge: GLM-5.3-flash × v3; "
             "error bars: SD across five seeds)", fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("paper/supplementary_figure_s1.png", dpi=300,
            bbox_inches="tight")
print("wrote paper/supplementary_figure_s1.png")
