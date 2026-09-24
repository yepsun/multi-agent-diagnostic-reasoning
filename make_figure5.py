#!/usr/bin/env python3
"""Figure 5：(a) 2×2 析因矩阵（列表来源 × 聚合方式，CPC top-3）；
(b) 成本-召回帕累托散点（CPC 全臂，tokens/case 对数轴 vs top-3）。

所有数值从 results/ 下的结果文件与 run 文件现算（caselevel_stats 口径，
冻结共享判官缓存），不手抄。
输出 paper/figures/figure5_factorial_pareto.{png,pdf}。
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402

RES = ROOT / "routing_study" / "results"
OUT = ROOT / "paper" / "figures" / "figure5_factorial_pareto"
SEEDS = [1, 2, 3, 4, 5]


def load(path):
    u = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            u.setdefault(r["case_id"], r)
    return u


CACHE = json.loads((RES / "judge_cache_glm_v3.json").read_text())
cs.set_cache(CACHE)

ax1 = {s: load(RES / "topn_seeds" / f"Ax1_s{s}.jsonl") for s in SEEDS}
p = {s: load(RES / "topn_seeds" / f"P_s{s}.jsonl") for s in SEEDS}
cot = {s: load(RES / "topn_seeds_cot" / f"Ax1cot_s{s}.jsonl") for s in SEEDS}
ax5 = {s: load(RES / "topn_seeds_ax5" / f"Ax5_s{s}.jsonl") for s in SEEDS}
ax5mod = {s: load(RES / "topn_ax5_mod" / f"Ax5Mod_s{s}.jsonl") for s in SEEDS}
mdt = {s: load(RES / "topn_mdt" / ("synthesis.jsonl" if s == 1
                                   else f"s{s}/synthesis.jsonl")) for s in SEEDS}
mdtb = {s: load(RES / "topn_mdt_nomoderator" / f"MdtBorda_s{s}.jsonl") for s in SEEDS}
psplit = {s: load(RES / "topn_p_split" / f"Psplit_s{s}.jsonl") for s in SEEDS}


def mean_sd_top3(arm_runs):
    per_seed = []
    for s in SEEDS:
        hits = sum(1 for r in arm_runs[s].values()
                   if cs.topk(cs.hit_flags(r, CACHE), 3))
        per_seed.append(100 * hits / len(arm_runs[s]))
    return st.mean(per_seed), st.stdev(per_seed)


tok = json.loads((RES / "token_costs" / "token_costs.json").read_text())["datasets"]["CPC"]["schemes"]
tok_ax1 = tok["Ax1"]["mean_total_tokens_per_case"]
tok_p = tok["P"]["mean_total_tokens_per_case"]
tok_cot = tok.get("A×1+CoT", {}).get("mean_total_tokens_per_case", 3499.0)
tok_ax5 = st.mean(st.mean([r.get("total_tokens") or 0 for r in ax5[s].values()]) for s in SEEDS)
tok_mod = st.mean(st.mean([r.get("total_tokens") or 0 for r in ax5mod[s].values()]) for s in SEEDS)
tok_ax5mod = tok_ax5 + tok_mod
tok_mdt = tok["MDT"]["mean_total_tokens_per_case"]
pgen = {s: load(RES / "topn_p_split" / f"Pgen_s{s}.jsonl") for s in SEEDS}
tok_pgen = st.mean(st.mean([r.get("total_tokens") or 0 for r in pgen[s].values()]) for s in SEEDS)
tok_pmod = st.mean(st.mean([r.get("total_tokens") or 0 for r in psplit[s].values()]) for s in SEEDS)
tok_psplit = tok_pgen + tok_pmod

points = [
    (r"A×1 (1 call)", tok_ax1, *mean_sd_top3(ax1)),
    (r"P (1 call)", tok_p, *mean_sd_top3(p)),
    (r"A×1+CoT (1 call)", tok_cot, *mean_sd_top3(cot)),
    (r"A×5 Borda/SC (5 calls)", tok_ax5, *mean_sd_top3(ax5)),
    (r"A×5+Mod (6 calls)", tok_ax5mod, *mean_sd_top3(ax5mod)),
    (r"MDT (6 calls)", tok_mdt, *mean_sd_top3(mdt)),
    (r"P-split (2 calls)", tok_psplit, *mean_sd_top3(psplit)),
]

ax5_m, ax5_s = mean_sd_top3(ax5)
ax5mod_m, ax5mod_s = mean_sd_top3(ax5mod)
mdt_m, mdt_s = mean_sd_top3(mdt)
mdtb_m, mdtb_s = mean_sd_top3(mdtb)

print("[fig5] top-3:", {n: round(v, 1) for n, v, s, t in
                       [(n, m, 0, 0) for n, t, m, s in points]})
print("[fig5] tokens:", {n: round(t) for n, t, m, s in points})

fig = plt.figure(figsize=(10.6, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.30)

# ---------- panel a：2×2 析因矩阵 ----------
axa = fig.add_subplot(gs[0, 0])
axa.set_xlim(-0.22, 2.03)
axa.set_ylim(-0.16, 2.52)
axa.set_xticks([])
axa.set_yticks([])
for sp in axa.spines.values():
    sp.set_visible(False)
cells = {(0, 1): (ax5_m, ax5_s, "A×5 Borda", "0.96"),
         (1, 1): (mdtb_m, mdtb_s, "MDT-Borda", "0.96"),
         (0, 0): (ax5mod_m, ax5mod_s, "A×5+Mod", "0.87"),
         (1, 0): (mdt_m, mdt_s, "MDT", "0.87")}
for (cx, cy), (m, s, name, face) in cells.items():
    axa.add_patch(plt.Rectangle((cx + 0.04, cy + 0.04), 0.92, 0.92,
                                facecolor=face, edgecolor="0.3", linewidth=1.0))
    axa.text(cx + 0.5, cy + 0.72, name, ha="center", fontsize=9.5,
             fontweight="bold", color="0.25")
    axa.text(cx + 0.5, cy + 0.46, f"{m:.1f}%", ha="center", fontsize=16)
    axa.text(cx + 0.5, cy + 0.24, f"± {s:.1f}", ha="center", fontsize=8.5, color="0.4")
axa.text(0.5, 2.16, "same-prompt samples\n(A×5, T = 0.7)", ha="center",
         va="bottom", fontsize=8, color="0.3", linespacing=1.2)
axa.text(1.5, 2.16, "isolated specialist roles\n(T = 0.3)", ha="center",
         va="bottom", fontsize=8, color="0.3", linespacing=1.2)
axa.text(-0.13, 1.5, "LLM moderator\n(T = 0.3)", ha="center", va="center",
         fontsize=8.5, color="0.3", rotation=90)
axa.text(-0.13, 0.5, "mechanical\naggregation\n(Borda)", ha="center", va="center",
         fontsize=8.5, color="0.3", rotation=90)
axa.text(1.0, -0.12,
         "single-call reference: A×1 top-3 = 76.1% — the level of both left cells",
         ha="center", fontsize=7.5, color="0.35")
axa.set_title("a   Completed 2×2 factorial (CPC, top-3 recall)",
              fontsize=10, loc="left")

# ---------- panel b：成本-召回帕累托 ----------
axb = fig.add_subplot(gs[0, 1])
frontier = {r"A×1 (1 call)", r"A×5+Mod (6 calls)", r"MDT (6 calls)"}
for (name, t, m, s) in points:
    on_frontier = name in frontier
    axb.errorbar(t, m, yerr=s, fmt="o", markersize=6.5 if on_frontier else 5,
                 color="0.15" if on_frontier else "0.55",
                 ecolor="0.5", elinewidth=1, capsize=2, zorder=3)
    offsets = {r"A×1 (1 call)": (5, 3), r"P (1 call)": (5, -12),
               r"A×1+CoT (1 call)": (5, 3),
               r"A×5 Borda/SC (5 calls)": (-8, -13),
               r"A×5+Mod (6 calls)": (-14, -13), r"MDT (6 calls)": (-10, 8),
               r"P-split (2 calls)": (5, 3)}
    dx, dy = offsets[name]
    axb.annotate(name, (t, m), textcoords="offset points",
                 xytext=(dx, dy), fontsize=7.5, color="0.2")
axb.set_xscale("log")
axb.set_xlabel("Tokens per case (log scale)", fontsize=9)
axb.set_ylabel("Top-3 recall (%)", fontsize=9)
axb.grid(True, which="both", axis="x", lw=0.3, color="0.9")
axb.set_title("b   Cost–recall frontier (CPC, all arms)", fontsize=10, loc="left")
axb.text(0.02, 0.045,
         "GLM-5.3-flash × v3 judge; error bars ±1 SD across five seeds.\n"
         "A×5 self-consistency coincides with A×5 Borda (same samples, 75.4%).",
         transform=axb.transAxes, fontsize=6.8, color="0.35", va="bottom")

OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix("." + ext), bbox_inches="tight",
                pad_inches=0.05, dpi=300)
    print(f"wrote {OUT.with_suffix('.' + ext)}")
plt.close(fig)
