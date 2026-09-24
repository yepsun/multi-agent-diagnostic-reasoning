#!/usr/bin/env python3
"""Build the three main figures of the manuscript from stored result files.

Every number drawn here is read from files under routing_study/results/ (never
transcribed by hand), except the display labels of the five MDT roles, which are
design descriptors taken from the ROLES constant in
routing_study/scripts/mdt_cpc.py.

Usage
-----
    ./.venv/bin/python paper/make_main_figures.py                # all figures
    ./.venv/bin/python paper/make_main_figures.py --figures 2    # one figure
    ./.venv/bin/python paper/make_main_figures.py --strategies Ax1,P,MDT,Ax1cot

Outputs (paper/figures/): figure1_study_design.{png,pdf},
figure2_main_results.{png,pdf}, figure3_boundary_conditions.{png,pdf}.
The script also prints the values it drew, so the figures can be checked
against routing_study/results/*.md without leaving the terminal.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "routing_study" / "results"
OUTDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "hatch.linewidth": 0.6,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PRIMARY_CACHE = "judge_cache_glm_v3.json"
JUDGE_LABEL = "GLM-5.3-flash x v3"
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
N_BOOT = 10000
BOOT_SEED = 20260917  # same seed as routing_study/scripts/stats_caselevel.py

# --------------------------------------------------------------------------
# Strategy arms.  To add an arm: add one Arm here, one SOURCES entry per
# dataset below, and pass its key to --strategies.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    key: str          # key used inside the result files ("Ax1", "P", "MDT", ...)
    label: str        # display label
    tone: float       # grayscale tone of the bar face (0 black - 1 white)


ARMS = {
    "Ax1": Arm("Ax1", r"A$\times$1", 0.30),
    "P": Arm("P", "P", 0.55),
    "MDT": Arm("MDT", "MDT", 0.80),
    "Ax1cot": Arm("Ax1cot", r"A$\times$1 + CoT", 0.15),
    "Ax5": Arm("Ax5", r"A$\times$5", 0.42),
}
DEFAULT_STRATEGIES = ["Ax1", "P", "MDT"]

# --------------------------------------------------------------------------
# Data sources per (dataset, arm).  Each spec is one of
#   ("holdout", split)                 -> routing_study/results/holdout46_primary.json
#   ("erreason", outdir)               -> <outdir>/seed_summary.json
#   ("recompute", run_files, cache)    -> per-case judgements replayed from the run
#                                         files and the shared judge cache, exactly as
#                                         routing_study/scripts/stats_caselevel.py does
# --------------------------------------------------------------------------


def _cpc_runs(arm_key: str) -> list[str]:
    if arm_key == "Ax1":
        return [f"topn_seeds/Ax1_s{s}.jsonl" for s in EXPECTED_SEEDS]
    if arm_key == "P":
        return [f"topn_seeds/P_s{s}.jsonl" for s in EXPECTED_SEEDS]
    if arm_key == "MDT":
        return ["topn_mdt/synthesis.jsonl"] + [f"topn_mdt/s{s}/synthesis.jsonl" for s in (2, 3, 4, 5)]
    if arm_key == "Ax1cot":
        return [f"topn_seeds_cot/Ax1cot_s{s}.jsonl" for s in EXPECTED_SEEDS]
    if arm_key == "Ax5":
        return [f"topn_seeds_ax5/Ax5_s{s}.jsonl" for s in EXPECTED_SEEDS]
    raise KeyError(arm_key)


def _mcr_runs(arm_key: str) -> list[str]:
    if arm_key == "Ax1":
        return ["topn_mcr/ax1.jsonl"] + [f"topn_mcr_seeds/Ax1_s{s}.jsonl" for s in (2, 3, 4, 5)]
    if arm_key == "P":
        return ["topn_mcr/p.jsonl"] + [f"topn_mcr_seeds/P_s{s}.jsonl" for s in (2, 3, 4, 5)]
    if arm_key == "MDT":
        return ["topn_mcr/mdt_synth.jsonl"] + [f"topn_mcr_seeds/s{s}/mdt_synth.jsonl" for s in (2, 3, 4, 5)]
    raise KeyError(arm_key)


SOURCES = {
    "CPC": {k: ("holdout", "full87") for k in ("Ax1", "P", "MDT")},
    "MCR": {k: ("recompute", _mcr_runs(k), PRIMARY_CACHE) for k in ("Ax1", "P", "MDT")},
    "ER": {k: ("erreason", "topn_erreason") for k in ("Ax1", "P", "MDT")},
}
# Extra arms: CPC runs of the A x 1 + CoT control and of the A x 5 control.  Their
# run files follow the same "<dir>/<Arm>_s<seed>.jsonl" layout as topn_seeds/, and
# the shared GLM cache holds their adjudications, so the generic replay loader
# resolves them.  An arm is held back from the figure until all five seeds exist.
SOURCES["CPC"]["Ax1cot"] = ("recompute", _cpc_runs("Ax1cot"), PRIMARY_CACHE)
SOURCES["CPC"]["Ax5"] = ("recompute", _cpc_runs("Ax5"), PRIMARY_CACHE)

DATASETS = {
    "CPC": {"caption": r"CPC cases ($n$ = 87)", "panel_letter": "a"},
    "MCR": {"caption": r"MedCaseReasoning, external validation ($n$ = 406)", "panel_letter": "b"},
    "ER": {"caption": r"ER-Reason encounters ($n$ = 364), two input conditions", "panel_letter": "c"},
}
ENDPOINTS = ("1", "3", "5")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

_JSON_CACHE: dict[str, dict] = {}


def rj(rel: str) -> dict:
    if rel not in _JSON_CACHE:
        _JSON_CACHE[rel] = json.loads((RES / rel).read_text())
    return _JSON_CACHE[rel]


def load_jsonl(rel: str) -> dict:
    with open(RES / rel) as fh:
        return {json.loads(line)["case_id"]: json.loads(line) for line in fh if line.strip()}


@dataclass
class ArmData:
    """Per-seed top-k rates (percent) and, when available, per-case rates."""

    per_seed: dict[str, list[float]]
    case_rate: dict[str, np.ndarray] | None = None
    source: str = ""

    def n_seeds(self) -> int:
        return min((len(v) for v in self.per_seed.values()), default=0)


def _hit_flags(rec: dict, judge: dict) -> list:
    flags = []
    for cand in rec["top5"][:5]:
        key = rec["gold"][:150] + "||" + cand[:150]
        flags.append(bool(judge[key]) if key in judge else None)
    return flags


def _topk(flags: list, k: int):
    window = flags[:k]
    if not any(x is not None for x in window):
        return None
    return any(x is True for x in window)


def recompute(run_files: list[str], cache: str) -> ArmData:
    judge = rj(cache)
    have = [p for p in run_files if (RES / p).exists()]
    if not have:
        raise FileNotFoundError(f"no run files found for {run_files}")
    if len(have) != len(run_files):
        print(f"  [warn] {len(have)}/{len(run_files)} run files present for {run_files[0]}")
    runs = [load_jsonl(p) for p in have]
    ids = sorted(runs[0])
    per_seed: dict[str, list[float]] = {k: [] for k in ENDPOINTS}
    case_rate: dict[str, np.ndarray] = {}
    for k in ENDPOINTS:
        per_case: list[list] = [[] for _ in ids]
        for run in runs:
            hits = []
            for i, cid in enumerate(ids):
                rec = run.get(cid)
                if rec is None:
                    per_case[i].append(None)
                    continue
                flag = _topk(_hit_flags(rec, judge), int(k))
                per_case[i].append(flag)
                hits.append(flag)
            judged = [h for h in hits if h is not None]
            per_seed[k].append(100.0 * sum(judged) / len(judged))
        case_rate[k] = np.array([sum(v) / len(v) for v in per_case
                                 if v and all(x is not None for x in v)])
    return ArmData(per_seed, case_rate, source=f"replayed from {len(have)} run files + {cache}")


def arm_data(dataset: str, arm_key: str) -> ArmData | None:
    spec = SOURCES.get(dataset, {}).get(arm_key)
    if spec is None:
        return None
    kind = spec[0]
    if kind == "holdout":
        endpoints = rj("holdout46_primary.json")["judges"]["glm_v3"]["splits"][spec[1]]["endpoints"]
        per_seed = {k: list(endpoints[k]["strategies"][arm_key]["per_seed_pct"]) for k in ENDPOINTS}
        return ArmData(per_seed, None, source="holdout46_primary.json (glm_v3)")
    if kind == "erreason":
        strata = rj(f"{spec[1]}/seed_summary.json")["strata"]["all"]["per_seed"]
        per_seed = {k: [100.0 * r for r in strata[arm_key][k]] for k in ENDPOINTS}
        return ArmData(per_seed, None, source=f"{spec[1]}/seed_summary.json")
    if kind == "recompute":
        try:
            return recompute(spec[1], spec[2])
        except FileNotFoundError as exc:
            print(f"  [warn] {dataset}/{arm_key}: {exc} — arm unavailable")
            return None
    raise ValueError(f"unknown source kind {kind!r}")


def stored_comparisons(dataset: str) -> dict[tuple[str, str, str], dict]:
    """Pairwise case-level statistics already in the result files, in percent."""
    out: dict[tuple[str, str, str], dict] = {}
    if dataset == "CPC":
        endpoints = rj("holdout46_primary.json")["judges"]["glm_v3"]["splits"]["full87"]["endpoints"]
        for k in ENDPOINTS:
            for name, c in endpoints[k]["comparisons"].items():
                a, b = name.split("_vs_")
                out[(a, b, k)] = {"p": c["wilcoxon_p"], "diff": c["mean_diff_pct"],
                                  "ci": list(c["boot95_ci_pct"]), "n": c["n_cases_paired"],
                                  "where": "holdout46_primary.json"}
    elif dataset == "MCR":
        topk = rj("stats_caselevel.json")["MCR406"]["topk"]
        for k in ENDPOINTS:
            for name, c in topk[k]["comparisons"].items():
                a, b = name.split("_vs_")
                out[(a, b, k)] = {"p": c["wilcoxon_p"], "diff": 100.0 * c["mean_diff"],
                                  "ci": [100.0 * v for v in c["boot95_ci"]], "n": c["n_cases"],
                                  "where": "stats_caselevel.json"}
    elif dataset == "ER":
        topk = rj("topn_erreason/seed_summary.json")["strata"]["all"]["topk"]
        for k in ENDPOINTS:
            for name, c in topk[k]["comparisons"].items():
                a, b = name.split("_vs_")
                out[(a, b, k)] = {"p": c["wilcoxon_p"], "diff": 100.0 * c["mean_diff"],
                                  "ci": [100.0 * v for v in c["boot95_ci"]], "n": c["n_cases"],
                                  "where": "topn_erreason/seed_summary.json"}
    return out


def _wilcoxon_p(ca: np.ndarray, cb: np.ndarray) -> float:
    if np.all(ca - cb == 0):
        return 1.0
    return float(wilcoxon(ca, cb, zero_method="wilcox").pvalue)


def _boot_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def comparison(a: ArmData, b: ArmData, ka: str, kb: str, k: str, stored: dict) -> dict | None:
    """Stored statistic when the repository has one, otherwise replayed."""
    for key, sign in (((ka, kb, k), 1), ((kb, ka, k), -1)):
        hit = stored.get(key)
        if hit is not None:
            if sign == 1:
                return hit
            lo, hi = hit["ci"]
            return {**hit, "diff": -hit["diff"], "ci": [-hi, -lo]}
    if a.case_rate is None or b.case_rate is None:
        return None
    ca, cb = a.case_rate[k], b.case_rate[k]
    if len(ca) != len(cb):
        return None
    diff = ca - cb
    lo, hi = _boot_ci(diff)
    return {"p": _wilcoxon_p(ca, cb), "diff": 100.0 * float(diff.mean()),
            "ci": [100.0 * lo, 100.0 * hi], "n": len(ca),
            "where": "replayed in make_main_figures.py"}


def stars(p: float | None) -> str:
    if p is None:
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def bracket(ax, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.7, color="0.15",
            solid_joinstyle="miter", clip_on=False, zorder=6)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=8,
            color="0.15", clip_on=False, zorder=6)


def bracket_bands(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group spans into horizontal bands so that no two labels can collide."""
    bands: list[list[tuple[int, int]]] = []
    for pair in pairs:
        for band in bands:
            if all(pair[1] < q[0] or pair[0] > q[1] for q in band):
                band.append(pair)
                break
        else:
            bands.append([pair])
    return bands


def tone_ramp(n: int, lo: float = 0.25, hi: float = 0.85) -> list[float]:
    """Evenly spaced grayscale tones, dark to light, for n bars in a panel."""
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def style_bar(ax, x: float, mean: float, sd: float, tone: float, barw: float) -> None:
    ax.bar(x, mean, barw, color=str(tone), edgecolor="0.15", linewidth=0.7, zorder=3)
    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="0.15", elinewidth=0.8,
                capsize=1.7, capthick=0.8, zorder=4)


def box(ax, x, y, w, h, text, fontsize=8, dashed=False, face="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.010",
        linewidth=0.8 if not dashed else 0.7, linestyle="--" if dashed else "-",
        edgecolor="0.25", facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, linespacing=1.35)


def arrow(ax, xy_from, xy_to, lw=0.8):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color="0.25",
                                shrinkA=0, shrinkB=0, mutation_scale=8), zorder=1)


# --------------------------------------------------------------------------
# Figure 1 - study design
# --------------------------------------------------------------------------


def figure1(write: bool):
    calls = fig1_call_counts()
    fig = plt.figure(figsize=(7.2, 4.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.030, 0.860, 0.940, 0.108,
        "Shared input: identical case text, identical backbone (qwen3.8-flash, 176B, thinking disabled),\n"
        "identical output format: a ranked top-5 differential list\n"
        "(ii) on ER-Reason the input text itself varies: presentation only vs + objective results of the visit",
        fontsize=8, face="0.93")

    colx = {"Ax1": 0.030, "P": 0.360, "MDT": 0.690}
    colw = {"Ax1": 0.260, "P": 0.260, "MDT": 0.290}
    titles = {"Ax1": r"A$\times$1 — single direct call",
              "P": "P — five personas, one context",
              "MDT": "MDT — isolated roles + moderator"}
    for key, x0 in colx.items():
        cx = x0 + colw[key] / 2
        arrow(ax, (cx, 0.860), (cx, 0.842), lw=0.8)
        ax.text(cx, 0.824, titles[key], ha="center", va="center", fontsize=8.5, fontweight="bold")
        arrow(ax, (cx, 0.806), (cx, 0.762), lw=0.6)

    # --- A x 1: one call, one context
    x0 = colx["Ax1"]
    box(ax, x0, 0.520, colw["Ax1"], 0.240,
        "One call\n\nsingle context\nno intermediate\nreasoning text\n(ranked list only)",
        fontsize=8, face="0.90")
    box(ax, x0, 0.350, colw["Ax1"], 0.080, "ranked top-5 list", fontsize=8)
    arrow(ax, (x0 + colw["Ax1"] / 2, 0.520), (x0 + colw["Ax1"] / 2, 0.430))
    ax.text(x0 + colw["Ax1"] / 2, 0.285, f"{calls['Ax1']} call / case", ha="center", va="center", fontsize=8)

    # --- P: one call, five perspectives sharing a context
    x0 = colx["P"]
    box(ax, x0, 0.455, colw["P"], 0.305,
        "One call\n\none context, five\nperspectives in sequence:\nattending internist ·\n"
        "pathophysiology ·\nimaging & laboratory ·\nepidemiology · skeptic",
        fontsize=8, face="0.90")
    box(ax, x0, 0.350, colw["P"], 0.080, "ranked top-5 list", fontsize=8)
    arrow(ax, (x0 + colw["P"] / 2, 0.455), (x0 + colw["P"] / 2, 0.430))
    ax.text(x0 + colw["P"] / 2, 0.285, f"{calls['P']} call / case", ha="center", va="center", fontsize=8)

    # --- MDT: five mutually blind role calls plus a moderator
    x0, w = colx["MDT"], colw["MDT"]
    cx = x0 + w / 2
    roles = ["Attending internist — unifying dx",
             "Pathophysiologist — mechanism",
             "Imaging & lab — objective data",
             "Epidemiologist — risk factors",
             "Skeptic / challenger — objection"]
    h, gap = 0.033, 0.006
    rows = [0.751 - i * (h + gap) for i in range(len(roles))]
    for role, y in zip(roles, rows):
        box(ax, x0, y, w, h, role, fontsize=8, face="0.90")
        arrow(ax, (x0 - 0.014, 0.800), (x0 - 0.004, y + h / 2), lw=0.5)
    ax.add_patch(FancyBboxPatch((x0 - 0.016, rows[-1] - 0.008), w + 0.032, rows[0] - rows[-1] + h + 0.016,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                linewidth=0.7, linestyle=":", edgecolor="0.45", facecolor="none", zorder=1))
    ax.text(cx, rows[-1] - 0.040, "the five role calls are mutually blind", ha="center", va="center",
            fontsize=7.5, color="0.25", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    mod_y, mod_h = 0.452, 0.056
    box(ax, x0, mod_y, w, mod_h, "Moderator\nintegrates the five opinions", fontsize=8, face="0.76")
    for y in rows:
        arrow(ax, (cx, y), (cx, mod_y + mod_h), lw=0.5)
    box(ax, x0, 0.350, w, 0.080, "ranked top-5 list", fontsize=8)
    arrow(ax, (cx, mod_y), (cx, 0.430))
    ax.text(cx, 0.285, f"{calls['MDT']} calls / case", ha="center", va="center", fontsize=8)

    # the controlled contrast (i): P against MDT
    ax.annotate("", xy=(0.648, 0.480), xytext=(0.648, 0.740),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="0.3", mutation_scale=7))
    ax.text(0.640, 0.610, "(i)", fontsize=8, va="center", ha="right", fontweight="bold", color="0.15")

    ax.plot([0.030, 0.970], [0.235, 0.235], lw=0.6, color="0.75")
    footer = [
        "(i)  P vs MDT, the controlled contrast: roles, prompt material, model and output format are held",
        "       fixed, and the only change is whether the five perspectives are isolated from one another.",
        "(ii) ER-Reason input contrast: the same 364 encounters with identical prompts, model, decoding,",
        "       seeds and judge; the only change is the input text —",
        "       presentation only (ED provider note truncated before \u201cMedical Decision Making\u201d)",
        "       versus   presentation + objective results of the same visit (laboratory values, vital signs,",
        "       imaging / ECG / ultrasound reports).",
        "Endpoint: the reference diagnosis appears within the top-1, top-3 or top-5 candidates, adjudicated by",
        "GLM-5.3-flash under the v3 rules, blinded to strategy.  Call counts per case, from",
        f"token_costs/token_costs.json:  A\u00d71 = {calls['Ax1']},  P = {calls['P']},  MDT = {calls['MDT']}.",
    ]
    for i, line in enumerate(footer):
        ax.text(0.030, 0.200 - i * 0.0188, line, fontsize=8, va="center", color="0.3" if i < 7 else "0.15")

    save(fig, OUTDIR / "figure1_study_design", write)


def fig1_call_counts() -> dict[str, int]:
    """Mean calls per case per scheme, read from the token cost table."""
    costs = rj("token_costs/token_costs.json")
    seen: dict[str, set] = {}
    for ds in costs["datasets"].values():
        for scheme, spec in ds["schemes"].items():
            seen.setdefault(scheme, set()).add(spec["mean_calls_per_case"])
    out = {}
    for scheme, values in seen.items():
        if len(values) != 1:
            raise ValueError(f"{scheme}: inconsistent call counts across datasets: {values}")
        out[scheme] = int(values.pop())
    return out


# --------------------------------------------------------------------------
# Figure 2 - main results
# --------------------------------------------------------------------------


def figure2(write: bool, strategy_keys: list[str]):
    arms = [ARMS[k] for k in strategy_keys if k in ARMS]
    for k in strategy_keys:
        if k not in ARMS:
            print(f"  [warn] unknown strategy {k!r}: ignored")
    if len(arms) < 2:
        raise SystemExit("need at least two strategies with a style entry in ARMS")

    data = load_panel_data(arms)
    usable = {}
    for key, panel in data.items():
        keep = [a for a in arms if panel["arms"].get(a.key) is not None]
        missing = [a.key for a in arms if panel["arms"].get(a.key) is None]
        if missing:
            print(f"  [warn] {key}: no complete data for {', '.join(missing)} — not drawn in this panel")
        if len(keep) < 2:
            print(f"  [warn] {key}: fewer than two strategies available — panel skipped")
            continue
        panel["arms"] = {a.key: panel["arms"][a.key] for a in keep}
        usable[key] = panel

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], hspace=0.58, wspace=0.30,
                          left=0.085, right=0.985, top=0.945, bottom=0.135)
    layout = {"CPC": gs[0, 0], "MCR": gs[0, 1], "ER": gs[1, :]}
    for key in ("CPC", "MCR", "ER"):
        if key in usable:
            _draw_topk_panel(fig.add_subplot(layout[key]), key, usable[key],
                             [ARMS[k] for k in usable[key]["arms"]])

    drawn_arms = [a for a in arms if any(a.key in d["arms"] for d in usable.values())]
    fig.legend(handles=[Patch(facecolor=str(a.tone), edgecolor="0.15", lw=0.7, label=a.label)
                        for a in drawn_arms],
               loc="lower center", ncol=len(drawn_arms), frameon=False,
               bbox_to_anchor=(0.5, 0.055), handlelength=1.4, columnspacing=1.8)
    for i, line in enumerate([
        f"Primary judge: {JUDGE_LABEL}.  Error bars: +/-1 SD across the five seeds.  Brackets: paired Wilcoxon",
        "signed-rank test on per-case five-seed hit rates (*** p < 0.001, ** p < 0.01, * p < 0.05, n.s. = not significant).",
    ]):
        fig.text(0.5, 0.014 + (1 - i) * 0.019, line, ha="center", va="bottom", fontsize=8, color="0.3")
    save(fig, OUTDIR / "figure2_main_results", write)


def load_panel_data(arms):
    workup = rj("workup_vs_baseline.json")["strata"]["all"]
    out = {}
    for key in ("CPC", "MCR", "ER"):
        panel = {"arms": {}, "stored": stored_comparisons(key), "meta": DATASETS[key]}
        for arm in arms:
            d = arm_data(key, arm.key)
            if d is None:
                panel["arms"][arm.key] = None
                continue
            if d.n_seeds() < len(EXPECTED_SEEDS):
                print(f"  [warn] {key}/{arm.key}: only {d.n_seeds()}/{len(EXPECTED_SEEDS)} seeds available "
                      "— arm held back until the run finishes")
                panel["arms"][arm.key] = None
                continue
            panel["arms"][arm.key] = d
        if key == "ER":
            for arm in arms:
                d = panel["arms"].get(arm.key)
                if d is None or arm.key not in ("Ax1", "P", "MDT"):
                    continue
                d.workup = {k: workup["mean_sd"][arm.key]["workup"][k] for k in ENDPOINTS}
                d.workup_gain = {k: workup["case_level"][arm.key]["per_k"][k] for k in ENDPOINTS}
        out[key] = panel
    _verify_against_stats_caselevel(out)
    return out


def _verify_against_stats_caselevel(panels):
    """Compare any replayed per-case rates with the stored case-level means.

    The MedCaseReasoning bars are replayed from the run files and the judge cache
    rather than read from a summary, so this is the check that keeps them honest
    against routing_study/results/stats_caselevel.json.
    """
    ref_all = rj("stats_caselevel.json")
    for key, ref_key in (("CPC", "CPC87"), ("MCR", "MCR406")):
        ref = ref_all.get(ref_key)
        if ref is None or key not in panels:
            continue
        worst, where = 0.0, "-"
        for arm_key, d in panels[key]["arms"].items():
            if d is None or d.case_rate is None:
                continue
            for k in ENDPOINTS:
                stored = ref["topk"][k]["case_rate_mean"].get(arm_key)
                if stored is None:
                    continue
                delta = abs(float(d.case_rate[k].mean()) - stored) * 100.0
                if delta > worst:
                    worst, where = delta, f"{arm_key}/top-{k}"
        if where == "-":
            continue
        print(f"  consistency vs stats_caselevel.json ({ref_key}): "
              f"max |replayed case-level mean - stored| = {worst:.4f} pp (worst cell {where})")


def _draw_topk_panel(ax, key, panel, arms):
    paired = key == "ER"
    n_arm = len(arms)
    barw = 0.11 if paired else 0.20
    groups = np.arange(len(ENDPOINTS))

    def bar_x(g, i, cond=0):
        if paired:
            return g + (-0.25 if cond == 0 else 0.25) + 0.145 * (i - (n_arm - 1) / 2)
        return g + 0.26 * (i - (n_arm - 1) / 2)

    means: dict[str, dict[str, float]] = {}
    sds: dict[str, dict[str, float]] = {}
    for arm in arms:
        d = panel["arms"][arm.key]
        means[arm.key] = {k: st.mean(d.per_seed[k]) for k in ENDPOINTS}
        sds[arm.key] = {k: st.stdev(d.per_seed[k]) for k in ENDPOINTS}
        if paired:
            means[arm.key + "_w"] = {k: 100.0 * d.workup[k]["mean"] for k in ENDPOINTS}
            sds[arm.key + "_w"] = {k: 100.0 * d.workup[k]["sd"] for k in ENDPOINTS}

    peak = max([means[a.key][k] + sds[a.key][k] for a in arms for k in ENDPOINTS]
               + ([means[a.key + "_w"][k] + sds[a.key + "_w"][k] for a in arms for k in ENDPOINTS] if paired else []))

    for i, arm in enumerate(arms):
        for j, k in enumerate(ENDPOINTS):
            for cond, tag in ((0, ""), (1, "_w")) if paired else ((0, ""),):
                mean = means[arm.key + tag][k]
                sd = sds[arm.key + tag][k]
                x = bar_x(groups[j], i, cond)
                ax.bar(x, mean, barw, color=str(arm.tone), edgecolor="0.15", linewidth=0.7, zorder=3)
                ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="0.15", elinewidth=0.8,
                            capsize=1.6, capthick=0.8, zorder=4)

    # Brackets come in horizontal bands; two brackets may share a band only when
    # their spans do not overlap, so no two labels can collide.
    bands: list[list[tuple[int, int]]] = []
    for pair in [(i, i + 1) for i in range(n_arm - 1)] + ([(0, n_arm - 1)] if n_arm > 2 else []):
        for band in bands:
            if all(pair[1] < q[0] or pair[0] > q[1] for q in band):
                band.append(pair)
                break
        else:
            bands.append([pair])

    top = peak * (1.16 + 0.085 * len(bands))
    step = 0.085 * peak
    pad = 0.028 * peak

    for j, k in enumerate(ENDPOINTS):
        drawn = [means[a.key][k] + sds[a.key][k] for a in arms]
        if paired:
            drawn += [means[a.key + "_w"][k] + sds[a.key + "_w"][k] for a in arms]
        y0 = max(drawn) + pad
        for li, band in enumerate(bands):
            for ia, ib in band:
                a, b = arms[ia], arms[ib]
                stat = comparison(panel["arms"][a.key], panel["arms"][b.key], a.key, b.key, k, panel["stored"])
                x1 = bar_x(groups[j], ia) - barw / 2
                x2 = bar_x(groups[j], ib) + barw / 2
                bracket(ax, x1, x2, y0 + li * step, 0.022 * peak, stars(stat["p"] if stat else None))

    if paired:
        for g in groups:
            ax.plot([g, g], [0, peak * 1.02], lw=0.7, ls=(0, (3, 2)), color="0.72", zorder=1)
        ax.text(bar_x(0, (n_arm - 1) / 2, 0), top * 0.995, "presentation\nonly", fontsize=8,
                ha="center", va="top", color="0.15", linespacing=1.25)
        ax.text(bar_x(0, (n_arm - 1) / 2, 1), top * 0.995, "plus objective\nresults", fontsize=8,
                ha="center", va="top", color="0.15", linespacing=1.25)

    ax.set_ylim(0, top)
    ax.set_xlim(-0.58, len(ENDPOINTS) - 1 + 0.58)
    ax.set_xticks(groups)
    ax.set_xticklabels([f"top-{k}" for k in ENDPOINTS])
    ax.set_ylabel("Top-$k$ recall (%)")
    ax.set_title(f"{panel['meta']['panel_letter']}   {panel['meta']['caption']}", loc="left", fontsize=9, pad=6)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    _log_panel(key, arms, means, sds, panel, paired)


def _log_panel(key, arms, means, sds, panel, paired):
    print(f"\n[{key}] values drawn (mean +/- SD over five seeds, percent)")
    for arm in arms:
        row = " | ".join(f"top-{k} {means[arm.key][k]:5.1f} +/- {sds[arm.key][k]:4.1f}" for k in ENDPOINTS)
        print(f"  {arm.label:12s} {row}   [{panel['arms'][arm.key].source}]")
        if paired:
            row = " | ".join(f"top-{k} {means[arm.key + '_w'][k]:5.1f} +/- {sds[arm.key + '_w'][k]:4.1f}"
                             for k in ENDPOINTS)
            print(f"  {'+ objective':12s} {row}   [workup_vs_baseline.json]")
    print("  brackets (paired Wilcoxon on per-case five-seed hit rates, unadjusted p):")
    for k in ENDPOINTS:
        for ia in range(len(arms)):
            for ib in range(ia + 1, len(arms)):
                a, b = arms[ia], arms[ib]
                stat = comparison(panel["arms"][a.key], panel["arms"][b.key], a.key, b.key, k, panel["stored"])
                if stat is None:
                    print(f"    top-{k} {a.key} - {b.key}: no stored or replayable statistic")
                else:
                    print(f"    top-{k} {a.key} - {b.key} = {stat['diff']:+6.1f} pp "
                          f"[{stat['ci'][0]:+6.1f}, {stat['ci'][1]:+6.1f}], p = {stat['p']:.4g}"
                          f"  {stars(stat['p'])}   ({stat['where']})")


# --------------------------------------------------------------------------
# Figure 3 - boundary conditions
# --------------------------------------------------------------------------


def figure3(write: bool):
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.60, wspace=0.30,
                          left=0.105, right=0.985, top=0.945, bottom=0.135)
    _fig3a(fig.add_subplot(gs[0, 0]))
    _fig3b(fig.add_subplot(gs[0, 1]))
    _fig3c(fig.add_subplot(gs[1, :]))
    save(fig, OUTDIR / "figure3_boundary_conditions", write)


def _fig3a(ax):
    """Effect size of MDT against A x 1 on the development and the held-out cases."""
    splits = [("dev41", "dev 41"), ("heldout46", "held-out 46"), ("full87", "full CPC 87")]
    markers = {"1": "o", "3": "s", "5": "^"}
    offsets = {"1": 0.20, "3": 0.0, "5": -0.20}
    primary = rj("holdout46_primary.json")["judges"]["glm_v3"]["splits"]
    print("\n[3a] MDT vs A x 1, primary judge: effect size in pp [95% case-level cluster-bootstrap CI]")
    rows = list(range(len(splits)))[::-1]
    for row, (split, label) in zip(rows, splits):
        for k in ENDPOINTS:
            c = primary[split]["endpoints"][k]["comparisons"]["MDT_vs_Ax1"]
            lo, hi = c["boot95_ci_pct"]
            y = row + offsets[k]
            ax.errorbar(c["mean_diff_pct"], y,
                        xerr=[[c["mean_diff_pct"] - lo], [hi - c["mean_diff_pct"]]],
                        fmt=markers[k], ms=4.0, mfc="0.15", mec="0.15", mew=0.8,
                        ecolor="0.35", elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=5)
            ax.text(max(hi, 0) + 0.7, y, f"p = {c['wilcoxon_p']:.3g}",
                    fontsize=8, va="center", color="0.2")
            print(f"  {label:12s} top-{k}  {c['mean_diff_pct']:+5.1f} pp [{lo:+5.1f}, {hi:+5.1f}]  "
                  f"p = {c['wilcoxon_p']:.4g}  {stars(c['wilcoxon_p'])}")
    handles = [Line2D([], [], marker=markers[k], ls="none", mfc="0.15", mec="0.15", ms=4,
                      label=f"top-{k}") for k in ENDPOINTS]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=1, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.3)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(rows)
    ax.set_yticklabels([s[1] for s in splits])
    ax.set_ylim(-0.62, len(splits) - 0.38)
    ax.set_xlim(-10, 26)
    ax.set_xlabel("MDT − A$\\times$1 (percentage points)")
    ax.set_title("a   Development vs held-out", loc="left", fontsize=9, pad=6)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def _fig3b(ax):
    """Word coverage, which does not depend on the LLM judge."""
    d = rj("word_coverage_5seeds.json")
    strata = [("全部", "All cases\n(n = 364)"), ("症状级金标签", "Symptom-level\n(n = 168)"),
              ("疾病级金标签", "Disease-level\n(n = 196)")]
    arms = [ARMS["Ax1"], ARMS["P"], ARMS["MDT"]]
    pretty = {"Ax1": "A×1", "P": "P", "MDT": "MDT"}
    short = {"Ax1": "ax1", "P": "p", "MDT": "mdt_synth"}
    print("\n[3b] reference-label word coverage of the top-5 list (percent, five seeds)")
    for gi, (zh, _) in enumerate(strata):
        block = d["main"][zh]
        for i, arm in enumerate(arms):
            by_seed = block["by_seed"][pretty[arm.key]]
            seeds = [100.0 * by_seed[f"s{s}"] for s in EXPECTED_SEEDS]
            mean, sd = 100.0 * by_seed["mean"], 100.0 * by_seed["sd"]
            x = gi + 0.26 * (i - 1)
            ax.bar(x, mean, 0.22, color=str(arm.tone), edgecolor="0.15", lw=0.7, zorder=3)
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="0.15", elinewidth=0.8,
                        capsize=1.6, capthick=0.8, zorder=4)
            ax.plot([x] * len(seeds), seeds, marker=".", ms=2.4, ls="none",
                    color="white", mec="0.15", mew=0.3, zorder=5)
            print(f"  {zh:8s} {pretty[arm.key]:5s} {mean:5.1f} +/- {sd:4.1f}   seeds "
                  f"{'  '.join(f'{v:.1f}' for v in seeds)}")
        y0 = max(100.0 * block["by_seed"][pretty[a.key]]["mean"] for a in arms) + 1.6
        for li, (ka, kb) in enumerate([("Ax1", "P"), ("P", "MDT"), ("Ax1", "MDT")]):
            stat = block["comparisons"][f"{short[ka]}_vs_{short[kb]}"]
            ia = [a.key for a in arms].index(ka)
            ib = [a.key for a in arms].index(kb)
            x1 = gi + 0.26 * (ia - 1) + 0.11
            x2 = gi + 0.26 * (ib - 1) - 0.11
            bracket(ax, x1, x2, y0 + li * 5.0, 0.8, stars(stat["wilcoxon_p"]))
            print(f"    {zh} {ka} vs {kb}: {100 * stat['mean_diff']:+5.1f} pp, 95% CI "
                  f"[{100 * stat['boot95_ci'][0]:+.1f}, {100 * stat['boot95_ci'][1]:+.1f}], "
                  f"p = {stat['wilcoxon_p']:.4g}  {stars(stat['wilcoxon_p'])}")
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([s[1] for s in strata])
    ax.set_ylim(0, 66)
    ax.set_ylabel("Word coverage (%)")
    ax.set_title("b   Judge-independent word coverage", loc="left", fontsize=9, pad=6)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=str(a.tone), edgecolor="0.15", lw=0.7,
                             label=a.label.replace("$\\times$", "×")) for a in arms]
                      + [Line2D([], [], marker=".", ls="none", color="white", mec="0.15",
                                label="individual seeds")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=0.9, labelspacing=0.3, borderaxespad=0.0)


def _fig3c(ax):
    """How often the two judges disagree, overall and by stratum."""
    d = rj("judge_agreement_by_stratum.json")
    rows = [("All pairs", d["overall"]),
            ("CPC\n(all seeds)", d["by_dataset"]["CPC"]),
            ("MCR\n(all seeds)", d["by_dataset"]["MCR"]),
            ("ER-Reason\n(seed 1)", d["by_dataset"]["ER"]),
            ("ER-Reason\nsymptom labels", d["er_strata"]["symptom"]),
            ("ER-Reason\ndisease labels", d["er_strata"]["disease"])]
    print("\n[3c] disagreement between the primary and the sensitivity judge (same v3 rules)")
    ys = np.arange(len(rows))[::-1]
    for y, (label, block) in zip(ys, rows):
        rate = 100.0 * block["disagree_rate"]
        a = 100.0 * block["directions"]["glm_yes_ds_no"] / block["n_pairs"]
        ax.barh(y, a, 0.62, color="0.35", edgecolor="0.15", lw=0.7, zorder=3)
        ax.barh(y, rate - a, 0.62, left=a, color="0.88", edgecolor="0.15", lw=0.7, zorder=3)
        ax.text(rate + 0.22, y, f"{rate:.2f}%\n{block['n_disagree']}/{block['n_pairs']}",
                fontsize=8, va="center", color="0.2", linespacing=1.3)
        print(f"  {label.replace(chr(10), ' '):32s} {rate:5.2f}%   {block['n_disagree']:5d}/"
              f"{block['n_pairs']:6d} pairs   primary-yes {block['directions']['glm_yes_ds_no']:5d} / "
              f"sensitivity-yes {block['directions']['glm_no_ds_yes']:4d}")
    ax.axvline(100.0 * d["overall"]["disagree_rate"], color="0.4", lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Judge disagreement rate (%)")
    ax.set_title("c   Two-judge disagreement", loc="left", fontsize=9, pad=6)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="0.35", edgecolor="0.15", lw=0.7,
                             label="primary yes, sensitivity no"),
                       Patch(facecolor="0.88", edgecolor="0.15", lw=0.7,
                             label="primary no, sensitivity yes")],
              loc="lower right", frameon=False, ncol=1,
              borderaxespad=0.3, labelspacing=0.3)


# --------------------------------------------------------------------------
# Figure 4 - controls that isolate the source of the benefit
# --------------------------------------------------------------------------


def figure4(write: bool):
    fig = plt.figure(figsize=(7.2, 7.3))
    gs = fig.add_gridspec(2, 2, hspace=0.64, wspace=0.30,
                          left=0.085, right=0.985, top=0.955, bottom=0.105)
    _fig4a(fig.add_subplot(gs[0, 0]))
    _fig4b(fig.add_subplot(gs[0, 1]))
    _fig4c(fig.add_subplot(gs[1, 0]))
    _fig4d(fig.add_subplot(gs[1, 1]))
    save(fig, OUTDIR / "figure4_controls", write)


def _pair_p(block: dict, a: str, b: str) -> float | None:
    """Wilcoxon p for a vs b from a {pair_name: stats} block (either order)."""
    for name in (f"{a}_vs_{b}", f"{b}_vs_{a}"):
        stat = block.get(name)
        if isinstance(stat, dict) and "wilcoxon_p" in stat:
            return stat["wilcoxon_p"]
    return None


def _pair_topk_p(block: dict, a: str, b: str, k: str) -> float | None:
    """Same, for a {pair_name: {top-k: stats}} block."""
    for name in (f"{a}_vs_{b}", f"{b}_vs_{a}"):
        sub = block.get(name)
        if isinstance(sub, dict):
            stat = sub.get(f"top{k}")
            if isinstance(stat, dict) and "wilcoxon_p" in stat:
                return stat["wilcoxon_p"]
    return None


def _keyed_p(block: dict, key: str) -> float | None:
    stat = block.get(key)
    return stat.get("wilcoxon_p") if isinstance(stat, dict) else None


def _flat_offsets(n, spacing=0.22):
    """Evenly spaced bars, one bar per arm."""
    return [spacing * (i - (n - 1) / 2) for i in range(n)], 0.16


def _paired_offsets(n, pair_step=0.40, in_pair=0.07):
    """n/2 adjacent pairs: the two members of a pair sit closer to each other."""
    half = n // 2
    centres = [pair_step * (i - (half - 1) / 2) for i in range(half)]
    return [c + s * in_pair for c in centres for s in (-1, 1)], 0.12


def _triplet_offsets(n, triplet_step=0.30, spacing=0.155):
    """Two side-by-side triplets, as used for the ER input contrast in Figure 2."""
    half = n // 2
    centres = [triplet_step * (i - (half - 1) / 2) for i in range(half)]
    return [c + spacing * (j - (half - 1) / 2) for c in centres for j in range(half)], 0.115


def _ctl_panel(ax, bars, values, brackets, title, offsets, barw, legend=None,
               ylim_top=None):
    """Grouped bar panel for the Figure 4 controls.

    bars     : [(label, tone)] in left-to-right order within an endpoint group
    values   : {bar index: {endpoint: (mean, sd)}} in percent
    brackets : [(i, j, {endpoint: p})], drawn in bands above each endpoint group
    legend   : [(label, tone)] entries for the key (defaults to bars)
    """
    n = len(bars)
    groups = np.arange(len(ENDPOINTS))
    for j in range(len(ENDPOINTS)):
        for i in range(n):
            style_bar(ax, groups[j] + offsets[i], values[i][ENDPOINTS[j]][0],
                      values[i][ENDPOINTS[j]][1], bars[i][1], barw)

    peak = max(values[i][k][0] + values[i][k][1] for i in range(n) for k in ENDPOINTS)
    bands = bracket_bands([(i, j) for i, j, _ in brackets])
    top = ylim_top if ylim_top is not None else peak * (1.17 + 0.088 * len(bands))
    pad, step = 0.032 * peak, 0.088 * peak
    for j, k in enumerate(ENDPOINTS):
        y0 = max(values[i][k][0] + values[i][k][1] for i in range(n)) + pad
        for li, band in enumerate(bands):
            for i, i2, pmap in brackets:
                if (i, i2) not in band:
                    continue
                bracket(ax, groups[j] + offsets[i] - barw / 2,
                        groups[j] + offsets[i2] + barw / 2,
                        y0 + li * step, 0.022 * peak, stars(pmap.get(k)))

    ax.set_ylim(0, top)
    ax.set_xlim(-0.62, len(ENDPOINTS) - 1 + 0.62)
    ax.set_xticks(groups)
    ax.set_xticklabels([f"top-{k}" for k in ENDPOINTS])
    ax.set_ylabel("Top-$k$ recall (%)")
    ax.set_title(title, loc="left", fontsize=9, pad=6)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=str(tone), edgecolor="0.15", lw=0.7, label=label)
                       for label, tone in (legend if legend is not None else bars)],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.0, labelspacing=0.3, borderaxespad=0.0)

    print(f"\n[{title.strip()}]")
    for k in ENDPOINTS:
        print(f"  top-{k}: " + " | ".join(
            f"{bars[i][0]} {values[i][k][0]:5.1f} +/- {values[i][k][1]:4.1f}" for i in range(n)))
    for i, j, pmap in brackets:
        cells = " | ".join(f"top-{k} p = " + ("n/a" if pmap.get(k) is None
                                              else f"{pmap[k]:.4g} {stars(pmap[k])}")
                           for k in ENDPOINTS)
        print(f"  bracket {bars[i][0]} vs {bars[j][0]}: {cells}")


def _fig4a(ax):
    """Reasoning elicitation: A x 1 against A x 1 + CoT, with P and MDT for reference."""
    d = rj("ax1_cot_cpc.json")
    order = ["Ax1", "Ax1cot", "P", "MDT"]
    bars = [(r"A$\times$1", 0.30), (r"A$\times$1 + CoT", 0.47), ("P", 0.63), ("MDT", 0.80)]
    ms = d["mean_sd"]
    values = {i: {k: (100.0 * ms[a][f"top{k}"]["mean"], 100.0 * ms[a][f"top{k}"]["sd"])
                  for k in ENDPOINTS} for i, a in enumerate(order)}
    cmp = {k: d["caselevel"]["CPC87"]["topk"][k]["comparisons"] for k in ENDPOINTS}
    brackets = [
        (0, 1, {k: _pair_p(cmp[k], "Ax1cot", "Ax1") for k in ENDPOINTS}),
        (1, 3, {k: _pair_p(cmp[k], "Ax1cot", "MDT") for k in ENDPOINTS}),
    ]
    print("\n[4a] ax1_cot_cpc.json (mean_sd, caselevel.CPC87); judge GLM-5.3-flash x v3")
    offsets, barw = _flat_offsets(len(bars))
    _ctl_panel(ax, bars, values, brackets,
               "a   Reasoning elicitation (CPC, n = 87, 5 seeds)", offsets, barw)


def _fig4b(ax):
    """Budget-matched control: five sampled calls under two aggregations."""
    d = rj("ax5_full87.json")["splits"]["full87"]
    order = ["Ax1", "Ax5", "Ax5_sc", "MDT"]
    bars = [(r"A$\times$1 (1 call)", 0.30), (r"A$\times$5, Borda", 0.45),
            (r"A$\times$5, self-cons.", 0.60), ("MDT (6 calls)", 0.80)]
    ms = d["arms"]
    values = {i: {k: (100.0 * ms[a]["mean_sd"][f"top{k}"]["mean"],
                      100.0 * ms[a]["mean_sd"][f"top{k}"]["sd"])
                  for k in ENDPOINTS} for i, a in enumerate(order)}
    cmp = d["comparisons"]
    brackets = [
        (0, 1, {k: _pair_topk_p(cmp, "Ax5", "Ax1", k) for k in ENDPOINTS}),
        (1, 3, {k: _pair_topk_p(cmp, "Ax5", "MDT", k) for k in ENDPOINTS}),
        (2, 3, {k: _pair_topk_p(cmp, "Ax5_sc", "MDT", k) for k in ENDPOINTS}),
    ]
    print("\n[4b] ax5_full87.json splits.full87 (arms.mean_sd, comparisons)")
    offsets, barw = _flat_offsets(len(bars))
    _ctl_panel(ax, bars, values, brackets,
               "b   Compute budget (CPC, n = 87, 5 seeds)", offsets, barw)


def _fig4c(ax):
    """Model family: the same three strategies on deepseek-flash and on qwen3.8-flash."""
    ds_meta = rj("dsflash_family_cpc.json")["meta"]
    d = rj("dsflash_family_cpc.json")["splits"]["full87"]
    order = ["Ax1", "P", "MDT", "Ax1_qwen", "P_qwen", "MDT_qwen"]
    tones = tone_ramp(6, 0.25, 0.85)
    bars = [(f"{name}, deepseek-flash", tones[i]) for i, name in
            enumerate((r"A$\times$1", "P", "MDT"))] + \
           [(f"{name}, qwen3.8-flash", tones[i + 3]) for i, name in
            enumerate((r"A$\times$1", "P", "MDT"))]
    ms = d["arms"]
    values = {i: {k: (100.0 * ms[a]["mean_sd"][f"top{k}"]["mean"],
                      100.0 * ms[a]["mean_sd"][f"top{k}"]["sd"])
                  for k in ENDPOINTS} for i, a in enumerate(order)}
    cmp = d["comparisons"]
    brackets = [
        (1, 2, {k: _pair_topk_p(cmp, "MDT", "P", k) for k in ENDPOINTS}),
        (0, 2, {k: _pair_topk_p(cmp, "MDT", "Ax1", k) for k in ENDPOINTS}),
    ]
    print("\n[4c] dsflash_family_cpc.json splits.full87; both families scored by the same "
          "GLM-5.3-flash x v3 judge")
    offsets, barw = _triplet_offsets(len(bars))
    _ctl_panel(ax, bars, values, brackets,
               f"c   Model family (CPC, n = 87, {ds_meta['n_seeds']} seeds)", offsets, barw)
    ymax = ax.get_ylim()[1] * 0.94
    for g in range(len(ENDPOINTS)):
        ax.plot([g, g], [0, ymax], lw=0.7, ls=(0, (3, 2)), color="0.72", zorder=1)


def _fig4d(ax):
    """Scale: 176B against 2400B for both single-call strategies, plus MDT at 176B."""
    main = rj("scale_comparison.json")["judges"]["GLM"]["main"]
    acc = main["per_seed_acc"]
    order = ["Ax1_176", "Ax1_2400", "P_176", "P_2400"]
    bars = [(r"A$\times$1, 176B", 0.30), (r"A$\times$1, 2400B", 0.45),
            ("P, 176B", 0.60), ("P, 2400B", 0.80)]
    values = {i: {k: (100.0 * acc[a][f"top{k}"]["mean"], 100.0 * acc[a][f"top{k}"]["sd"])
                  for k in ENDPOINTS} for i, a in enumerate(order)}
    cmp = main["comparisons"]
    brackets = [
        (0, 1, {k: _keyed_p(cmp, f"top{k}/Ax1_2400_minus_Ax1_176") for k in ENDPOINTS}),
        (2, 3, {k: _keyed_p(cmp, f"top{k}/P_2400_minus_P_176") for k in ENDPOINTS}),
    ]
    mdt = 100.0 * acc["MDT_176"]["top5"]["mean"]
    ax1_2400 = 100.0 * acc["Ax1_2400"]["top5"]["mean"]
    p_ref = _keyed_p(cmp, "top5/MDT_176_minus_Ax1_2400")
    print(f"\n[4d] scale_comparison.json judges.GLM.main; MDT at 176B top-5 = {mdt:.1f}% vs "
          f"A x 1 at 2400B top-5 = {ax1_2400:.1f}%, p = {p_ref:.4g}")
    offsets, barw = _paired_offsets(len(bars))
    _ctl_panel(ax, bars, values, brackets,
               "d   Model scale (CPC, n = 87, 5 seeds)", offsets, barw, ylim_top=118)
    x5 = float(len(ENDPOINTS) - 1)
    ax.plot([x5 - 0.55, x5 + 0.55], [mdt, mdt], lw=1.1, ls=(0, (5, 2)), color="0.15", zorder=7)
    ax.text(0.985, 0.985,
            f"dashed line: MDT at 176B (6 calls), top-5 = {mdt:.1f}%\n"
            f"A$\\times$1 at 2400B, top-5 = {ax1_2400:.1f}%, p = {p_ref:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color="0.2",
            linespacing=1.35)


# --------------------------------------------------------------------------


def save(fig, stem: Path, write: bool):
    if write:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            target = stem.with_suffix("." + ext)
            fig.savefig(target, bbox_inches="tight", pad_inches=0.04)
            print(f"  wrote {target.relative_to(ROOT)}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Build the manuscript main figures.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default="1,2,3,4", help="figures to build (default 1,2,3,4)")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help="comma-separated arm keys for Figure 2 (keys of ARMS)")
    ap.add_argument("--dry-run", action="store_true", help="load the data but write no files")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.figures.split(",") if s.strip()}
    strategy_keys = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if "1" in wanted:
        print("Figure 1 - study design")
        figure1(not args.dry_run)
    if "2" in wanted:
        print("\nFigure 2 - main results")
        figure2(not args.dry_run, strategy_keys)
    if "3" in wanted:
        print("\nFigure 3 - boundary conditions")
        figure3(not args.dry_run)
    if "4" in wanted:
        print("\nFigure 4 - controls")
        figure4(not args.dry_run)
    print("\n" + ("dry run: no file written" if args.dry_run else f"figures written to {OUTDIR.relative_to(ROOT)}"))


if __name__ == "__main__":
    main()
