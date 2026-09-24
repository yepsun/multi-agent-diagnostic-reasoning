#!/usr/bin/env python3
"""病例级聚合口径重算主对比（CPC87 / MCR406）。

取数与判定逻辑复用 recalc_main_judge.py：
- hits[(ds, m, s, cid)] = top5 候选的判官命中 flags（judge_cache_glm_v3.json，
  键 gold[:150]+"||"+cand[:150]）。
- topk(flags, k)：前 k 个候选中任一为 True 即命中；全部缺失判官返回 None。

新口径（投稿用）：
1. 病例级命中率：每个病例 × 方案 × 指标，对 5 seeds 的命中取均值（0-1 连续值），
   跨病例做配对 Wilcoxon 符号秩检验（MDT vs Ax1、MDT vs P、P vs Ax1）。
2. 病例级 cluster bootstrap：按病例重抽样 10,000 次，报告均值差 95% 百分位 CI。
3. 多数决口径：病例在 >=3/5 seeds 命中记为对，做精确 McNemar（补充）。

旧口径（对照用）：跨 seed 合并的配对 McNemar（CPC 435 对、MCR 2030 对），
与 recalc_main_judge.py 输出一致。

用法：python stats_caselevel.py
输出：routing_study/results/stats_caselevel.json + stats_caselevel.md
"""
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
CACHE = B / "judge_cache_glm_v3.json"
OUT_JSON = B / "stats_caselevel.json"
OUT_MD = B / "stats_caselevel.md"
N_BOOT = 10000
RNG = np.random.default_rng(20260917)

J = json.loads(CACHE.read_text())
key = lambda g, c: g[:150] + "||" + c[:150]


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


runs = {}
for s in range(1, 6):
    runs[("cpc", "Ax1", s)] = load(B / f"topn_seeds/Ax1_s{s}.jsonl")
    runs[("cpc", "P", s)] = load(B / f"topn_seeds/P_s{s}.jsonl")
    runs[("cpc", "MDT", s)] = load(
        B / "topn_mdt/synthesis.jsonl" if s == 1
        else B / f"topn_mdt/s{s}/synthesis.jsonl")
    runs[("mcr", "Ax1", s)] = load(
        B / "topn_mcr/ax1.jsonl" if s == 1
        else B / f"topn_mcr_seeds/Ax1_s{s}.jsonl")
    runs[("mcr", "P", s)] = load(
        B / "topn_mcr/p.jsonl" if s == 1
        else B / f"topn_mcr_seeds/P_s{s}.jsonl")
    runs[("mcr", "MDT", s)] = load(
        B / "topn_mcr/mdt_synth.jsonl" if s == 1
        else B / f"topn_mcr_seeds/s{s}/mdt_synth.jsonl")

SPLITS = {
    "CPC87": ("cpc", sorted(runs[("cpc", "Ax1", 1)])),
    "MCR406": ("mcr", sorted(runs[("mcr", "Ax1", 1)])),
}

missing = 0
hits = {}
for (ds, m, s), cases in runs.items():
    for cid, rec in cases.items():
        flags = []
        for c in rec["top5"][:5]:
            k = key(rec["gold"], c)
            if k in J:
                flags.append(bool(J[k]))
            else:
                missing += 1
                flags.append(None)
        hits[(ds, m, s, cid)] = flags
print(f"判官缓存 {CACHE.name}: {len(J)} 对 | 缺失 {missing}")


def topk(t, k):
    f = t[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def mcnemar(b, c):
    """精确二项 McNemar。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def case_rates(ds, m, ids, k):
    """返回 dict cid -> 5-seed 命中率（仅 5 seeds 全部可判定的病例）。"""
    rates = {}
    for cid in ids:
        vals = [topk(hits[(ds, m, s, cid)], k) for s in range(1, 6)]
        if any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / 5.0
    return rates


def case_majority(ds, m, ids, k):
    """返回 dict cid -> 多数决（>=3/5 seeds 命中为 1，否则 0；仅 5 seeds 全部可判定）。"""
    out = {}
    for cid in ids:
        vals = [topk(hits[(ds, m, s, cid)], k) for s in range(1, 6)]
        if any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) >= 3)
    return out


def boot_ci(diffs):
    """病例级 cluster bootstrap：按病例重抽样，均值差 95% 百分位 CI。"""
    d = np.asarray(diffs)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


METHODS = ["Ax1", "P", "MDT"]
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]

results = {}
for name, (ds, ids) in SPLITS.items():
    res = {"n_cases": len(ids), "topk": {}}
    for k in (1, 3, 5):
        rates = {m: case_rates(ds, m, ids, k) for m in METHODS}
        maj = {m: case_majority(ds, m, ids, k) for m in METHODS}
        res["topk"][k] = {
            "case_rate_mean": {
                m: st.mean(rates[m].values()) if rates[m] else None
                for m in METHODS
            },
            "comparisons": {},
        }
        for a, b in PAIRS:
            common = sorted(set(rates[a]) & set(rates[b]))
            ra = np.array([rates[a][c] for c in common])
            rb = np.array([rates[b][c] for c in common])
            diff = ra - rb
            if np.all(diff == 0):
                wp = 1.0
            else:
                wp = float(wilcoxon(ra, rb, zero_method="wilcox").pvalue)
            md, lo, hi = boot_ci(diff)

            cm = sorted(set(maj[a]) & set(maj[b]))
            ao = sum(1 for c in cm if maj[a][c] and not maj[b][c])
            bo = sum(1 for c in cm if maj[b][c] and not maj[a][c])
            mp = mcnemar(ao, bo)

            # 旧口径：跨 seed 合并 McNemar
            pao = pbo = 0
            for s in range(1, 6):
                for c in ids:
                    ha = topk(hits[(ds, a, s, c)], k)
                    hb = topk(hits[(ds, b, s, c)], k)
                    if ha and not hb:
                        pao += 1
                    elif hb and not ha:
                        pbo += 1
            pp = mcnemar(pao, pbo)

            res["topk"][k]["comparisons"][f"{a}_vs_{b}"] = {
                "n_cases": len(common),
                "mean_rate_a": float(ra.mean()) if len(ra) else None,
                "mean_rate_b": float(rb.mean()) if len(rb) else None,
                "mean_diff": md,
                "boot95_ci": [lo, hi],
                "wilcoxon_p": wp,
                "majority": {
                    "a_only": ao, "b_only": bo, "mcnemar_p": mp,
                },
                "pooled_mcnemar": {
                    "a_only": pao, "b_only": pbo, "p": pp,
                },
            }
    results[name] = res

OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"写出 {OUT_JSON}")


def fmt_p(p):
    if p is None:
        return "NA"
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


lines = [
    "# 病例级聚合统计口径（CPC87 / MCR406，GLM 判官缓存 glm_v3）",
    "",
    "- 新口径 1（主）：每病例 5-seed 命中率（0-1），跨病例配对 Wilcoxon 符号秩检验。",
    "- 新口径 2（主）：病例级 cluster bootstrap（按病例重抽样 10,000 次）均值差 95% 百分位 CI。",
    "- 新口径 3（补充）：多数决（>=3/5 seeds 命中记为对）精确 McNemar。",
    "- 旧口径：跨 seed 合并配对 McNemar（CPC 435 对、MCR 2030 对，违反独立性）。",
    "- 「方向」列：旧/新（Wilcoxon）显著性结论是否一致（同向=显著性状态相同且差值方向相同；反转=方向相反）。",
    "",
]
for name, res in results.items():
    lines.append(f"## {name} (n={res['n_cases']} 病例)")
    lines.append("")
    lines.append(
        "| top-k | 对比 | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
        "多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p | 显著性 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in (1, 3, 5):
        for pair, cmp in res["topk"][k]["comparisons"].items():
            a, b = pair.split("_vs_")
            old = cmp["pooled_mcnemar"]
            wp = cmp["wilcoxon_p"]
            mp = cmp["majority"]["mcnemar_p"]
            lo, hi = cmp["boot95_ci"]
            old_sig = old["p"] < 0.05
            new_sig = wp < 0.05
            old_dir = (cmp["pooled_mcnemar"]["a_only"]
                       - cmp["pooled_mcnemar"]["b_only"])
            new_dir = cmp["mean_diff"]
            if (old_dir > 0) != (new_dir > 0) and old_dir != 0 and new_dir != 0:
                verdict = "反转"
            elif old_sig == new_sig:
                verdict = "同向" if old_sig else "均不显著"
            else:
                verdict = ("旧显著→新不显著" if old_sig else "旧不显著→新显著")
            maj = cmp["majority"]
            lines.append(
                f"| top-{k} | {a} vs {b} | "
                f"{cmp['mean_rate_a']*100:.1f}% vs {cmp['mean_rate_b']*100:.1f}% | "
                f"{cmp['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                f"{fmt_p(wp)}{sig(wp)} | "
                f"{maj['a_only']}:{maj['b_only']} p={fmt_p(mp)}{sig(mp)} | "
                f"{old['a_only']}:{old['b_only']} p={fmt_p(old['p'])}{sig(old['p'])} | "
                f"{verdict} |")
        # 各方案命中率
        cm = res["topk"][k]["case_rate_mean"]
        lines.append(
            f"| top-{k} | 方案命中率 | "
            f"Ax1 {cm['Ax1']*100:.1f}% / P {cm['P']*100:.1f}% / MDT {cm['MDT']*100:.1f}% | "
            f"— | — | — | — | — |")
    lines.append("")
    lines.append("（* p<0.05；命中率行为病例级 5-seed 均值，仅供参考）")
    lines.append("")

OUT_MD.write_text("\n".join(lines))
print(f"写出 {OUT_MD}")
