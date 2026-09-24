#!/usr/bin/env python3
"""ER 安全分层分析：按 acuity 与 disposition 分层的各臂命中率与配对检验。

回应评审盲点 B1：top-1 miss 不等价——漏诊高危病例的代价远高于低危病例。
本脚本把 ER-Reason 364 例按
  - acuity：高危 = Immediate/Emergent（155），低危 = Urgent/Less Urgent/Non-Urgent（209）
  - disposition：住院 = Admit/Transfer/OR Admit（220），非住院 = Discharge/其他（144）
分层，报告各臂（A×1/P/MDT，5 seeds）分层的 top-1/3/5 命中率（病例级 5-seed
均值），以及高危/住院层内 A×1 vs MDT、A×1 vs P 的配对 Wilcoxon。
命中判定沿用冻结共享缓存；纯离线。

输出：results/er_safety_strata.{json,md}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_erreason import load_cases  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
ER = RESULTS / "topn_erreason"
SEEDS = (1, 2, 3, 4, 5)
OUT_JSON = RESULTS / "er_safety_strata.json"
OUT_MD = RESULTS / "er_safety_strata.md"
HIGH_ACUITY = {"Immediate", "Emergent"}
ADMITTED = {"Admit", "Transfer to Another Facility", "OR Admit"}


def load_uniq(path):
    return cs.load(path)


def build_arms():
    names = {"Ax1": "ax1", "P": "p", "MDT": "mdt_synth"}
    arms = {}
    for arm, stem in names.items():
        arms[arm] = {s: load_uniq(ER / ("ax1.jsonl" if s == 1 else f"s{s}/ax1.jsonl")
                                  if arm == "Ax1" else
                                  ER / ("p.jsonl" if s == 1 else f"s{s}/p.jsonl")
                                  if arm == "P" else
                                  ER / ("mdt_synth.jsonl" if s == 1 else f"s{s}/mdt_synth.jsonl"))
                     for s in SEEDS}
    return arms


def strata_rates(arms, cids, k, cache):
    out = {}
    for arm in ("Ax1", "P", "MDT"):
        per_case = []
        for c in cids:
            vals = []
            for s in SEEDS:
                row = arms[arm][s][c]
                flags = cs.hit_flags(row, cache)
                vals.append(cs.topk(flags, k))
            per_case.append(sum(float(v) for v in vals) / len(vals))
        out[arm] = per_case
    return out


def main():
    cases = load_cases()
    by_id = {c["case_id"]: c for c in cases}
    ids = [c["case_id"] for c in cases]
    high = [i for i in ids if by_id[i].get("acuity") in HIGH_ACUITY]
    low = [i for i in ids if i not in set(high)]
    admit = [i for i in ids if by_id[i].get("disposition") in ADMITTED]
    nonadmit = [i for i in ids if i not in set(admit)]

    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    arms = build_arms()

    out = {}
    L = ["# ER 安全分层：按 acuity 与 disposition 的各臂命中率\n",
         "- 高危 = Immediate/Emergent；低危 = 其余。住院 = Admit/Transfer/OR Admit。"
         "命中率为病例级 5-seed 均值；层内配对检验为配对 Wilcoxon 双侧。"
         "命中判定沿用冻结共享缓存（GLM-5.3-flash × v3）。\n"]
    strata = {"高危（Immediate/Emergent）": high, "低危（其余）": low,
              "住院（Admit/Transfer/OR）": admit, "非住院（Discharge/其他）": nonadmit,
              "全量": ids}
    for name, sids in strata.items():
        r = {k: {arm: 100 * np.mean(v) for arm, v in strata_rates(arms, sids, k, cache).items()}
             for k in (1, 3, 5)}
        out[name] = {"n": len(sids), "rates": r}
        L.append(f"\n## {name}（n={len(sids)}）\n")
        L.append("| 臂 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for arm in ("Ax1", "P", "MDT"):
            L.append(f"| {arm} | {r[1][arm]:.1f} | {r[3][arm]:.1f} | {r[5][arm]:.1f} |")
        # 层内配对检验（top-1 与 top-3，A×1 vs MDT / A×1 vs P）
        L.append("\n层内配对 Wilcoxon（病例级 5-seed 命中率）：\n")
        L.append("| 层 | 对比 | 终点 | 差值 | p |")
        L.append("|---|---|---|---|---|")
        for (a, b) in (("Ax1", "MDT"), ("Ax1", "P")):
            for k in (1, 3):
                ra = {arm: strata_rates(arms, sids, k, cache)[arm] for arm in (a, b)}
                d = np.array(ra[a]) - np.array(ra[b])
                d_nz = d[d != 0]
                if len(d_nz) == 0:
                    continue
                p = wilcoxon(d_nz).pvalue
                L.append(f"| {name} | {a} − {b} | top-{k} | {100*d.mean():+.1f}pp | {p:.4f} |")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
