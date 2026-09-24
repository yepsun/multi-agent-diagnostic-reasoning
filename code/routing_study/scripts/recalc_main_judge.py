#!/usr/bin/env python3
"""判官口径敏感性：用指定判官缓存重算三套主结果（CPC87 / 留出集46 / MCR406），
并与 DS-v3 口径并排。默认载入 judge_cache_glm_v3.json（GLM 主判官口径）。

用法：python recalc_main_judge.py [cache_json]
输出：三套数据集的 A×1/P/MDT × top-1/3/5（5-seed 均值±SD）+ 配对 McNemar。
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "judge_cache_glm_v3.json"
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

dev41 = set(load(B / "topn_ablation/ax1.jsonl"))
cpc_ids = set(runs[("cpc", "Ax1", 1)])
SPLITS = {
    "CPC87": {("cpc", c) for c in cpc_ids},
    "CPC留出46": {("cpc", c) for c in cpc_ids - dev41},
    "MCR406": {("mcr", c) for c in runs[("mcr", "Ax1", 1)]},
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
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


for name, subset in SPLITS.items():
    ds = "cpc" if name.startswith("CPC") else "mcr"
    ids = [c for d, c in subset]
    print(f"\n=== {name} (n={len(ids)}) 5-seed 均值±SD ===")
    for m in ["Ax1", "P", "MDT"]:
        cells = []
        for k in (1, 3, 5):
            per = []
            for s in range(1, 6):
                vals = [topk(hits[(ds, m, s, c)], k) for c in ids]
                vals = [v for v in vals if v is not None]
                per.append(sum(vals) / len(vals))
            cells.append(f"top{k} {st.mean(per)*100:.1f}±{st.stdev(per)*100:.1f}")
        print(f"  {m:4s} " + " | ".join(cells))
    print("  配对 McNemar:")
    for k in (1, 3, 5):
        line = []
        for a, b in [("MDT", "Ax1"), ("MDT", "P"), ("Ax1", "P")]:
            ao = bo = 0
            for s in range(1, 6):
                for c in ids:
                    ha = topk(hits[(ds, a, s, c)], k)
                    hb = topk(hits[(ds, b, s, c)], k)
                    if ha and not hb:
                        ao += 1
                    elif hb and not ha:
                        bo += 1
            line.append(f"{a}-only {ao}:{b}-only {bo} p={mcnemar(ao,bo):.4f}")
        print(f"    top-{k}: " + " | ".join(line))
