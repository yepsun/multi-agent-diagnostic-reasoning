#!/usr/bin/env python3
"""多数决阈值敏感性：主分析 27 个检验在阈值 t∈{2,3,4,5}/5 下的多数决 McNemar。

回应盲点 A7：主分析以病例级 5-seed 命中率 + 配对 Wilcoxon 为准，多数决
McNemar（≥3/5 命中）为次要口径。本脚本在 t=2/3/4/5 四个阈值下重算全部
27 个主检验（3 数据集 × 3 对比 × 3 终点）的多数决精确 McNemar，检验结论
对阈值选择的稳健性。纯离线，只读 run 文件与冻结缓存。

输出：results/majority_threshold_sensitivity.{json,md}
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
SEEDS = (1, 2, 3, 4, 5)
OUT_JSON = RESULTS / "majority_threshold_sensitivity.json"
OUT_MD = RESULTS / "majority_threshold_sensitivity.md"


def er_path(arm, seed):
    base = ER = RESULTS / "topn_erreason"
    f = {"Ax1": "ax1", "P": "p", "MDT": "mdt_synth"}[arm]
    return base / (f"{f}.jsonl" if seed == 1 else f"s{seed}/{f}.jsonl")


def cpc_path(arm, seed):
    if arm in ("Ax1", "P"):
        return RESULTS / "topn_seeds" / f"{arm}_s{seed}.jsonl"
    return RESULTS / "topn_mdt" / ("synthesis.jsonl" if seed == 1
                                   else f"s{seed}/synthesis.jsonl")


def mcr_path(arm, seed):
    base = RESULTS / "topn_mcr"
    seeds_base = RESULTS / "topn_mcr_seeds"
    return {"Ax1": base / "ax1.jsonl" if seed == 1 else seeds_base / f"Ax1_s{seed}.jsonl",
            "P": base / "p.jsonl" if seed == 1 else seeds_base / f"P_s{seed}.jsonl",
            "MDT": base / "mdt_synth.jsonl" if seed == 1
            else seeds_base / f"s{seed}/mdt_synth.jsonl"}[arm]


DATASETS = {"CPC": cpc_path, "MCR": mcr_path, "ER-Reason": er_path}
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]
THRESHOLDS = (2, 3, 4, 5)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def main():
    cache = json.loads((RESULTS / "judge_cache_glm_v3.json").read_text())
    cs.set_cache(cache)
    out = {}
    L = ["# 多数决阈值敏感性：27 个主检验在 t/5 阈值下的多数决 McNemar\n",
         "- 病例在阈值 t 下记为命中 = 5 个 seed 中至少 t 次命中；列出各阈值下的"
         "不一致对 (a:b) 与精确 p。\n"]
    for ds, pathfn in DATASETS.items():
        arms = {}
        for arm in ("Ax1", "P", "MDT"):
            arms[arm] = {s: cs.load(pathfn(arm, s)) for s in SEEDS}
        cids = sorted(arms["Ax1"][1].keys())
        out[ds] = {}
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            out[ds][key] = {}
            for k in (1, 3, 5):
                row = {"endpoint": f"top{k}"}
                for t in THRESHOLDS:
                    va, vb = [], []
                    for c in cids:
                        ha = sum(1 for s in SEEDS
                                 if cs.topk(cs.hit_flags(arms[a][s][c], cache), k))
                        hb = sum(1 for s in SEEDS
                                 if cs.topk(cs.hit_flags(arms[b][s][c], cache), k))
                        va.append(ha >= t)
                        vb.append(hb >= t)
                    bon = sum(1 for x, y in zip(va, vb) if x and not y)
                    con = sum(1 for x, y in zip(va, vb) if y and not x)
                    p = mcnemar_exact(bon, con)
                    out[ds][key][f"t{t}"] = {"a_only": bon, "b_only": con, "p": p}
                sig = "".join("●" if out[ds][key][f"t{t}"]["p"] < 0.05 else "○"
                              for t in THRESHOLDS)
                row["sig_pattern"] = sig
                L.append(f"- {ds} {key} top-{k}: " + " | ".join(
                    f"t={t}: {out[ds][key][f't{t}']['a_only']}:{out[ds][key][f't{t}']['b_only']}, "
                    f"p={out[ds][key][f't{t}']['p']:.4f}" for t in THRESHOLDS)
                    + f"  [{sig}] (●=p<0.05, 顺序 t=2/3/4/5)")
        L.append("")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
