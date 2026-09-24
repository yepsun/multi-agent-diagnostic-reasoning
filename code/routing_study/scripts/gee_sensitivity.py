#!/usr/bin/env python3
"""GEE 敏感性分析：对主分析的 27 个检验（3 数据集 × 3 对比 × 3 终点）补
聚类稳健的广义估计方程（GEE，logistic 链接、exchangeable 工作相关、
按病例聚类）。

回应评审 Major 6：主分析以"病例 5-seed 命中率"为分析单元的配对 Wilcoxon
丢弃平局病例，且不传播 seed 内方差；本脚本改在观测级（case × seed × arm，
每数据集每对比 2·n·5 个观测）拟合 arm 效应，以病例为聚类单元，
报告 log-odds 比 β、OR、稳健 95% CI 与 Wald p，并给出与主分析
Wilcoxon 结论的对照。

判定命中：与主实验完全一致——run 文件 top5 逐个查冻结共享缓存
judge_cache_glm_v3.json（GLM-5.3-flash × v3）；top-k = 前 k 个候选任一 YES。

输出：results/gee_sensitivity.{json,md}。纯离线，不写缓存、不调模型。
"""
import json
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
SEEDS = [1, 2, 3, 4, 5]
OUT_JSON = RESULTS / "gee_sensitivity.json"
OUT_MD = RESULTS / "gee_sensitivity.md"
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]
ENDPOINTS = [1, 3, 5]
DATASETS = ["CPC", "MCR", "ER-Reason"]


def arm_files(ds, arm, seed):
    """返回 (数据集, 臂, seed) 的 run 文件路径。"""
    if ds == "CPC":
        return {"Ax1": RESULTS / "topn_seeds" / f"Ax1_s{seed}.jsonl",
                "P": RESULTS / "topn_seeds" / f"P_s{seed}.jsonl",
                "MDT": (RESULTS / "topn_mdt" / "synthesis.jsonl" if seed == 1
                        else RESULTS / "topn_mdt" / f"s{seed}" / "synthesis.jsonl")}[arm]
    if ds == "MCR":
        return {"Ax1": (RESULTS / "topn_mcr" / "ax1.jsonl" if seed == 1
                        else RESULTS / "topn_mcr_seeds" / f"Ax1_s{seed}.jsonl"),
                "P": (RESULTS / "topn_mcr" / "p.jsonl" if seed == 1
                      else RESULTS / "topn_mcr_seeds" / f"P_s{seed}.jsonl"),
                "MDT": (RESULTS / "topn_mcr" / "mdt_synth.jsonl" if seed == 1
                        else RESULTS / "topn_mcr_seeds" / f"s{seed}" / "mdt_synth.jsonl")}[arm]
    if ds == "ER-Reason":
        base = RESULTS / "topn_erreason" / ("" if seed == 1 else f"s{seed}")
        return {"Ax1": base / "ax1.jsonl",
                "P": base / "p.jsonl",
                "MDT": base / "mdt_synth.jsonl"}[arm]
    raise ValueError(ds)


def build_arms(ds):
    """{arm: {seed: {cid: row}}}，并校验五 seed 病例集一致。"""
    arms = {}
    for arm in ("Ax1", "P", "MDT"):
        arms[arm] = {s: cs.load(arm_files(ds, arm, s)) for s in SEEDS}
    ref = set(arms["Ax1"][SEEDS[0]])
    for arm in arms:
        for s in SEEDS:
            if set(arms[arm][s]) != ref:
                print(f"[警告] {ds} {arm} s{s} 病例集与基准不一致 "
                      f"({len(arms[arm][s])} vs {len(ref)})")
    return arms, sorted(ref)


def observations(arms, ids, arm_a, arm_b, k, cache):
    """观测级 (hit, arm, case_id, seed) 长表；任一臂该 seed 缺行则跳过。"""
    rows = []
    for cid in ids:
        for s in SEEDS:
            for arm, val in ((arm_a, 1), (arm_b, 0)):
                row = arms[arm][s].get(cid)
                if row is None:
                    continue
                flags = cs.hit_flags(row, cache)
                tk = cs.topk(flags, k)
                if tk is None:
                    continue
                rows.append((int(tk), val, cid, s))
    return rows


def fit_gee(rows):
    import numpy as np
    import pandas as pd
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    df = pd.DataFrame(rows, columns=["hit", "arm", "case_id", "seed"])
    try:
        m = GEE.from_formula("hit ~ arm", groups=df["case_id"],
                             data=df, family=Binomial(),
                             cov_struct=Exchangeable())
        r = m.fit()
        beta = float(r.params["arm"])
        se = float(r.bse["arm"])
        p = float(r.pvalues["arm"])
        lo, hi = (float(x) for x in r.conf_int().loc["arm"])
        return {"n_obs": int(len(df)), "n_cases": int(df["case_id"].nunique()),
                "beta": beta, "se": se, "or": float(np.exp(beta)),
                "ci_lo_or": float(np.exp(lo)), "ci_hi_or": float(np.exp(hi)),
                "p": p, "converged": bool(r.converged)}
    except Exception as e:  # 收敛失败等，如实记录
        return {"n_obs": int(len(df)), "n_cases": int(df["case_id"].nunique()),
                "error": f"{type(e).__name__}: {e}"}


def main():
    cache = json.loads(cs.GLM_CACHE.read_text())
    out = {}
    for ds in DATASETS:
        arms, ids = build_arms(ds)
        out[ds] = {}
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            out[ds][key] = {}
            for k in ENDPOINTS:
                rows = observations(arms, ids, a, b, k, cache)
                res = fit_gee(rows)
                res["endpoint"] = f"top{k}"
                out[ds][key][f"top{k}"] = res
                print(f"{ds} {key} top{k}: {res}", flush=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(out)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)


def fmt_p(p):
    if p is None:
        return "NA"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}".rstrip("0").rstrip(".")


def render_md(out):
    L = ["# GEE 敏感性分析：主分析 27 个检验的聚类稳健复核\n",
         "- 观测级 logistic GEE（exchangeable 工作相关），按病例聚类；"
         "每行报告 first-named 臂的 arm 效应（log-odds 比 β、OR 及稳健 95% CI、"
         "Wald p）。命中判定与主实验共用冻结缓存，纯离线计算。\n"]
    for ds in DATASETS:
        L.append(f"\n## {ds}\n")
        L.append("| 对比 | 终点 | n 观测 / 病例 | β (log-OR) | OR [95% CI] | Wald p |")
        L.append("|---|---|---|---|---|---|")
        for key, by_k in out[ds].items():
            a, b = key.split("_vs_")
            for k in ENDPOINTS:
                r = by_k[f"top{k}"]
                if "error" in r:
                    L.append(f"| {a} vs {b} | top-{k} | {r['n_obs']} / "
                             f"{r['n_cases']} | 拟合失败：{r['error']} | — | — |")
                else:
                    L.append(f"| {a} vs {b} | top-{k} | {r['n_obs']} / "
                             f"{r['n_cases']} | {r['beta']:+.3f} | "
                             f"{r['or']:.2f} [{r['ci_lo_or']:.2f}, "
                             f"{r['ci_hi_or']:.2f}] | {fmt_p(r['p'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
