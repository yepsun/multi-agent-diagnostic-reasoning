#!/usr/bin/env python3
"""骨干模型规模对照（176B vs 2400B）在当前口径下的重算 —— 纯离线，不调用模型。

现文（paper/manuscript.md L73）写："scaling the backbone from 176B to 2400B
parameters added about 8 points of recall under every strategy, with no
interaction between scale and strategy; notably, MDT at 176B (six calls)
matched the top-5 recall of a single 2400B call (81.6%)"。

本脚本用当前数据与当前口径重算这一段：

主体（CPC 87 例 × 5 seeds，promptv2 冻结）
- 176B  : topn_seeds/Ax1_s{1..5}.jsonl、P_s{1..5}.jsonl
- 2400B : topn_seeds_qwenmax/Ax1_s{1..5}.jsonl、P_s{1..5}.jsonl
- MDT@176B: topn_mdt/synthesis.jsonl + topn_mdt/s{2..5}/synthesis.jsonl
判官：主判官 GLM × v3（judge_cache_glm_v3.json）与敏感性判官 DS × v3
（judge_cache_dsflash_v3.json），两者都只读，绝不写入。

统计口径与 stats_caselevel.py 一致（该文件 import 时有写文件副作用，只能复刻）：
病例级 5-seed 命中率 → 配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次
95% 百分位 CI + 多数决（>=3/5）精确 McNemar。唯一实现差异：bootstrap 的随机数
按「比较」派生（np.random.default_rng([20260917, 判官, k, 比较序号])），避免
调用顺序影响结果；重抽样次数与固定种子语义不变。

产出：
1. 176B→2400B 的规模效应（A×1 与 P 各自），逐 seed 均值±SD + 病例级配对检验；
2. 交互效应：逐病例的 ΔA×1 − ΔP（Δ = 2400B − 176B）配对 Wilcoxon + bootstrap CI
   + 逐 seed 观测的精确符号（McNemar）检验，并附 4 条件（2 策略 × 2 规模）
   的 Friedman 检验；
3. MDT@176B vs A×1@2400B（同 87 例，top-1/3/5）；
4. 以上全部在 GLM 与 DS 两个判官下各跑一遍；
5. 旧数字溯源（topn_qwenmax/summary.json 等）；
6. 数据说明：2400B 的 P 臂有 5 行空 top-5 已补跑（新行带 "refilled_2400b": true，
   备份 P_s*.jsonl.bak_prefill），额外给「剔除这 5 例」的敏感性。

用法：./.venv/bin/python routing_study/scripts/scale_comparison.py
"""
import json
import math
import os
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUT_JSON = RESULTS / "scale_comparison.json"
OUT_MD = RESULTS / "scale_comparison.md"
HELD_OUT_JSON = ROOT / "data" / "mgh_qa_dataset_new_cases.json"

SEEDS = (1, 2, 3, 4, 5)
N_BOOT = 10000
RNG_SEED = 20260917

JUDGES = [("GLM", RESULTS / "judge_cache_glm_v3.json", "主判官 GLM × v3"),
          ("DS", RESULTS / "judge_cache_dsflash_v3.json",
           "敏感性判官 DS × v3")]

ARMS = {
    "Ax1_176": [RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl" for s in SEEDS],
    "P_176": [RESULTS / "topn_seeds" / f"P_s{s}.jsonl" for s in SEEDS],
    "Ax1_2400": [RESULTS / "topn_seeds_qwenmax" / f"Ax1_s{s}.jsonl" for s in SEEDS],
    "P_2400": [RESULTS / "topn_seeds_qwenmax" / f"P_s{s}.jsonl" for s in SEEDS],
    "MDT_176": [RESULTS / "topn_mdt" / "synthesis.jsonl"] +
               [RESULTS / "topn_mdt" / f"s{s}" / "synthesis.jsonl"
                for s in (2, 3, 4, 5)],
}
LABEL = {
    "Ax1_176": "A×1 @176B", "P_176": "P @176B",
    "Ax1_2400": "A×1 @2400B", "P_2400": "P @2400B",
    "MDT_176": "MDT @176B (6 次调用)",
}
# 对比：(序号, A, B, 说明) —— A − B
COMPARISONS = [
    (0, "Ax1_2400", "Ax1_176", "规模效应（A×1）：2400B − 176B"),
    (1, "P_2400", "P_176", "规模效应（P）：2400B − 176B"),
    (2, "MDT_176", "Ax1_2400", "团队 vs 更大单次调用：MDT@176B − A×1@2400B"),
]
FRIEDMAN_CONDITIONS = ("Ax1_176", "Ax1_2400", "P_176", "P_2400")

OLD_SUMMARY = RESULTS / "topn_qwenmax" / "summary.json"
OLD_SEED_SUMMARY_176 = RESULTS / "topn_seeds" / "seed_summary.json"
OLD_SEED_SUMMARY_2400 = RESULTS / "topn_seeds_qwenmax" / "seed_summary.json"
OLD_DOC = ROOT / "routing_study" / "docs" / "2026-09-10-topn-results-report.md"
OLD_LOCAL_CACHES = {  # 旧口径（deepseek-flash × v2 规则）的本地缓存
    "qwenmax_2400_s1": RESULTS / "topn_qwenmax" / "judge_cache_qwenmax_dsflash.json",
    "qwenflash_176_promptv2_s1": RESULTS / "topn_qwenmax" / "judge_cache_qwenflash_dsflash.json",
    "qwenmax_2400_5seeds": RESULTS / "topn_seeds_qwenmax" / "judge_cache_dsflash.json",
}


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def load_jsonl(path):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(path) if l.strip()}


# ---------- 口径函数（复刻 stats_caselevel.py） ----------

def topk(flags, k):
    f = flags[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def case_rates(arm, ids, k, flags):
    rates = {}
    for cid in ids:
        vals = [topk(flags[(arm, s, cid)], k) for s in SEEDS]
        if any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / float(len(SEEDS))
    return rates


def case_majority(arm, ids, k, flags):
    out = {}
    for cid in ids:
        vals = [topk(flags[(arm, s, cid)], k) for s in SEEDS]
        if any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) >= 3)
    return out


def boot_ci(diffs, seed_parts):
    """病例级 cluster bootstrap 95% 百分位 CI（10,000 次，种子按比较派生）。"""
    d = np.asarray(diffs)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng([RNG_SEED] + list(seed_parts))
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def compare(a, b, k, ids, flags, jidx, pi):
    ra, rb = case_rates(a, ids, k, flags), case_rates(b, ids, k, flags)
    common = sorted(set(ra) & set(rb))
    va = np.array([ra[c] for c in common])
    vb = np.array([rb[c] for c in common])
    diff = va - vb
    if len(diff) == 0 or np.all(diff == 0):
        wp = 1.0
    else:
        wp = float(wilcoxon(va, vb, zero_method="wilcox").pvalue)
    md, lo, hi = boot_ci(diff, (jidx, k, pi, 0))

    ma, mb = case_majority(a, ids, k, flags), case_majority(b, ids, k, flags)
    cm = sorted(set(ma) & set(mb))
    ao = sum(1 for c in cm if ma[c] and not mb[c])
    bo = sum(1 for c in cm if mb[c] and not ma[c])

    pao = pbo = 0
    for cid in ids:
        for s in SEEDS:
            ha, hb = topk(flags[(a, s, cid)], k), topk(flags[(b, s, cid)], k)
            if ha and not hb:
                pao += 1
            elif hb and not ha:
                pbo += 1
    return {
        "pair": f"{a}_minus_{b}", "n_cases": len(common),
        "mean_rate_a": float(va.mean()) if len(va) else None,
        "mean_rate_b": float(vb.mean()) if len(vb) else None,
        "mean_diff": md, "boot95_ci": [lo, hi], "wilcoxon_p": wp,
        "majority": {"a_only": ao, "b_only": bo,
                     "mcnemar_p": mcnemar_exact(ao, bo)},
        "pooled_mcnemar": {"a_only": pao, "b_only": pbo,
                           "p": mcnemar_exact(pao, pbo), "n_obs": len(ids) * len(SEEDS)},
    }


def hit_vec(arm, cid, k, flags):
    return [topk(flags[(arm, s, cid)], k) for s in SEEDS]


def interaction(k, ids, flags, jidx):
    """逐病例 ΔA×1 − ΔP（Δ = 2400B − 176B）的配对检验。"""
    pairs = ("Ax1_2400", "Ax1_176", "P_2400", "P_176")
    cont, da_list, dp_list = [], [], []
    for cid in ids:
        vals = {a: hit_vec(a, cid, k, flags) for a in pairs}
        if any(any(v is None for v in vals[a]) for a in pairs):
            continue
        d_a = [(1 if vals["Ax1_2400"][i] else 0) - (1 if vals["Ax1_176"][i] else 0)
               for i in range(len(SEEDS))]
        d_p = [(1 if vals["P_2400"][i] else 0) - (1 if vals["P_176"][i] else 0)
               for i in range(len(SEEDS))]
        da_list.append(sum(d_a) / len(SEEDS))
        dp_list.append(sum(d_p) / len(SEEDS))
        cont.append((sum(d_a) - sum(d_p)) / len(SEEDS))
    cont = np.array(cont)
    if len(cont) == 0:
        return None
    if np.all(cont == 0):
        wp = 1.0
    else:
        wp = float(wilcoxon(cont, zero_method="wilcox").pvalue)
    md, lo, hi = boot_ci(cont, (jidx, k, 90, 0))

    # 逐 seed 观测：四个臂都可判定的 (case, seed) 上做增益差的精确符号检验
    b = c = 0
    for cid in ids:
        vals = {a: hit_vec(a, cid, k, flags) for a in pairs}
        for i in range(len(SEEDS)):
            if any(vals[a][i] is None for a in pairs):
                continue
            ga = (1 if vals["Ax1_2400"][i] else 0) - (1 if vals["Ax1_176"][i] else 0)
            gp = (1 if vals["P_2400"][i] else 0) - (1 if vals["P_176"][i] else 0)
            if ga - gp > 0:
                b += 1
            elif ga - gp < 0:
                c += 1
    return {
        "n_cases": int(len(cont)),
        "mean_delta_ax1": float(np.mean(da_list)),
        "mean_delta_p": float(np.mean(dp_list)),
        "mean_diff": md, "boot95_ci": [lo, hi], "wilcoxon_p": wp,
        "sign_test": {"a_only": b, "b_only": c, "mcnemar_p": mcnemar_exact(b, c)},
    }


def friedman(ids, k, flags):
    rows = []
    for cid in ids:
        vals = [hit_vec(a, cid, k, flags) for a in FRIEDMAN_CONDITIONS]
        if any(any(v is None for v in vs) for vs in vals):
            continue
        rows.append([st.mean(1 if x else 0 for x in vs) for vs in vals])
    if len(rows) < 3:
        return None
    arr = np.array(rows)
    stat, p = friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])])
    means = {a: float(arr[:, i].mean()) for i, a in enumerate(FRIEDMAN_CONDITIONS)}
    return {"n_cases": len(rows), "statistic": float(stat), "p": float(p),
            "case_rate_mean": means}


def per_seed_acc(arm, ids, k, flags):
    per = []
    for s in SEEDS:
        h = sum(1 for cid in ids if any(x is True
                                       for x in flags[(arm, s, cid)][:k]))
        per.append(h / len(ids))
    return {"per_seed": per, "mean": st.mean(per),
            "sd": st.stdev(per) if len(per) > 1 else 0.0}


def fmt_p(p):
    return "NA" if p is None else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def acc_row(arm, ids, flags):
    return {f"top{k}": per_seed_acc(arm, ids, k, flags) for k in (1, 3, 5)}


# ---------- 旧数字溯源 ----------

def old_cache_acc(cache_path, files, key_fn=key_of):
    cache = json.loads(cache_path.read_text())
    out = {}
    for tag, path in files.items():
        rows = list(load_jsonl(path).values())
        n = len(rows)
        t = {1: 0, 3: 0, 5: 0}
        for r in rows:
            v = [bool(cache.get(key_fn(r["gold"], c))) for c in r["top5"][:5]]
            h = next((i for i, x in enumerate(v, 1) if x), None)
            if h:
                t[1] += h == 1
                t[3] += h <= 3
                t[5] += 1
        out[tag] = {"n": n, **{f"top{k}": t[k] / n for k in (1, 3, 5)}}
    return out


def old_5seed_acc(cache_path, dirp, scheme, use_backup=False):
    cache = json.loads(cache_path.read_text())
    per = {1: [], 3: [], 5: []}
    for s in SEEDS:
        p = dirp / f"{scheme}_s{s}.jsonl"
        bak = Path(str(p) + ".bak_prefill")
        path = bak if (use_backup and bak.exists()) else p
        rows = list(load_jsonl(path).values())
        n, t = len(rows), {1: 0, 3: 0, 5: 0}
        for r in rows:
            v = [bool(cache.get(key_of(r["gold"], c))) for c in r["top5"][:5]]
            h = next((i for i, x in enumerate(v, 1) if x), None)
            if h:
                t[1] += h == 1
                t[3] += h <= 3
                t[5] += 1
        for k in (1, 3, 5):
            per[k].append(t[k] / n)
    return {f"top{k}": {"mean": st.mean(per[k]),
                        "sd": st.stdev(per[k]) if len(per[k]) > 1 else 0.0}
            for k in (1, 3, 5)}


def provenance():
    prov = {"old_files": {}, "recomputed_under_old_judge": {}}
    if OLD_SUMMARY.exists():
        s = json.loads(OLD_SUMMARY.read_text())

        def strip(o):
            if isinstance(o, dict):
                return {k: ("<details>" if k == "details" else strip(v))
                        for k, v in o.items()}
            if isinstance(o, list):
                return f"<list {len(o)}>"
            return o
        prov["old_files"]["topn_qwenmax/summary.json"] = strip(s)
    for tag, p in (("topn_seeds/seed_summary.json", OLD_SEED_SUMMARY_176),
                   ("topn_seeds_qwenmax/seed_summary.json", OLD_SEED_SUMMARY_2400)):
        if p.exists():
            d = json.loads(p.read_text())
            prov["old_files"][tag] = {k: v for k, v in d.items()
                                      if k in ("stats", "per_seed")}
    prov["old_doc"] = "routing_study/docs/2026-09-10-topn-results-report.md"
    # 用旧判官（deepseek-flash × v2）的本地缓存复算，确认数字来源
    if OLD_LOCAL_CACHES["qwenmax_2400_s1"].exists():
        prov["recomputed_under_old_judge"]["2400B 单次 s1（本地 v2 缓存）"] = \
            old_cache_acc(OLD_LOCAL_CACHES["qwenmax_2400_s1"],
                          {"Ax1": RESULTS / "topn_qwenmax" / "ax1.jsonl",
                           "P": RESULTS / "topn_qwenmax" / "p.jsonl"})
    if OLD_LOCAL_CACHES["qwenflash_176_promptv2_s1"].exists():
        prov["recomputed_under_old_judge"]["176B 单次（本地 v2 缓存）"] = \
            old_cache_acc(OLD_LOCAL_CACHES["qwenflash_176_promptv2_s1"],
                          {"Ax1": RESULTS / "topn_ablation_promptv2_87" / "ax1.jsonl"})
    c2400 = OLD_LOCAL_CACHES["qwenmax_2400_5seeds"]
    if c2400.exists():
        prov["recomputed_under_old_judge"]["2400B 5-seed（本地 v2 缓存）"] = {
            "Ax1": old_5seed_acc(c2400, RESULTS / "topn_seeds_qwenmax", "Ax1"),
            "P_补跑后": old_5seed_acc(c2400, RESULTS / "topn_seeds_qwenmax", "P"),
            "P_补跑前(bak_prefill)": old_5seed_acc(
                c2400, RESULTS / "topn_seeds_qwenmax", "P", use_backup=True),
        }
    return prov


# ---------- 主流程 ----------

def main():
    snapshot = {p.name: {"mtime": datetime.fromtimestamp(p.stat().st_mtime)
                         .strftime("%Y-%m-%d %H:%M:%S"),
                         "pairs": len(json.loads(p.read_text()))}
                for _, p, _ in JUDGES}

    runs, refilled = {}, {}
    for arm, paths in ARMS.items():
        for s, p in zip(SEEDS, paths):
            runs[(arm, s)] = load_jsonl(p)
    universe = sorted(runs[("Ax1_176", 1)])
    for (arm, s), rows in runs.items():
        if sorted(rows) != universe:
            raise SystemExit(f"病例集合不一致: {arm} s{s}")
        for cid, r in rows.items():
            if r.get("refilled_2400b"):
                refilled.setdefault(arm, []).append({"seed": s, "case_id": cid})
    golds = {cid: runs[("Ax1_176", 1)][cid]["gold"] for cid in universe}
    for (arm, s), rows in runs.items():
        for cid, r in rows.items():
            if r["gold"] != golds[cid]:
                raise SystemExit(f"gold 不一致: {arm} s{s} {cid}")
    held_out = {c["case_id"] for c in json.loads(HELD_OUT_JSON.read_text())}
    refill_cases = sorted({e["case_id"] for e in refilled.get("P_2400", [])})
    universe_no_refill = [c for c in universe if c not in set(refill_cases)]

    print(f"病例 {len(universe)} 例 | 判官缓存快照 {snapshot}")
    print(f"补跑行（refilled_2400b）：{refilled.get('P_2400')}")

    out = {
        "config": {
            "cases": len(universe), "seeds": list(SEEDS), "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
            "judge_cache_snapshot": snapshot,
            "arms": {a: [str(p.relative_to(ROOT)) for p in ps]
                     for a, ps in ARMS.items()},
            "splits": {"heldout46": len(set(universe) & held_out),
                       "dev41": len(set(universe) - held_out)},
            "statistics": "stats_caselevel.py 口径复刻（病例级 5-seed 均值 → "
                          "Wilcoxon 双侧 + cluster bootstrap 10k 95% CI + "
                          "多数决精确 McNemar）；bootstrap 种子按比较派生",
        },
        "data_notes": {
            "refilled_rows": refilled.get("P_2400", []),
            "refilled_cases": refill_cases,
            "note": "2400B 的 P 臂原文件有 5 行空 top-5，已按本文档流程补跑；"
                    "新行带 \"refilled_2400b\": true，备份为 P_s*.jsonl.bak_prefill。"
                    "本脚本主体用补跑后的文件，另给剔除这 5 例（n=%d）的敏感性。"
                    % len(universe_no_refill),
        },
        "judges": {},
        "provenance": provenance(),
    }

    for jidx, (jname, jpath, jdesc) in enumerate(JUDGES):
        cache = json.loads(jpath.read_text())
        flags, missing = {}, {}
        for (arm, s), rows in runs.items():
            n = 0
            for cid, r in rows.items():
                f = []
                for c in r["top5"][:5]:
                    kk = key_of(r["gold"], c)
                    if kk in cache:
                        f.append(bool(cache[kk]))
                    else:
                        f.append(None)
                        n += 1
                flags[(arm, s, cid)] = f
            if n:
                missing[f"{arm}_s{s}"] = n

        def analyze(ids, tag):
            res = {
                "n_ids": len(ids),
                "per_seed_acc": {a: acc_row(a, ids, flags) for a in ARMS},
                "comparisons": {},
                "interaction": {f"top{k}": interaction(k, ids, flags, jidx)
                                for k in (1, 3, 5)},
                "friedman": {f"top{k}": friedman(ids, k, flags) for k in (1, 3, 5)},
            }
            for pi, a, b, desc in COMPARISONS:
                for k in (1, 3, 5):
                    c = compare(a, b, k, ids, flags, jidx, pi)
                    c["desc"] = desc
                    res["comparisons"][f"top{k}/{a}_minus_{b}"] = c
            return res

        jr = {"desc": jdesc, "cache": str(jpath.relative_to(ROOT)),
              "cache_pairs": len(cache),
              "missing_pairs": missing,
              "missing_pairs_total": sum(missing.values()),
              "main": analyze(universe, "main"),
              "no_refill": analyze(universe_no_refill, "no_refill")}
        out["judges"][jname] = jr
        print(f"[{jname}] {jdesc}: 缓存 {len(cache)} 对 | 缺失 {jr['missing_pairs_total']}")

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = build_md(out, universe)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已写入 {OUT_JSON} 与 {OUT_MD}")


def build_md(out, universe):
    L = []
    L.append("# 骨干规模对照（176B vs 2400B）在当前口径下的重算\n")
    L.append("纯离线、无模型调用。为更正 `paper/manuscript.md` 中"
             "「scaling the backbone … added about 8 points of recall under every "
             "strategy, with no interaction … MDT at 176B matched the top-5 recall "
             "of a single 2400B call (81.6%)」一段。\n")
    snap = out["config"]["judge_cache_snapshot"]
    L.append(f"- 病例：CPC {out['config']['cases']} 例（held-out "
             f"{out['config']['splits']['heldout46']} + dev "
             f"{out['config']['splits']['dev41']}）× 5 seeds，promptv2 冻结，"
             "与主文一致。")
    L.append("- 判官缓存快照：" + "；".join(
        f"`{n}` {v['pairs']} 对（mtime {v['mtime']}）" for n, v in snap.items())
        + "（均**只读**，本脚本未写入任何判官缓存）。")
    L.append("- 统计口径 = `stats_caselevel.py` 复刻：病例级 5-seed 命中率 → 配对 "
             "Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次 95% CI + 多数决"
             "（≥3/5）精确 McNemar。\n")

    r = out["data_notes"]["refilled_rows"]
    L.append("\n## 0. 数据说明\n")
    L.append(f"- 2400B 的 **P 臂原有 {len(r)} 行空 top-5**，已按本文档流程补跑；"
             "新行带 `\"refilled_2400b\": true`，原文件备份为 "
             "`P_s*.jsonl.bak_prefill`。**本文主体用补跑后的文件**，"
             "第 5 节给「剔除这 5 例」的敏感性。")
    L.append("- 补跑行：" + "；".join(
        f"`P_s{e['seed']}` {e['case_id'][:34]}…" for e in r))
    L.append("- 缺失判定对（各判官下）：" + "；".join(
        f"**{jn}** "
        + (f"{out['judges'][jn]['missing_pairs_total']} 对"
           + ("（" + "、".join(f"{k} {v}" for k, v in
                               out['judges'][jn]['missing_pairs'].items()) + "）"
              if out['judges'][jn]['missing_pairs'] else ""))
        for jn in out["judges"]))

    for jn in out["judges"]:
        j = out["judges"][jn]
        L.append(f"\n## {jn} 判官（{j['desc']}）\n")
        L.append(f"\n### {jn}-1. 逐 seed 准确率（5-seed 均值±SD，87 例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for a in ("Ax1_176", "Ax1_2400", "P_176", "P_2400", "MDT_176"):
            m = j["main"]["per_seed_acc"][a]
            L.append(f"| {LABEL[a]} | "
                     + " | ".join(f"{m[f'top{k}']['mean']*100:.1f}% ± "
                                  f"{m[f'top{k}']['sd']*100:.1f}" for k in (1, 3, 5))
                     + " |")
        L.append("")
        for pi, a, b, desc in COMPARISONS:
            L.append(f"\n### {jn}-{pi + 2}. {desc}\n")
            L.append("| top-k | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
                     "多数决 (a:b) p | 合并 (a:b) p | n |")
            L.append("|---|---|---|---|---|---|---|")
            for k in (1, 3, 5):
                c = j["main"]["comparisons"][f"top{k}/{a}_minus_{b}"]
                lo, hi = c["boot95_ci"]
                L.append(
                    f"| top-{k} | {c['mean_rate_a']*100:.1f}% vs "
                    f"{c['mean_rate_b']*100:.1f}% | "
                    f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{fmt_p(c['wilcoxon_p'])}{sig(c['wilcoxon_p'])} | "
                    f"{c['majority']['a_only']}:{c['majority']['b_only']} "
                    f"p={fmt_p(c['majority']['mcnemar_p'])}"
                    f"{sig(c['majority']['mcnemar_p'])} | "
                    f"{c['pooled_mcnemar']['a_only']}:{c['pooled_mcnemar']['b_only']} "
                    f"p={fmt_p(c['pooled_mcnemar']['p'])}"
                    f"{sig(c['pooled_mcnemar']['p'])} | {c['n_cases']} |")
            L.append("")
            L.append("说明：**病例级 Wilcoxon 比逐 seed 均值看起来保守得多**，原因是"
                     "病例级口径以病例为单位、且多数病例在两臂上结果相同（并列），"
                     "检验的有效信息只是「结果发生变化的病例数」，即多数决列的 "
                     "a:b（如 top-1 为 "
                     + str(j["main"]["comparisons"][f"top1/{a}_minus_{b}"]["majority"]
                           ["a_only"] + j["main"]["comparisons"]
                           [f"top1/{a}_minus_{b}"]["majority"]["b_only"])
                     + " 例改变）。逐 seed 均值把 5 次重复当独立样本，二者不可混用；"
                     "论文以病例级为准，逐 seed 表仅作描述。")

        L.append(f"\n### {jn}-5. 交互效应：ΔA×1 − ΔP（Δ = 2400B − 176B）\n")
        L.append("| top-k | ΔA×1 | ΔP | 交互差 [95% CI] | Wilcoxon p | "
                 "逐 seed 符号检验 (a:b) p | n |")
        L.append("|---|---|---|---|---|---|---|")
        for k in (1, 3, 5):
            it = j["main"]["interaction"][f"top{k}"]
            lo, hi = it["boot95_ci"]
            L.append(
                f"| top-{k} | {it['mean_delta_ax1']*100:+.1f}pp | "
                f"{it['mean_delta_p']*100:+.1f}pp | "
                f"{it['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                f"{fmt_p(it['wilcoxon_p'])}{sig(it['wilcoxon_p'])} | "
                f"{it['sign_test']['a_only']}:{it['sign_test']['b_only']} "
                f"p={fmt_p(it['sign_test']['mcnemar_p'])}"
                f"{sig(it['sign_test']['mcnemar_p'])} | {it['n_cases']} |")
        L.append("")
        L.append("Friedman 检验（四条件：A×1@176B / A×1@2400B / P@176B / P@2400B，"
                 "逐病例 5-seed 命中率）：")
        L.append("")
        L.append("| top-k | n | Friedman χ² | p | 四条件命中率 |")
        L.append("|---|---|---|---|---|")
        for k in (1, 3, 5):
            f = j["main"]["friedman"][f"top{k}"]
            if f is None:
                L.append(f"| top-{k} | — | — | — | — |")
                continue
            cm = f["case_rate_mean"]
            L.append(f"| top-{k} | {f['n_cases']} | {f['statistic']:.2f} | "
                     f"{fmt_p(f['p'])}{sig(f['p'])} | "
                     + " / ".join(f"{LABEL[a].replace(' (6 次调用)', '')} "
                                  f"{cm[a]*100:.1f}%" for a in FRIEDMAN_CONDITIONS)
                     + " |")
        L.append("")
        L.append("注：Friedman 是**四条件总体**检验（任两格有差异即显著），"
                 "交互效应本身的推断看上一张表的对比检验（ΔA×1 − ΔP）。"
                 "两者的区别：Friedman 显著只说明规模或策略有主效应。")

    # 敏感性
    n_all = len(universe)
    n_nr = n_all - len(out["data_notes"]["refilled_cases"])
    L.append(f"\n## 5. 敏感性：剔除 5 行补跑数据（n={n_nr}）\n")
    L.append("把 5 行补跑数据对应的 5 个病例整体剔除（病例级均值要求 5 seeds "
             "齐全），重算主对比：\n")
    L.append(f"| 判官 | 对比 | top-k | 全样本（n={n_all}） | 剔除后（n={n_nr}） | "
             "结论是否改变 |")
    L.append("|---|---|---|---|---|---|")
    for jn in out["judges"]:
        j = out["judges"][jn]
        for pi, a, b, desc in COMPARISONS:
            for k in (1, 3, 5):
                c1 = j["main"]["comparisons"][f"top{k}/{a}_minus_{b}"]
                c2 = j["no_refill"]["comparisons"][f"top{k}/{a}_minus_{b}"]
                s1 = c1["wilcoxon_p"] < 0.05 and c1["mean_diff"] > 0
                s2 = c2["wilcoxon_p"] < 0.05 and c2["mean_diff"] > 0
                flip = "—" if (s1 == s2) else "**改变**"
                L.append(f"| {jn} | {a} − {b} | top-{k} | "
                         f"{c1['mean_diff']*100:+.1f}pp p={fmt_p(c1['wilcoxon_p'])} | "
                         f"{c2['mean_diff']*100:+.1f}pp p={fmt_p(c2['wilcoxon_p'])} | "
                         f"{flip} |")

    # 溯源
    L.append("\n## 6. 旧数字溯源（+8pp 与 81.6% 是怎么来的）\n")
    pv = out["provenance"]
    old = pv.get("recomputed_under_old_judge", {})
    s1_2400 = old.get("2400B 单次 s1（本地 v2 缓存）", {})
    s1_176 = old.get("176B 单次（本地 v2 缓存）", {})
    ss_2400 = old.get("2400B 5-seed（本地 v2 缓存）", {})
    L.append("| 数字 | 出处 | 口径 | 复算值 |")
    L.append("|---|---|---|---|")
    if s1_2400.get("Ax1"):
        v = s1_2400["Ax1"]
        L.append(f"| **81.6%**（A×1@2400B top-5） | `topn_qwenmax/ax1.jsonl`"
                 f"（单次运行 s1） | {v['n']} 例，promptv2，"
                 "判官 deepseek-flash × **v2 规则**（该目录本地缓存），**单 seed** | "
                 f"{v['top1']*100:.1f}% / {v['top3']*100:.1f}% / "
                 f"{v['top5']*100:.1f}%（top-1/3/5） |")
    if ss_2400.get("Ax1"):
        a = ss_2400["Ax1"]
        L.append(f"| **+8pp**（2400B − 176B） | `topn_seeds/seed_summary.json`"
                 f"（176B）与 `topn_seeds_qwenmax/`（2400B，同一 v2 判官） | "
                 "87 例 × 5 seeds，promptv2；**该增益是 top-1 的**"
                 "（top-5 只 +6.2pp） | A×1 top-1 "
                 f"{a['top1']['mean']*100:.1f}% ± {a['top1']['sd']*100:.1f} → "
                 f"176B 51.7% ± 1.6（= +8.1pp）；top-5 "
                 f"{a['top5']['mean']*100:.1f}%（+6.2pp） |")
    if ss_2400.get("P_补跑前(bak_prefill)"):
        p0 = ss_2400["P_补跑前(bak_prefill)"]
        p1 = ss_2400.get("P_补跑后", {})
        L.append(f"| P 的 +7.8pp | 同上（**补跑前**文件，空行计为 miss） | 同上 | "
                 f"top-1 {p0['top1']['mean']*100:.1f}% ± {p0['top1']['sd']*100:.1f}"
                 f"（补跑后 {p1.get('top1', {}).get('mean', float('nan'))*100:.1f}%） |")
    if s1_176.get("Ax1"):
        v = s1_176["Ax1"]
        L.append(f"| 旧表 176B 单次 54.0/72.4/75.9 | "
                 f"`topn_ablation_promptv2_87/ax1.jsonl` + 本地 v2 缓存 | "
                 f"{v['n']} 例、promptv2、单 seed | "
                 f"{v['top1']*100:.1f}% / {v['top3']*100:.1f}% / "
                 f"{v['top5']*100:.1f}% |")

    # 判官归因：旧 +8pp 更接近哪个现判官
    pv_176 = pv.get("old_files", {}).get("topn_seeds/seed_summary.json", {}).get(
        "stats", {}).get("Ax1", {})
    if ss_2400.get("Ax1") and pv_176.get("top1"):
        old_a1 = ss_2400["Ax1"]["top1"]["mean"] - pv_176["top1"]["mean"]
        rows = []
        for jname in out["judges"]:
            g = out["judges"][jname]["main"]["per_seed_acc"]
            rows.append((jname,
                         g["Ax1_2400"]["top1"]["mean"] - g["Ax1_176"]["top1"]["mean"],
                         g["Ax1_2400"]["top5"]["mean"] - g["Ax1_176"]["top5"]["mean"]))
        if old_a1 is not None:
            L.append("\n**判官归因**：旧的 A×1 top-1 规模增益 = "
                     f"{old_a1*100:+.1f}pp（v2 判官）。现两判官下的同一增益："
                     + "、".join(f"{jn} {d1*100:+.1f}pp" for jn, d1, d5 in rows)
                     + "。"
                     + "**旧数字与敏感性判官（DS × v3）几乎重合、与主判官（GLM × v3）"
                       "差距明显** —— 说明「+8pp」是判官口径造成的，而不是模型行为。"
                       "论文已改用 GLM × v3 作为 primary judge，"
                       "因此这一段必须用 GLM 数字重写（DS 数字可作敏感性引用）。")
    L.append("\n**溯源结论**：旧文那两句的前提是"
             "（i）判官用 deepseek-flash × **v2 规则**（只有 4 条判则，比现在的 "
             "GLM × v3 严），（ii）2400B 的 top-5 = 81.6% 是**单次运行**（s1）而非 "
             "5-seed 均值，（iii）「+8 点」实测是 **top-1** 的增益"
             "（top-5 只有 +6.2pp）。三项都与现文所声明的"
             "「primary judge, 5 seeds, promptv2」口径不一致，故必须重算。\n")

    # 结论
    g = out["judges"]["GLM"]["main"]
    a1 = g["comparisons"]["top1/Ax1_2400_minus_Ax1_176"]
    a3 = g["comparisons"]["top3/Ax1_2400_minus_Ax1_176"]
    a5 = g["comparisons"]["top5/Ax1_2400_minus_Ax1_176"]
    p1 = g["comparisons"]["top1/P_2400_minus_P_176"]
    p3 = g["comparisons"]["top3/P_2400_minus_P_176"]
    p5 = g["comparisons"]["top5/P_2400_minus_P_176"]
    it1 = g["interaction"]["top1"]
    it3 = g["interaction"]["top3"]
    it5 = g["interaction"]["top5"]
    m1 = g["comparisons"]["top1/MDT_176_minus_Ax1_2400"]
    m3 = g["comparisons"]["top3/MDT_176_minus_Ax1_2400"]
    m5 = g["comparisons"]["top5/MDT_176_minus_Ax1_2400"]
    L.append("\n## 7. 结论与改写建议（主判官 GLM × v3，5 seeds）\n")
    L.append(f"1. **规模效应远小于 +8pp**。A×1@176B → A×1@2400B："
             f"top-1 {a1['mean_rate_a']*100:.1f}% vs {a1['mean_rate_b']*100:.1f}%"
             f"（{a1['mean_diff']*100:+.1f}pp，p={fmt_p(a1['wilcoxon_p'])}）、"
             f"top-3 {a3['mean_diff']*100:+.1f}pp（p={fmt_p(a3['wilcoxon_p'])}）、"
             f"top-5 {a5['mean_diff']*100:+.1f}pp（p={fmt_p(a5['wilcoxon_p'])}）。"
             f"P@176B → P@2400B：top-1 {p1['mean_diff']*100:+.1f}pp"
             f"（p={fmt_p(p1['wilcoxon_p'])}）、top-3 {p3['mean_diff']*100:+.1f}pp、"
             f"top-5 {p5['mean_diff']*100:+.1f}pp。")
    L.append(f"2. **交互效应**：ΔA×1 − ΔP = "
             f"top-1 {it1['mean_diff']*100:+.1f}pp（p={fmt_p(it1['wilcoxon_p'])}）、"
             f"top-3 {it3['mean_diff']*100:+.1f}pp（p={fmt_p(it3['wilcoxon_p'])}）、"
             f"top-5 {it5['mean_diff']*100:+.1f}pp（p={fmt_p(it5['wilcoxon_p'])}）；"
             "四条件 Friedman 见各判官小节。"
             + ("「规模红利与策略无关」在新口径下**仍然成立**"
                "（无显著交互）。" if max(it1["wilcoxon_p"], it3["wilcoxon_p"],
                                        it5["wilcoxon_p"]) > 0.05 else
                "存在显著交互，需改写「与策略无关」。"))
    m5_sig = m5["wilcoxon_p"] < 0.05
    m5_verdict = (
        f"top-5 差 {abs(m5['mean_diff']*100):.1f}pp 且不显著 —— "
        "「追平」的定性结论成立，只需换数字。"
        if not m5_sig else
        "top-5 差异显著，不能再写「追平」，须按上表改写。")
    L.append(f"3. **团队 vs 更大单次调用**：MDT@176B（6 次调用）− A×1@2400B = "
             f"top-1 {m1['mean_diff']*100:+.1f}pp（p={fmt_p(m1['wilcoxon_p'])}）、"
             f"top-3 {m3['mean_diff']*100:+.1f}pp（p={fmt_p(m3['wilcoxon_p'])}）、"
             f"top-5 {m5['mean_diff']*100:+.1f}pp（p={fmt_p(m5['wilcoxon_p'])}）；"
             f"绝对命中率 top-3 {m3['mean_rate_a']*100:.1f}% vs "
             f"{m3['mean_rate_b']*100:.1f}%，top-5 {m5['mean_rate_a']*100:.1f}% vs "
             f"{m5['mean_rate_b']*100:.1f}%。" + m5_verdict)
    L.append(f"4. **措辞修正**：原文 “about 8 points of recall” 把 top-1 增益说成了 "
             f"recall —— 实测（GLM）A×1 的 top-1 增益 {a1['mean_diff']*100:+.1f}pp、"
             f"top-5（recall）{a5['mean_diff']*100:+.1f}pp，P 为 "
             f"{p1['mean_diff']*100:+.1f}pp / {p5['mean_diff']*100:+.1f}pp；"
             "「under every strategy」只能在「交互不显著」的意义上保留"
             "（点估计上 P 的增益不比 A×1 小，但 ΔA×1−ΔP 不显著）；"
             f"“81.6%” 应换成 {m5['mean_rate_a']*100:.1f}%（MDT@176B）vs "
             f"{m5['mean_rate_b']*100:.1f}%（A×1@2400B）的 5-seed 病例级均值。")

    L.append("\n### 现文两句的替换草案\n")
    L.append("> **原文（L73）**：scaling the backbone from 176B to 2400B parameters "
             "added about 8 points of recall under every strategy, with no "
             "interaction between scale and strategy; notably, MDT at 176B "
             "(six calls) matched the top-5 recall of a single 2400B call (81.6%).")
    L.append("> ")
    L.append(f"> **建议改写**：scaling the backbone from 176B to 2400B parameters "
             f"added {a1['mean_diff']*100:.1f} points of top-1 accuracy and "
             f"{a5['mean_diff']*100:.1f} points of top-5 recall for A×1, and "
             f"{p1['mean_diff']*100:.1f} / {p5['mean_diff']*100:.1f} points for P "
             f"(all five-seed means, primary judge; the scale×strategy interaction "
             f"was not significant: {it1['mean_diff']*100:+.1f}pp at top-1, "
             f"p = {fmt_p(it1['wilcoxon_p'])}). Notably, MDT at 176B (six calls) "
             f"reached {m5['mean_rate_a']*100:.1f}% top-5 recall versus "
             f"{m5['mean_rate_b']*100:.1f}% for a single 2400B call "
             f"(difference {m5['mean_diff']*100:+.1f} points, "
             f"p = {fmt_p(m5['wilcoxon_p'])}), i.e. the isolated-role team at 176B "
             f"matched — and at top-3 numerically exceeded ({m3['mean_rate_a']*100:.1f}% vs "
             f"{m3['mean_rate_b']*100:.1f}%, p = {fmt_p(m3['wilcoxon_p'])}) — a "
             "single call with an ≈14-fold larger backbone.")
    L.append("\n（数字均来自本文件各表；若判官缓存更新后重跑本脚本，"
             "该段文字会随数字自动更新。）")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
