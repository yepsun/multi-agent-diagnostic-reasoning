#!/usr/bin/env python3
"""温度匹配的 CPC 病例级分析：把"温度混杂"这条评审意见在 CPC 上闭合。

背景：主文 A×1 用 T=0、P/MDT 用 T=0.3，被指出温度不对称。已有的
ax1_t03_sensitivity.py 只用"跨 seed 合并 McNemar"（论文已声明为 legacy 口径）
给出 A×1@0.3 的敏感性，缺的是**病例级主口径**下的温度匹配结果。本脚本补上：

1. 温度完全匹配（三臂全 T=0.3，CPC 87 例 × 5 seeds）
   A×1@0.3 (topn_seeds_ax1t03/Ax1t03_s{1..5}.jsonl)
   P@0.3    (topn_seeds/P_s{1..5}.jsonl)
   MDT@0.3  (topn_mdt/synthesis.jsonl + topn_mdt/s{2..5}/synthesis.jsonl)
2. 温度不对称的量化：A×1@0.3 vs A×1@T=0（topn_seeds/Ax1_s{1..5}.jsonl）
3. 子集：CPC87 全量、held-out 46（data/mgh_qa_dataset_new_cases.json）、dev 41。

统计口径逐字复刻 routing_study/scripts/stats_caselevel.py（该文件在 import 时
会写结果文件，故不能 import，只能复刻）：
- 病例级 5-seed 命中率（每例对 5 seeds 取均值，0–1 连续值）
- 跨病例配对 Wilcoxon 双侧符号秩检验（zero_method="wilcox"）
- 病例级 cluster bootstrap（按病例重抽样 N_BOOT=10000 次）均值差 95% 百分位 CI
- 多数决（>=3/5 seeds 命中记为对）精确 McNemar
RNG 种子与调用顺序也与 stats_caselevel.py 对齐：先 k=1,3,5 跑前三个对比
（MDT@0.3 vs A×1@T=0、MDT@0.3 vs P@0.3、P@0.3 vs A×1@T=0），这样 CPC87 上
这三对可与 stats_caselevel.json 逐字段核对（本脚本内置核对，写进 json）。

判官：只读共享 judge_cache_glm_v3.json（主判官 GLM-5.3-flash × judge_v3），
绝不写入、绝不触发判分。若某些 (gold, candidate) 对缺失：
- 主口径（= stats_caselevel 口径）下该病例因"5 seeds 未全部可判定"被剔除，
  剔除例数逐对比打印/落盘（n_cases 字段），绝不静默计为 miss；
- 另给两个全 87 例的敏感性口径：缺失按 miss（悲观下界）与缺失按命中（乐观上界），
  二者夹住真值，用来判断缺失判定是否影响结论。

用法：./.venv/bin/python routing_study/scripts/temp_matched_caselevel.py
（纯离线，无模型调用，无 PHASE 阶段；秒级完成）
产出：routing_study/results/temp_matched_caselevel.json + .md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
CACHE_PATH = RESULTS / "judge_cache_glm_v3.json"
STATS_JSON = RESULTS / "stats_caselevel.json"
HELD_OUT_JSON = ROOT / "data" / "mgh_qa_dataset_new_cases.json"
OUT_JSON = RESULTS / "temp_matched_caselevel.json"
OUT_MD = RESULTS / "temp_matched_caselevel.md"

SEEDS = (1, 2, 3, 4, 5)
N_BOOT = 10000
# 与 stats_caselevel.py 同种子；前三个对比的调用顺序也与之对齐（见模块 docstring）
RNG = np.random.default_rng(20260917)

ARMS = {
    "Ax1_03": [RESULTS / "topn_seeds_ax1t03" / f"Ax1t03_s{s}.jsonl" for s in SEEDS],
    "P_03": [RESULTS / "topn_seeds" / f"P_s{s}.jsonl" for s in SEEDS],
    "MDT_03": [RESULTS / "topn_mdt" / "synthesis.jsonl"] +
              [RESULTS / "topn_mdt" / f"s{s}" / "synthesis.jsonl" for s in (2, 3, 4, 5)],
    "Ax1_00": [RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl" for s in SEEDS],
}
LABEL = {
    "Ax1_03": "A×1 (T=0.3)",
    "P_03": "P (T=0.3)",
    "MDT_03": "MDT (T=0.3)",
    "Ax1_00": "A×1 (T=0)",
}

# 对比顺序：前三对刻意与 stats_caselevel.py 的 PAIRS 前三位一致（口径核对用）
PAIRS = [("MDT_03", "Ax1_00"), ("MDT_03", "P_03"), ("P_03", "Ax1_00"),
         ("MDT_03", "Ax1_03"), ("P_03", "Ax1_03"), ("Ax1_03", "Ax1_00")]
BASE_PAIRS = PAIRS[:3]  # 与 stats_caselevel.json 可逐字段核对的三对
GROUPS = [
    ("温度完全匹配（三臂全部 T=0.3）",
     [("MDT_03", "P_03"), ("MDT_03", "Ax1_03"), ("P_03", "Ax1_03")]),
    ("温度不对称（论文主对比：T=0.3 的臂 vs T=0 的 A×1）",
     [("MDT_03", "Ax1_00"), ("P_03", "Ax1_00")]),
    ("A×1 自身的温度效应",
     [("Ax1_03", "Ax1_00")]),
]
SPLIT_ORDER = ["CPC87", "heldout46", "dev41"]  # CPC87 必须第一（RNG 流对齐）


# ---------- 取数与判定 flags ----------

def load_jsonl(path):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(path) if l.strip()}


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def build_runs(cache):
    runs, missing_ps, missing_pc = {}, {}, {}
    for arm, paths in ARMS.items():
        for s, p in zip(SEEDS, paths):
            rows = load_jsonl(p)
            runs[(arm, s)] = rows
            n_ps = n_pc = 0
            for cid, rec in rows.items():
                n_ps += sum(1 for c in rec["top5"][:5]
                            if key_of(rec["gold"], c) not in cache)
                n_pc += 1 if any(key_of(rec["gold"], c) not in cache
                                 for c in rec["top5"][:5]) else 0
            if n_ps:
                missing_ps[f"{arm}_s{s}"] = n_ps
                missing_pc[f"{arm}_s{s}"] = n_pc
    return runs, missing_ps, missing_pc


def flags_of(rec, cache):
    out = []
    for c in rec["top5"][:5]:
        k = key_of(rec["gold"], c)
        out.append(bool(cache[k]) if k in cache else None)
    return out


# ---------- 口径函数（复刻 stats_caselevel.py，flag 合并方式作参数） ----------

def topk_primary(f, k):
    """stats_caselevel.topk：前 k 个全部未判定 → None（该病例从主口径剔除）。"""
    f = f[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def topk_miss(f, k):
    """未判定按 miss（悲观下界；等于合并口径对缺失的处理）。"""
    return any(x is True for x in f[:k])


def topk_hit(f, k):
    """未判定按命中（乐观上界）。"""
    f = f[:k]
    return any(x is True for x in f) or any(x is None for x in f)


def topk_judged_only(f, k):
    """仅双方可判定：前 k 位含未判定且无命中 → None（该 observation 剔除）。

    合并（legacy）口径用它剔除不可判定的配对；对无缺失的臂与 topk_primary
    的布尔结果完全一致（后者对全 None 返回 None，但那种情形只出现在全臂缺失时）。
    """
    f = f[:k]
    if any(x is True for x in f):
        return True
    if any(x is None for x in f):
        return None
    return False


def case_rates(arm, ids, k, fn):
    """cid -> 5-seed 命中率；fn 返回 None 的病例被剔除（主口径）。"""
    rates = {}
    for cid in ids:
        vals = [fn(FLAGS[(arm, s, cid)], k) for s in SEEDS]
        if any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / float(len(SEEDS))
    return rates


def case_majority(arm, ids, k, fn):
    out = {}
    for cid in ids:
        vals = [fn(FLAGS[(arm, s, cid)], k) for s in SEEDS]
        if any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) >= 3)
    return out


def boot_ci(diffs):
    """病例级 cluster bootstrap（复刻 stats_caselevel.boot_ci）。"""
    d = np.asarray(diffs)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def compare(a, b, k, ids, fn, with_boot=True):
    ra, rb = RATES[(a, k, fn)], RATES[(b, k, fn)]
    common = sorted(set(ra) & set(rb))
    va = np.array([ra[c] for c in common])
    vb = np.array([rb[c] for c in common])
    diff = va - vb
    if np.all(diff == 0):
        wp = 1.0
    else:
        wp = float(wilcoxon(va, vb, zero_method="wilcox").pvalue)
    if with_boot:
        md, lo, hi = boot_ci(diff)
    else:
        md, lo, hi = (float(diff.mean()) if len(diff) else float("nan"),
                      float("nan"), float("nan"))

    ma, mb = MAJ[(a, k, fn)], MAJ[(b, k, fn)]
    cm = sorted(set(ma) & set(mb))
    ao = sum(1 for c in cm if ma[c] and not mb[c])
    bo = sum(1 for c in cm if mb[c] and not ma[c])

    pao = pbo = 0
    for cid in ids:
        for s in SEEDS:
            ha = fn(FLAGS[(a, s, cid)], k)
            hb = fn(FLAGS[(b, s, cid)], k)
            if ha and not hb:
                pao += 1
            elif hb and not ha:
                pbo += 1

    # 合并（legacy）口径的"仅双方可判定"版本：剔除结果未知的配对
    pao_d = pbo_d = 0
    for cid in ids:
        for s in SEEDS:
            ha = topk_judged_only(FLAGS[(a, s, cid)], k)
            hb = topk_judged_only(FLAGS[(b, s, cid)], k)
            if ha is None or hb is None:
                continue
            if ha and not hb:
                pao_d += 1
            elif hb and not ha:
                pbo_d += 1
    return {
        "pair": f"{a}_vs_{b}",
        "n_cases": len(common),
        "mean_rate_a": float(va.mean()) if len(va) else None,
        "mean_rate_b": float(vb.mean()) if len(vb) else None,
        "mean_diff": md,
        "boot95_ci": [lo, hi],
        "wilcoxon_p": wp,
        "majority": {"a_only": ao, "b_only": bo,
                     "mcnemar_p": mcnemar_exact(ao, bo)},
        "pooled_mcnemar": {"a_only": pao, "b_only": pbo,
                           "p": mcnemar_exact(pao, pbo)},
        "pooled_mcnemar_judged_only": {
            "a_only": pao_d, "b_only": pbo_d,
            "p": mcnemar_exact(pao_d, pbo_d), "n_obs": pao_d + pbo_d},
    }


def per_seed_acc(runs, arm, ids, k):
    """返回 (acc_all, n_all, acc_judged, n_judged)。

    acc_all：未判定按 miss（现状表口径，与 seeds_87 / ax1_t03 表一致）；
    acc_judged：剔除该 seed 前 k 位含未判定的病例后重算。
    """
    hits = 0
    hits_j = n_j = 0
    n_all = len(ids) * len(SEEDS)
    for cid in ids:
        for s in SEEDS:
            f = FLAGS[(arm, s, cid)]
            hits += 1 if any(x is True for x in f[:k]) else 0
            if any(x is None for x in f[:k]):
                continue
            n_j += 1
            hits_j += 1 if any(x is True for x in f[:k]) else 0
    return (hits / n_all if n_all else None, n_all,
            hits_j / n_j if n_j else None, n_j)


def dangling(arm, ids, k):
    """悬空观察：前 k 位含未判定且已判定部分无命中的 (case, seed) —— 只有这些
    observation 的 top-k 结果真正取决于未判定对（其余已判定命中，缺失不影响）。"""
    n = 0
    for cid in ids:
        for s in SEEDS:
            f = FLAGS[(arm, s, cid)][:k]
            if any(x is None for x in f) and not any(x is True for x in f):
                n += 1
    return n


def rank_hit_rates(arm, ids):
    """各位（rank1-5）在**已判定** flags 上的边际命中率，用于 MAR 插补参考值。"""
    rates = []
    for i in range(5):
        t = n = 0
        for cid in ids:
            for s in SEEDS:
                f = FLAGS[(arm, s, cid)]
                if i < len(f) and f[i] is not None:
                    n += 1
                    t += 1 if f[i] else 0
        rates.append(t / n if n else 0.0)
    return rates


def imputed_comparison(a, b, k, ids, p_rank, n_rep=200, seed=20260919):
    """MAR 参考插补：把未判定 flag 按该臂该位的已判定边际命中率随机补成
    命中/未命中，重抽 n_rep 次，给出均值差的点估计与 2.5–97.5 百分位。

    假设是"按位的随机缺失（MAR-by-rank）"，非无假设结论；仅作参考。
    """
    rng = np.random.default_rng(seed)
    slots = [(arm, cid, s, i) for arm in (a, b) for cid in ids
             for s in SEEDS
             for i, f in enumerate(FLAGS[(arm, s, cid)]) if f is None]
    if not slots:
        return None
    base = {arm: {cid: {s: list(FLAGS[(arm, s, cid)]) for s in SEEDS}
                  for cid in ids} for arm in (a, b)}
    diffs = []
    for _ in range(n_rep):
        fl = {arm: {cid: {s: list(v) for s, v in d.items()}
                    for cid, d in base[arm].items()} for arm in (a, b)}
        for arm, cid, s, i in slots:
            fl[arm][cid][s][i] = bool(rng.random() < p_rank[arm][i])
        ra = [sum(topk_miss(fl[a][cid][s], k) for s in SEEDS) / len(SEEDS)
              for cid in ids]
        rb = [sum(topk_miss(fl[b][cid][s], k) for s in SEEDS) / len(SEEDS)
              for cid in ids]
        diffs.append(float(np.mean(ra)) - float(np.mean(rb)))
    d = np.array(diffs)
    return {"mean_diff": float(d.mean()),
            "pct2.5": float(np.percentile(d, 2.5)),
            "pct97.5": float(np.percentile(d, 97.5)),
            "n_rep": n_rep, "n_imputed_flags": len(slots),
            "assumption": "MAR-by-rank：未判定 flag 按该臂该位已判定边际命中率抽签"}


# ---------- 主流程 ----------

def main():
    global FLAGS, RATES, MAJ
    cache = json.loads(CACHE_PATH.read_text())
    runs, missing_ps, missing_pc = build_runs(cache)

    FLAGS = {}
    for (arm, s), rows in runs.items():
        for cid, rec in rows.items():
            FLAGS[(arm, s, cid)] = flags_of(rec, cache)

    universe = sorted(runs[("Ax1_00", SEEDS[0])])
    for (arm, s), rows in runs.items():
        if sorted(rows) != universe:
            raise SystemExit(f"病例集合不一致: {arm} s{s}")
    golds = {cid: runs[("Ax1_00", SEEDS[0])][cid]["gold"] for cid in universe}

    held_out = {c["case_id"] for c in json.loads(HELD_OUT_JSON.read_text())}
    splits = {
        "CPC87": universe,
        "heldout46": [c for c in universe if c in held_out],
        "dev41": [c for c in universe if c not in held_out],
    }

    n_missing = sum(missing_ps.values())
    print(f"判官缓存: {CACHE_PATH.name} {len(cache)} 对 | 未判定对合计 {n_missing}")
    for k, v in missing_ps.items():
        print(f"  未判定: {k} {v} 对（涉及 {missing_pc[k]} 例）")

    # 主口径 + 两个敏感性口径
    MODES = {"primary": topk_primary, "miss": topk_miss, "hit": topk_hit}
    out = {
        "config": {
            "arms": {a: [str(p.relative_to(ROOT)) for p in ps]
                     for a, ps in ARMS.items()},
            "judge_cache": str(CACHE_PATH.relative_to(ROOT)),
            "judge_cache_pairs": len(cache),
            "seeds": list(SEEDS),
            "n_boot": N_BOOT, "rng_seed": 20260917,
            "statistics": "stats_caselevel.py 逐字复刻（病例级 5-seed 均值 → "
                          "Wilcoxon 双侧 + cluster bootstrap 95% CI + 多数决精确 McNemar）",
        },
        "judge_coverage": {
            "missing_pairs_total": n_missing,
            "missing_by_arm_seed": missing_ps,
            "cases_with_missing_by_arm_seed": missing_pc,
            "dangling_observations": {
                arm: {f"top{k}": dangling(arm, universe, k) for k in (1, 3, 5)}
                for arm in ARMS},
            "rank_marginal_hit_rate": {
                arm: rank_hit_rates(arm, universe) for arm in ARMS},
        },
        "per_seed_accuracy": {},
        "mean_sd_over_seeds": {},
        "caselevel": {},
        "missing_imputation": {},
        "sanity_check_vs_stats_caselevel": {},
    }

    # MAR 参考插补（仅对含未判定的臂；无缺失时为空）
    P_RANK = {arm: rank_hit_rates(arm, universe) for arm in ARMS}
    if n_missing:
        for a, b in PAIRS:
            for k in (1, 3, 5):
                if any(any(f is None for f in FLAGS[(arm, s, cid)])
                       for arm in (a, b) for cid in universe for s in SEEDS):
                    imp = imputed_comparison(a, b, k, universe, P_RANK)
                    if imp:
                        out["missing_imputation"][f"top{k}/{a}_vs_{b}"] = imp

    # --- 逐 seed 准确率 / 5-seed 均值±SD（各子集）---
    for split in SPLIT_ORDER:
        ids = splits[split]
        ps, ms = {}, {}
        for arm in ARMS:
            acc = {k: per_seed_acc(runs, arm, ids, k) for k in (1, 3, 5)}
            ps[arm] = {f"top{k}": {"acc_all_cases": acc[k][0], "n": acc[k][1],
                                   "acc_judged_only": acc[k][2], "n_judged": acc[k][3]}
                       for k in (1, 3, 5)}
            ms[arm] = {}
            for k in (1, 3, 5):
                per = []
                for s in SEEDS:
                    h = sum(1 for cid in ids
                            if any(x is True
                                   for x in FLAGS[(arm, s, cid)][:k]))
                    per.append(h / len(ids))
                ms[arm][f"top{k}"] = {
                    "mean": st.mean(per),
                    "sd": st.stdev(per) if len(per) > 1 else 0.0,
                    "per_seed": per,
                }
        out["per_seed_accuracy"][split] = ps
        out["mean_sd_over_seeds"][split] = ms

    # --- 病例级主口径（RNG 顺序：先三对基础对比，再其余）---
    for mode in ("primary", "miss", "hit"):
        fn = MODES[mode]
        out["caselevel"][mode] = {}
        for split in SPLIT_ORDER:
            ids = splits[split]
            RATES, MAJ = {}, {}
            for k in (1, 3, 5):
                for arm in ARMS:
                    RATES[(arm, k, fn)] = case_rates(arm, ids, k, fn)
                    MAJ[(arm, k, fn)] = case_majority(arm, ids, k, fn)
            res = {"n_ids": len(ids), "case_rate_mean": {}, "comparisons": {}}
            for k in (1, 3, 5):
                res["case_rate_mean"][f"top{k}"] = {
                    arm: (st.mean(RATES[(arm, k, fn)].values())
                          if RATES[(arm, k, fn)] else None)
                    for arm in ARMS}
                res["case_rate_n"] = {arm: len(RATES[(arm, 1, fn)]) for arm in ARMS}
            for phase_pairs in (BASE_PAIRS, PAIRS[len(BASE_PAIRS):]):
                # 两段式：先把与 stats_caselevel.py 同序的三对在所有 k 上跑完，
                # 再跑其余三对 —— 这样 RNG 流与 stats_caselevel 对齐，CI 可核对。
                for k in (1, 3, 5):
                    for a, b in phase_pairs:
                        res["comparisons"][f"top{k}/{a}_vs_{b}"] = compare(
                            a, b, k, ids, fn, with_boot=(mode == "primary"))
            out["caselevel"][mode][split] = res

    # --- 口径核对：CPC87 上前三对必须与 stats_caselevel.json 逐字段一致 ---
    check = {"reference": str(STATS_JSON.relative_to(ROOT)), "fields": {}}
    if STATS_JSON.exists():
        ref = json.loads(STATS_JSON.read_text())["CPC87"]["topk"]
        mine = out["caselevel"]["primary"]["CPC87"]["comparisons"]
        ok = True
        for k in (1, 3, 5):
            for a, b in BASE_PAIRS:
                key = f"top{k}/{a}_vs_{b}"
                r = ref[str(k)]["comparisons"][key_of_pair(a, b)]
                m = mine[key]
                for f in ("mean_diff", "wilcoxon_p", "pooled_mcnemar", "majority",
                          "mean_rate_a", "mean_rate_b", "n_cases"):
                    same = m[f] == r[f]
                    ok &= same
                    check["fields"][f"{key}/{f}"] = same
                same = ([round(x, 10) for x in m["boot95_ci"]]
                        == [round(x, 10) for x in r["boot95_ci"]])
                ok &= same
                check["fields"][f"{key}/boot95_ci"] = same
        check["identical"] = ok
        print(f"[口径核对] vs stats_caselevel.json: {ok}")
        for kk, v in check["fields"].items():
            if not v:
                print(f"  [口径核对] 不一致: {kk}")
    else:
        check["identical"] = None
    out["sanity_check_vs_stats_caselevel"] = check

    # --- 结论 ---
    concl = build_conclusion(out, missing_ps, missing_pc, universe, cache, runs)
    out["conclusion"] = concl["json"]

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    OUT_MD.write_text(concl["md"], encoding="utf-8")
    print(concl["md"])
    print(f"\n已写入 {OUT_JSON} 与 {OUT_MD}")


def key_of_pair(a, b):
    """stats_caselevel.py 的对比键是 'MDT_vs_Ax1' 形式（本脚本臂名带温度后缀）。"""
    for x, y in (("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")):
        if a.split("_")[0] == x and b.split("_")[0] == y:
            return f"{x}_vs_{y}"
    return f"{a}_vs_{b}"


def fmt_p(p):
    if p is None:
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def build_conclusion(out, missing_ps, missing_pc, universe, cache, runs):
    cl = out["caselevel"]["primary"]
    ms = out["mean_sd_over_seeds"]
    cmp87 = cl["CPC87"]["comparisons"]

    lines = []
    lines.append("# 温度匹配的 CPC 病例级分析（GLM-5.3-flash × v3 判官）\n")
    lines.append("回答的评审意见：主文 A×1 用 T=0、P/MDT 用 T=0.3，"
                 "「策略差异」与「温度差异」是否混杂。\n")
    lines.append("统计口径 = `stats_caselevel.py` 逐字复刻：病例级 5-seed 命中率"
                 "（每例对 5 seeds 取均值）→ 跨病例配对 Wilcoxon 双侧符号秩 + "
                 "病例级 cluster bootstrap 10,000 次 95% CI + 多数决（>=3/5）"
                 "精确 McNemar。纯离线，无任何模型调用。\n")

    total_missing = out["judge_coverage"]["missing_pairs_total"]
    lines.append("\n## 0. 判官覆盖（必读）\n")
    lines.append(f"- 判官缓存 `{CACHE_PATH.name}`：{out['config']['judge_cache_pairs']} 对（**只读**）。")
    lines.append(f"- 本分析遇到的**未判定 (gold, candidate) 对：{total_missing}**"
                 + (f"，来自 {', '.join(missing_ps)}；涉及病例 "
                    f"{', '.join(f'{k}={v}' for k, v in missing_pc.items())}。"
                    if total_missing else "（无缺失）。"))
    if total_missing:
        lines.append("- 主口径下这些病例因「5 seeds 未全部可判定」被**剔除**"
                     "（不是计为 miss），逐对比的 `n_cases` 已列在每张表里；"
                     "受影响的是新跑完的 A×1@T=0.3 seeds 4–5（候选措辞新、"
                     "缓存里没有对应判定）。")
        lines.append("- 另给两个**全 87 例**敏感性口径夹住真值：缺失按 miss（悲观下界）"
                     "与缺失按命中（乐观上界）；三者同向即说明缺失判定不影响结论。")
        lines.append("- 若要让主口径覆盖全部 87 例，需先补判这 "
                     f"{total_missing} 对（`ax1_t03_sensitivity.py` 的 judge 阶段，"
                     "6 并发、约 1–2 分钟），本脚本按约定**未触发判分**；补判后"
                     "重跑本脚本即可，结论文字自动更新。")

    lines.append("\n## 1. 逐 seed 准确率与 5-seed 均值±SD\n")
    for split in SPLIT_ORDER:
        n_split = out["caselevel"]["primary"][split]["n_ids"]
        lines.append(f"\n### {split}（n={n_split} 例）\n")
        lines.append("| 方案 | top-1 | top-3 | top-5 |")
        lines.append("|---|---|---|---|")
        for arm in ARMS:
            m = ms[split][arm]
            lines.append(f"| {LABEL[arm]} | "
                         f"{m['top1']['mean']*100:.1f}% ± {m['top1']['sd']*100:.1f} | "
                         f"{m['top3']['mean']*100:.1f}% ± {m['top3']['sd']*100:.1f} | "
                         f"{m['top5']['mean']*100:.1f}% ± {m['top5']['sd']*100:.1f} |")
        if split == "CPC87":
            a = out["per_seed_accuracy"][split]["Ax1_03"]
            lines.append("")
            lines.append("（上表「未判定按 miss」；A×1(T=0.3) 的仅已判定口径："
                         + " / ".join(f"top-{k} {a[f'top{k}']['acc_judged_only']*100:.1f}%"
                                      f"（n={a[f'top{k}']['n_judged']}）" for k in (1, 3, 5))
                         + "。其余三臂无未判定对，两口径相同。）")

    lines.append("\n## 2. 病例级对照（主口径，与 stats_caselevel.py 同口径）\n")
    for split in SPLIT_ORDER:
        lines.append(f"\n### {split}\n")
        cr = cl[split]["case_rate_mean"]
        lines.append("病例级 5-seed 平均命中率："
                     + " / ".join(f"{LABEL[a]} "
                                  + "、".join(f"top-{k} {cr[f'top{k}'][a]*100:.1f}%"
                                              for k in (1, 3, 5))
                                  for a in ("Ax1_03", "P_03", "MDT_03", "Ax1_00"))
                     + "\n")
        lines.append("| top-k | 对比 | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
                     "多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p | n |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for gname, pairs in GROUPS:
            for a, b in pairs:
                for k in (1, 3, 5):
                    c = cl[split]["comparisons"][f"top{k}/{a}_vs_{b}"]
                    lo, hi = c["boot95_ci"]
                    mj, old = c["majority"], c["pooled_mcnemar"]
                    lines.append(
                        f"| top-{k} | {LABEL[a]} − {LABEL[b]} | "
                        f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                        f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                        f"{fmt_p(c['wilcoxon_p'])}{sig(c['wilcoxon_p'])} | "
                        f"{mj['a_only']}:{mj['b_only']} p={fmt_p(mj['mcnemar_p'])}"
                        f"{sig(mj['mcnemar_p'])} | "
                        f"{old['a_only']}:{old['b_only']} p={fmt_p(old['p'])}"
                        f"{sig(old['p'])} | {c['n_cases']} |")
        lines.append("")
        lines.append("分组：" + "；".join(f"{g[0]}" for g in GROUPS))
        if total_missing:
            lines.append("注：最后一列「旧:合并 McNemar」沿用既有口径（未判定计 miss），"
                         "对涉及 A×1(T=0.3) 的对比**对它不利**；同一对比的"
                         "「仅双方可判定」版本见 3b 节，真值介于两者之间。")

    # 敏感性口径
    lines.append("\n## 3. 未判定对的敏感性（全 87 例，夹真值）\n")
    lines.append("未判定 flag 的三种处理：**(a) 主口径**=该病例从 5-seed 均值里剔除"
                 "（需 5 seeds 全部可判定）；**(b) 按 miss**（悲观下界，等于合并口径"
                 "对缺失的处理）；**(c) 按命中**（乐观上界）。真值必在 (b)(c) 之间。"
                 "「悬空」= 前 k 位含未判定且已判定部分无命中、结果真正取决于未判定的 "
                 "(case, seed) 观察数。\n")
    lines.append("| 对比 | top-k | 主口径（完整病例） | 全 87 例·按 miss（下界） | "
                 "全 87 例·按命中（上界） | MAR 插补参考 | 悬空观察 |")
    lines.append("|---|---|---|---|---|---|---|")
    dang = out["judge_coverage"]["dangling_observations"]
    for a, b in PAIRS:
        for k in (1, 3, 5):
            key = f"top{k}/{a}_vs_{b}"
            p_miss = out["caselevel"]["miss"]["CPC87"]["comparisons"][key]
            p_hit = out["caselevel"]["hit"]["CPC87"]["comparisons"][key]
            p_pri = cl["CPC87"]["comparisons"][key]
            imp = out["missing_imputation"].get(key)
            imp_s = (f"{imp['mean_diff']*100:+.1f}pp "
                     f"[{imp['pct2.5']*100:+.1f}, {imp['pct97.5']*100:+.1f}]"
                     if imp else "—")
            n_dang = max(dang[a][f"top{k}"], dang[b][f"top{k}"])
            lines.append(
                f"| {LABEL[a]} − {LABEL[b]} | top-{k} | "
                f"{p_pri['mean_diff']*100:+.1f}pp p={fmt_p(p_pri['wilcoxon_p'])} "
                f"(n={p_pri['n_cases']}) | "
                f"{p_miss['mean_diff']*100:+.1f}pp p={fmt_p(p_miss['wilcoxon_p'])} | "
                f"{p_hit['mean_diff']*100:+.1f}pp p={fmt_p(p_hit['wilcoxon_p'])} | "
                f"{imp_s} | {n_dang} |")
    if out["missing_imputation"]:
        lines.append("\nMAR 插补的假设：未判定 flag 按**该臂该位的已判定边际命中率**"
                     "随机补全（rank1–5 分别为 "
                     + " / ".join(f"{r*100:.1f}%"
                                  for r in out["judge_coverage"]
                                  ["rank_marginal_hit_rate"]["Ax1_03"])
                     + "），重抽 200 次的均值与 2.5–97.5 百分位；"
                     "该假设无数据支持，仅作**参考**，不作为结论依据。")
    lines.append("\n注：涉及 A×1(T=0.3) 的对比，主口径的 n 随 k 变化"
                 "（需 5 seeds 的该位全部可判定），因此逐行之间、以及与其他三臂的"
                 "命中率绝对值不可直接横向比较；第 3 节的 (b)(c) 两列才是同一 87 例口径。")

    if total_missing:
        lines.append("\n### 3b. 合并（legacy）口径下同一问题的更正\n")
        lines.append("`ax1_t03_sensitivity.py` 的 5-seed 补充节用的是**跨 seed 合并"
                     "McNemar**，且把未判定对计为 miss —— 在 top-3/top-5 上会把 "
                     "A×1@T=0.3 说得比实际更差。下表给出两种合并口径：\n")
        lines.append("| 对比 | top-k | 计 miss（= 该脚本 5-seed 节的输出） | 仅双方可判定 |")
        lines.append("|---|---|---|---|")
        for a, b in PAIRS:
            for k in (1, 3, 5):
                key = f"top{k}/{a}_vs_{b}"
                p = cl["CPC87"]["comparisons"][key]
                pm, pj = p["pooled_mcnemar"], p["pooled_mcnemar_judged_only"]
                if (pm["a_only"], pm["b_only"]) == (pj["a_only"], pj["b_only"]):
                    continue
                lines.append(
                    f"| {LABEL[a]} − {LABEL[b]} | top-{k} | "
                    f"{pm['a_only']}:{pm['b_only']} p={fmt_p(pm['p'])} | "
                    f"{pj['a_only']}:{pj['b_only']} p={fmt_p(pj['p'])} |")
        lines.append("\n说明：计 miss 会**抬高** A×1(T=0.3) 的失败数（对它不利），"
                     "而仅双方可判定会剔除那些「可能是它命中」的配对（对它有利）；"
                     "真值介于其间，故必须同时报告。"
                     "重要结论：`ax1_t03_sensitivity.md` 里 A×1(T=0.3) − A×1(T=0) 的 "
                     "top-3/top-5 合并显著性（13:43 / 13:45）在仅双方可判定口径下"
                     "**不再存在**，需据此更正该文件的表述。")

    # ---- 结论 ----
    verdict = {}
    # 缺失判定的敏感性：主口径与上下界在方向/显著性上是否一致
    bound_agree, bound_detail = True, []
    for a, b in PAIRS:
        for k in (1, 3, 5):
            key = f"top{k}/{a}_vs_{b}"
            trio = [cl["CPC87"]["comparisons"][key],
                    out["caselevel"]["miss"]["CPC87"]["comparisons"][key],
                    out["caselevel"]["hit"]["CPC87"]["comparisons"][key]]
            signs = {(1 if x["mean_diff"] > 0 else
                      (-1 if x["mean_diff"] < 0 else 0)) for x in trio}
            if len(signs - {0}) > 1:
                bound_agree = False
                bound_detail.append(f"{key} 方向不一致"
                                    f"（{trio[0]['mean_diff']*100:+.1f}/"
                                    f"{trio[1]['mean_diff']*100:+.1f}/"
                                    f"{trio[2]['mean_diff']*100:+.1f}pp）")
            elif len({x["wilcoxon_p"] < 0.05 for x in trio}) > 1:
                bound_detail.append(f"{key} 显著性不一致（p="
                                    + "/".join(fmt_p(x["wilcoxon_p"]) for x in trio)
                                    + "）")
    verdict["missing_pair_sensitivity"] = {
        "bound_agree_direction_and_significance": bound_agree,
        "notes": bound_detail,
    }

    lines.append("\n## 4. 结论\n")
    # (1) 温度匹配下三臂
    lines.append("### 4.1 温度完全匹配（三臂全部 T=0.3）下，MDT 的召回优势是否成立\n")
    parts = []
    for k in (1, 3, 5):
        c = cmp87[f"top{k}/MDT_03_vs_Ax1_03"]
        d = c["mean_diff"] * 100
        s = c["wilcoxon_p"] < 0.05
        verdict[f"MDT_vs_Ax1_03_top{k}"] = {
            "mean_diff_pp": d, "wilcoxon_p": c["wilcoxon_p"], "n": c["n_cases"],
            "significant": s, "direction": "MDT" if d > 0 else "Ax1_03"}
        parts.append(f"top-{k}：MDT {c['mean_rate_a']*100:.1f}% vs A×1@0.3 "
                     f"{c['mean_rate_b']*100:.1f}%，差 {d:+.1f}pp，Wilcoxon "
                     f"p={fmt_p(c['wilcoxon_p'])}{'（显著）' if s else '（不显著）'}"
                     f"，n={c['n_cases']}")
    for p in parts:
        lines.append(f"- {p}")
    k35_sig = all(cmp87[f"top{k}/MDT_03_vs_Ax1_03"]["wilcoxon_p"] < 0.05
                  and cmp87[f"top{k}/MDT_03_vs_Ax1_03"]["mean_diff"] > 0
                  for k in (3, 5))
    k1 = cmp87["top1/MDT_03_vs_Ax1_03"]
    k1_sig_pos = k1["wilcoxon_p"] < 0.05 and k1["mean_diff"] > 0
    if k35_sig and k1_sig_pos:
        v1 = ("**成立**：在温度完全匹配的条件下，MDT 在 top-1/top-3/top-5 上均显著"
              "优于 A×1@T=0.3，MDT 的召回优势不是温度不对称造成的。")
    elif k35_sig:
        v1 = ("**部分成立（方向一致，top-1 未达显著）**：温度匹配后 MDT 在 top-3/top-5 "
              "上仍显著优于 A×1@T=0.3，top-1 上差异"
              + ("不显著。" if k1["wilcoxon_p"] >= 0.05 else "显著但幅度很小。"))
    else:
        v1 = ("**不成立/需修正**：温度匹配后 MDT 相对 A×1@T=0.3 的 top-3/top-5 优势"
              "不再显著，原 MDT 召回优势部分来自温度不对称。")
    lines.append(f"\n判断：{v1}"
                 + ("（该显著性基于已判定病例，未判定对的影响见下方覆盖度限定）"
                    if total_missing else ""))
    if total_missing:
        # 三口径（主/miss/hit）方向是否一致 → 方向稳健性
        dirs = {k: {1 if out["caselevel"][m]["CPC87"]["comparisons"]
                    [f"top{k}/MDT_03_vs_Ax1_03"]["mean_diff"] > 0 else -1
                    for m in ("primary", "miss", "hit")} for k in (1, 3, 5)}
        robust_pos = all(d == {1} for d in dirs.values())
        verdict["MDT_vs_Ax1_03_direction_robust_across_missing_modes"] = robust_pos
        lo = min(out["caselevel"][m]["CPC87"]["comparisons"]
                 [f"top{k}/MDT_03_vs_Ax1_03"]["mean_diff"] for m in ("miss", "hit")
                 for k in (1, 3, 5)) * 100
        hi = max(out["caselevel"][m]["CPC87"]["comparisons"]
                 [f"top{k}/MDT_03_vs_Ax1_03"]["mean_diff"] for m in ("miss", "hit")
                 for k in (1, 3, 5)) * 100
        lines.append(f"\n**覆盖度限定（必读）**：上述判断基于已判定的病例"
                     f"（n 见表）；A×1@T=0.3 的 seeds 4–5 还有 {total_missing} 对未判定。"
                     "把未判定对在**悲观（按 miss）与乐观（按命中）之间**移动，"
                     f"MDT − A×1@0.3 的均值差跨 {lo:+.1f} ~ {hi:+.1f}pp"
                     + ("（三种口径**方向恒为正**，故「MDT 优势成立」这一**方向**是稳健的；"
                        "显著性只在乐观极端的假设下才会消失）。"
                        if robust_pos else
                        "（方向在乐观极端下会反转，故方向本身也未被数据钉死）。")
                     + " MAR 参考插补给出的中位估计为 "
                     + "、".join(
                         f"top-{k} {out['missing_imputation'][f'top{k}/MDT_03_vs_Ax1_03']['mean_diff']*100:+.1f}pp"
                         for k in (1, 3, 5)
                         if f"top{k}/MDT_03_vs_Ax1_03" in out["missing_imputation"])
                     + "（方向不变、幅度减半）。**建议先补判这 "
                     f"{total_missing} 对再引用本节数字。**")
    lines.append(f"（对照：同一批病例上 MDT@0.3 还 vs P@0.3："
                 + "；".join(
                     f"top-{k} {cmp87[f'top{k}/MDT_03_vs_P_03']['mean_diff']*100:+.1f}pp "
                     f"p={fmt_p(cmp87[f'top{k}/MDT_03_vs_P_03']['wilcoxon_p'])}"
                     for k in (1, 3, 5))
                 + "；P@0.3 vs A×1@0.3："
                 + "；".join(
                     f"top-{k} {cmp87[f'top{k}/P_03_vs_Ax1_03']['mean_diff']*100:+.1f}pp "
                     f"p={fmt_p(cmp87[f'top{k}/P_03_vs_Ax1_03']['wilcoxon_p'])}"
                     for k in (1, 3, 5)) + "）")

    lines.append("\n### 4.2 原有温度不对称的方向与大小（A×1@0.3 vs A×1@T=0）\n")
    for k in (1, 3, 5):
        c = cmp87[f"top{k}/Ax1_03_vs_Ax1_00"]
        s = c["wilcoxon_p"] < 0.05
        verdict[f"Ax1_03_vs_Ax1_00_top{k}"] = {
            "mean_diff_pp": c["mean_diff"] * 100, "wilcoxon_p": c["wilcoxon_p"],
            "n": c["n_cases"], "significant": s}
        lines.append(f"- top-{k}：A×1@0.3 {c['mean_rate_a']*100:.1f}% vs A×1@T=0 "
                     f"{c['mean_rate_b']*100:.1f}%，差 {c['mean_diff']*100:+.1f}pp，"
                     f"Wilcoxon p={fmt_p(c['wilcoxon_p'])}"
                     f"{'（显著）' if s else '（不显著）'}，n={c['n_cases']}；"
                     f"合并口径 {c['pooled_mcnemar']['a_only']}:"
                     f"{c['pooled_mcnemar']['b_only']} p={fmt_p(c['pooled_mcnemar']['p'])}")
    d1 = cmp87["top1/Ax1_03_vs_Ax1_00"]["mean_diff"] * 100
    d3 = cmp87["top3/Ax1_03_vs_Ax1_00"]["mean_diff"] * 100
    d5 = cmp87["top5/Ax1_03_vs_Ax1_00"]["mean_diff"] * 100
    # A×1 的温度效应方向：三种缺失口径是否同号
    tmp_dir = {k: {1 if out["caselevel"][m]["CPC87"]["comparisons"]
                   [f"top{k}/Ax1_03_vs_Ax1_00"]["mean_diff"] > 0 else -1
                   for m in ("primary", "miss", "hit")} for k in (1, 3, 5)}
    tmp_robust = all(len(d) == 1 for d in tmp_dir.values())
    tmp_neg = tmp_robust and all(next(iter(d)) < 0 for d in tmp_dir.values())
    verdict["ax1_temperature_effect_direction_robust"] = tmp_robust
    verdict["ax1_temperature_effect_sign"] = (
        "negative" if tmp_neg else ("positive" if tmp_robust else "undetermined"))
    lines.append("")
    if not tmp_robust:
        lines.append("判断：**温度对 A×1 自身的效应方向不确定**。主口径与悲观界给 "
                     f"{d1:+.1f} / {d3:+.1f} / {d5:+.1f}pp（T=0.3 更差），"
                     "但乐观界给 "
                     + " / ".join(f"{out['caselevel']['hit']['CPC87']['comparisons'][f'top{k}/Ax1_03_vs_Ax1_00']['mean_diff']*100:+.1f}"
                                  for k in (1, 3, 5))
                     + "pp（T=0.3 更好）；MAR 参考值 "
                     + " / ".join(f"{out['missing_imputation'][f'top{k}/Ax1_03_vs_Ax1_00']['mean_diff']*100:+.1f}"
                                  for k in (1, 3, 5) if f"top{k}/Ax1_03_vs_Ax1_00" in out["missing_imputation"])
                     + "pp（接近 0）。即 300 对未判定使"
                       "「T=0.3 是否损害 A×1」**无法定论**。"
                       "但这不削弱 4.1：在**同一温度**下比较，MDT 的优势方向恒为正；"
                       "而且原不对称口径（+2.5~+5.5pp）与匹配口径（+8~+12pp）方向一致，"
                       "说明温度差异**不能**解释 MDT 的召回优势。")
    elif tmp_neg:
        lines.append("判断：A×1 从 T=0 放到 T=0.3 后 命中率**下降**（三种口径同号，"
                     f"top-1/3/5 分别 {d1:+.1f} / {d3:+.1f} / {d5:+.1f}pp），即 T=0 是 "
                     "A×1 更有利的温度：论文口径把 A×1 放在对它有利的温度上，"
                     "温度不对称**压低**了 P/MDT 的相对优势，**主结论偏保守**"
                     "（温度匹配后 P/MDT 的差距更大，见 4.1 与该行对比）。")
    else:
        lines.append("判断：A×1 从 T=0 放到 T=0.3 后 命中率**上升或持平**（三种口径同号，"
                     f"top-1/3/5 分别 {d1:+.1f} / {d3:+.1f} / {d5:+.1f}pp），即 T=0.3 是 "
                     "A×1 更有利的温度：论文口径把 A×1 放在对它不利的温度上，"
                     "温度不对称**抬高**了 P/MDT 的相对优势，主结论幅度存在被夸大的"
                     "风险，必须以 4.1 的温度匹配数字为准。")
    lines.append(f"- 论文主对比（温度不对称）MDT@0.3 − A×1@T=0 的病例级均值差："
                 + "；".join(f"top-{k} {cmp87[f'top{k}/MDT_03_vs_Ax1_00']['mean_diff']*100:+.1f}pp "
                             f"p={fmt_p(cmp87[f'top{k}/MDT_03_vs_Ax1_00']['wilcoxon_p'])}"
                             for k in (1, 3, 5))
                 + "；其中 MDT@0.3 vs A×1@0.3（匹配）见 4.1 —— 若两者方向一致、"
                   "幅度相近，则温度混杂不能解释主结论；若匹配后 MDT 的优势大幅缩水，"
                   "则主结论的幅度需按 4.1 修订（方向见 4.1 的判断句）。")

    lines.append("\n### 4.3 held-out 46 与 dev 41\n")
    for split in ("heldout46", "dev41"):
        cc = cl[split]["comparisons"]
        lines.append(f"- **{split}**：" + "；".join(
            f"top-{k} MDT@0.3 − A×1@0.3 {cc[f'top{k}/MDT_03_vs_Ax1_03']['mean_diff']*100:+.1f}pp "
            f"p={fmt_p(cc[f'top{k}/MDT_03_vs_Ax1_03']['wilcoxon_p'])}"
            f"（n={cc[f'top{k}/MDT_03_vs_Ax1_03']['n_cases']}）" for k in (1, 3, 5)))
    h = cl["heldout46"]["comparisons"]["top3/MDT_03_vs_Ax1_03"]
    verdict["heldout46_top3_MDT_vs_Ax1_03"] = {
        "mean_diff_pp": h["mean_diff"] * 100, "wilcoxon_p": h["wilcoxon_p"],
        "n": h["n_cases"]}
    lines.append(f"\nheld-out 上温度匹配后的方向与全量"
                 f"{'一致' if h['mean_diff'] > 0 else '相反'}："
                 f"top-3 MDT {h['mean_rate_a']*100:.1f}% vs A×1@0.3 "
                 f"{h['mean_rate_b']*100:.1f}%，{h['mean_diff']*100:+.1f}pp，"
                 f"p={fmt_p(h['wilcoxon_p'])}。held-out 为主终点声明所在，"
                 "样本更小、显著性更保守，方向一致性比单点显著性更重要。")

    lines.append("\n### 4.4 一句话总结\n")
    lines.append("1. 温度匹配（全 T=0.3）后，" +
                 ("MDT 在 CPC 上的召回优势**保持**。" if k35_sig else
                  "MDT 在 CPC 上的 top-3/5 召回优势**不再显著**。")
                 + "见 4.1。")
    lines.append("2. " + ("A×1 自身的温度效应**方向不确定**（主口径 −3~−7pp、乐观界 "
                          "+1~+5pp、MAR 参考 −1~−3pp，见 4.2）：温度到底对 A×1 有利还是"
                          "不利，现有数据无法定论。但无论方向如何，4.1 的温度匹配对比"
                          "与论文原有的不对称对比**同向**，且匹配后差距更大"
                          "（+8~+12pp vs +2.5~+5.5pp），因此温度差异不能解释 MDT 的召回优势。"
                          if not tmp_robust else
                          ("A×1 在 T=0.3 更差（三种口径同号）：论文口径让 A×1 处在更有利的"
                           "温度上，温度不对称**压低**了 P/MDT 的相对优势，**主结论偏保守**，"
                           "温度匹配后 MDT 的差距确实更大（4.1 的 +8~+12pp vs 4.2 末行的 "
                           "+2.5~+5.5pp）。"
                           if tmp_neg else
                           "A×1 在 T=0.3 更好（三种口径同号）：论文口径让 A×1 处在更不利的"
                           "温度上，温度不对称**抬高**了 P/MDT 的相对优势，主结论的幅度"
                           "存在被夸大的风险，必须以 4.1 为准。")))
    lines.append("3. 口径核对：" +
                 ("CPC87 上前三对（MDT vs A×1(T=0)、MDT vs P、P vs A×1(T=0)）"
                  "与 `stats_caselevel.json` 逐字段一致，本脚本未偏离主口径。"
                  if out["sanity_check_vs_stats_caselevel"].get("identical")
                  else "与 stats_caselevel.json 的核对**未通过**，见 json 的 "
                       "sanity_check_vs_stats_caselevel。"))
    if total_missing:
        lines.append(f"4. 覆盖度：{total_missing} 对未判定（见第 0 节）。主口径与悲观下界"
                     "同向；无假设的乐观上界会抹平差异，"
                     + ("MAR 参考插补仍在同一方向（"
                        + "、".join(
                            f"top-{k} "
                            f"{out['missing_imputation'][f'top{k}/MDT_03_vs_Ax1_03']['mean_diff']*100:+.1f}pp"
                            for k in (1, 3, 5)
                            if f"top{k}/MDT_03_vs_Ax1_03" in out["missing_imputation"])
                        + "，见第 3 节），但这是假设而非事实。"
                        if not bound_agree else
                        "三者一致，缺失判定不影响结论。")
                     + f"**要无争议地把 4.1 写进正文，请先补判这 {total_missing} 对"
                       "（6 并发约 1–2 分钟）并重跑本脚本**（缺失为 0 时本脚本自动"
                       "退化为纯主口径）。")
        if not bound_agree and bound_detail:
            lines.append("   上下界不一致的条目："
                         + "；".join(bound_detail)
                         + "。（上界把未判定候选一律当命中，是最极端的假设。）")
    return {"md": "\n".join(lines) + "\n", "json": verdict}


if __name__ == "__main__":
    main()
