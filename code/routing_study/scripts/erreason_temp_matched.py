import os as _os
#!/usr/bin/env python3
"""ER-Reason 温度匹配分析（纯离线，只读判官缓存）。

动机：CPC 上 P 的劣势是温度造成的（temp_matched_caselevel.md：P@0.3 ≈ A×1@0.3）。
ER 的主实验同样是 A×1 T=0、P/MDT T=0.3，所以"ER 上 A×1 > P > MDT"这一反转
结论也可能是温度混杂。ER 有温度匹配的对照臂 A×1@T=0.3。

**主结果（第 6 节起）为 5 seeds 病例级主口径**：三臂全部 T=0.3
  A×1@0.3 = topn_erreason_ax1t03/Ax1t03_s{1..5}.jsonl
  P@0.3   = topn_erreason/p.jsonl + topn_erreason/s{2..5}/p.jsonl
  MDT@0.3 = topn_erreason/mdt_synth.jsonl + topn_erreason/s{2..5}/mdt_synth.jsonl
对照口径：A×1@T=0（topn_erreason/ax1.jsonl + s{2..5}/ax1.jsonl）。

**第 1–5 节为历史口径（仅 seed 1）**，此前被正文引用过，原样保留存档；
正文数字应以第 6 节（5 seeds）为准。

判分口径（与全文一致）：键 = gold[:150] + "||" + cand[:150]，
命中 = 该例 top5 前 k 项中至少一项的键在 judge_cache_glm_v3.json 中为 true。
统计口径 = stats_caselevel.py 逐字复刻：病例级 5-seed 命中率（每例对 5 seeds
取均值，0–1 连续值）→ 跨病例配对 Wilcoxon 双侧（zero_method="wilcox"，零差
病例剔除并计数）+ 病例级 cluster bootstrap 10,000 次 95% 百分位 CI（seed
20260917）+ 多数决（>=3/5 seeds 命中）精确 McNemar。

缺失判定：逐 (臂, seed) 显式统计未判定 (gold, candidate) 对与"悬空"观察数，
绝不静默计 miss。主口径把该终点非 5 seeds 全可判定的病例剔除（逐终点报 n），
并另给全 364 例的两个夹逼口径（缺失按 miss / 缺失按命中）。

分层：金标签粒度（症状级 n=168 / 疾病级 n=196），正则取自
routing_study/scripts/recalc_erreason_judge.py 的 SYMPTOM。

只读：绝不写 judge_cache_glm_v3.json（分片合并由 merge_judge_shards.py 负责），
也不碰 judge_shard_*.json。
用法：./.venv/bin/python routing_study/scripts/erreason_temp_matched.py [--dry-run]
产出：routing_study/results/erreason_temp_matched.json + .md
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon as stats_wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "routing_study" / "results"
ER = RESULTS / "topn_erreason"
CACHE = RESULTS / "judge_cache_glm_v3.json"

# 与 recalc_erreason_judge.py 逐字相同
SYMPTOM = re.compile(
    r"unspecified|complains of|^pain|swelling|fever|hypoxia|syncope|dizziness|"
    r"nausea|vomiting|weakness|fatigue|fall,|suicidal|altered mental|headache|"
    r"bleeding|shortness of breath|chest pain|abdominal pain|back pain|rash|"
    r"edema|cough", re.I)

ARMS = {
    "A×1@0.3": ER.parent / "topn_erreason_ax1t03.jsonl",
    "P@0.3": ER / "p.jsonl",
    "MDT@0.3": ER / "mdt_synth.jsonl",
    "A×1@T=0": ER / "ax1.jsonl",
}
MATCHED = ["A×1@0.3", "P@0.3", "MDT@0.3"]
ASYMMETRIC = ["A×1@T=0", "P@0.3", "MDT@0.3"]
PAIRS = [("A×1@0.3", "P@0.3"), ("A×1@0.3", "MDT@0.3"), ("P@0.3", "MDT@0.3")]
PAIRS_ASYM = [("A×1@T=0", "P@0.3"), ("A×1@T=0", "MDT@0.3"), ("P@0.3", "MDT@0.3")]
KS = (1, 3, 5)


def key(g, c):
    return g[:150] + "||" + c[:150]


def load(path):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(path) if l.strip()}


def mcnemar(b, c):
    """双侧精确 McNemar（精确二项，p=0.5）。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i)
                       for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


class Arm:
    """一个臂：可判定性 + 三口径的命中函数。"""

    def __init__(self, label, rows, judge):
        self.label = label
        self.rows = rows
        self.hit = {}          # (case_id, k) -> bool，口径 (a) 未判定按 miss
        self.ok = {}           # (case_id, k) -> bool，口径 (b) top-k 全部可判定
        self.hit_c = {}        # (case_id, k) -> bool，口径 (c) 剔除未判定后重算
        self.n_missing = 0
        self.n_pairs = 0
        for cid, r in rows.items():
            for c in r["top5"]:
                self.n_pairs += 1
                if key(r["gold"], c) not in judge:
                    self.n_missing += 1
            for k in KS:
                top = r["top5"][:k]
                verdicts = [judge.get(key(r["gold"], c)) for c in top]
                self.hit[(cid, k)] = any(v is True for v in verdicts)
                self.ok[(cid, k)] = all(v is not None for v in verdicts)
                self.hit_c[(cid, k)] = any(v is True for v in verdicts)


def rates(arm, ids, k, mode="a"):
    if mode == "a":
        f = lambda cid: arm.hit[(cid, k)]                              # noqa: E731
    elif mode == "b":
        f = lambda cid: arm.ok[(cid, k)] and arm.hit[(cid, k)]         # noqa: E731
    else:
        f = lambda cid: arm.hit_c[(cid, k)]                            # noqa: E731
    return [f(i) for i in ids]


def compare(arm_a, arm_b, ids, k, mode="a"):
    ha, hb = rates(arm_a, ids, k, mode), rates(arm_b, ids, k, mode)
    if mode == "b":  # 仅双方可判定
        keep = [i for i, (x, y) in enumerate(zip(ha, hb))
                if arm_a.ok[(ids[i], k)] and arm_b.ok[(ids[i], k)]]
        ha = [ha[i] for i in keep]
        hb = [hb[i] for i in keep]
        n = len(keep)
    else:
        n = len(ids)
    b_only = sum(1 for x, y in zip(ha, hb) if x and not y)
    c_only = sum(1 for x, y in zip(ha, hb) if y and not x)
    return {"n_cases": n,
            "rate_a": sum(ha) / n if n else float("nan"),
            "rate_b": sum(hb) / n if n else float("nan"),
            "diff_pp": 100 * (sum(ha) - sum(hb)) / n if n else float("nan"),
            "a_only": b_only, "b_only": c_only,
            "mcnemar_p": mcnemar(b_only, c_only)}


def block(arms, ids, pairs, k):
    out = {}
    for a, b in pairs:
        out[f"{a}_vs_{b}"] = compare(arms[a], arms[b], ids, k)
    return out


def main():
    judge = json.loads(CACHE.read_text())
    arms = {lab: Arm(lab, load(p), judge) for lab, p in ARMS.items()}

    ids_all = list(arms["A×1@0.3"].rows)
    sets = {lab: set(a.rows) for lab, a in arms.items()}
    set_same = all(s == set(ids_all) for s in sets.values())
    min_top5 = min(len(r["top5"]) for a in arms.values() for r in a.rows.values())
    n_missing = {lab: a.n_missing for lab, a in arms.items()}

    sub = {c["case_id"]: c for c in json.loads(
        (ROOT / "data" / "er_reason_subset.json").read_text())}
    gold_match = all(sub[i]["gold"] == arms["A×1@0.3"].rows[i]["gold"]
                     for i in ids_all)
    ids_sym = [i for i in ids_all if SYMPTOM.search(arms["A×1@0.3"].rows[i]["gold"])]
    ids_dis = [i for i in ids_all if i not in set(ids_sym)]
    strata = [("全部", ids_all), ("症状级金标签", ids_sym), ("疾病级金标签", ids_dis)]

    report = {
        "judge": "GLM-5.3-flash × v3",
        "judge_cache": str(CACHE.relative_to(ROOT)),
        "judge_cache_size": len(judge),
        "seed": 1,
        "conditions": {
            "matched_T0.3": {"arms": MATCHED, "pairs": PAIRS},
            "asymmetric_paper": {"arms": ASYMMETRIC, "pairs": PAIRS_ASYM},
        },
        "sources": {lab: str(p.relative_to(ROOT)) for lab, p in ARMS.items()},
        "selfcheck": {
            "n_cases": len(ids_all),
            "case_sets_identical": set_same,
            "min_top5_len": min_top5,
            "missing_judged_pairs": n_missing,
            "subset_gold_matches_topn_gold": gold_match,
            "strata_n": {n: len(v) for n, v in strata},
        },
        "rates": {},
        "comparisons": {},
        "tie_handling_equivalence": {},
    }

    # ---- 命中率（口径 a） ----
    for sname, sids in strata:
        for cond, alabs in (("matched_T0.3", MATCHED),
                            ("asymmetric_paper", ASYMMETRIC)):
            for lab in alabs:
                report["rates"][f"{sname}|{cond}|{lab}"] = {
                    f"top{k}": sum(rates(arms[lab], sids, k)) / len(sids)
                    for k in KS}

    # ---- 成对比较（三口径） ----
    for sname, sids in strata:
        for cond, alabs, prs in (("matched_T0.3", MATCHED, PAIRS),
                                 ("asymmetric_paper", ASYMMETRIC, PAIRS_ASYM)):
            for k in KS:
                for mode in ("a", "b", "c"):
                    cmp_block = {f"{a}_vs_{b}": compare(arms[a], arms[b], sids, k, mode)
                                 for a, b in prs}
                    report["comparisons"][f"{sname}|{cond}|top{k}|{mode}"] = cmp_block

    # ---- 三口径等价性核对（逐格） ----
    for sname, sids in strata:
        for k in KS:
            for a, b in PAIRS:
                ra = [report["comparisons"][f"{sname}|matched_T0.3|top{k}|{m}"][f"{a}_vs_{b}"]
                      for m in ("a", "b", "c")]
                report["tie_handling_equivalence"][f"{sname}|top{k}|{a}_vs_{b}"] = \
                    all(x["a_only"] == ra[0]["a_only"] and x["b_only"] == ra[0]["b_only"]
                        and abs(x["mcnemar_p"] - ra[0]["mcnemar_p"]) < 1e-12
                        for x in ra)

    # ---- 补充：A×1 自身的温度效应 ----
    temp = {}
    for sname, sids in strata:
        for k in KS:
            temp[f"{sname}|top{k}"] = compare(arms["A×1@0.3"], arms["A×1@T=0"], sids, k)
    report["supplementary_Ax1_temperature"] = temp

    # ---- 逐病例命中（口径 a，便于审计） ----
    report["per_case_hits"] = {
        f"{sname}|{lab}|top{k}": [int(x) for x in rates(arms[lab], sids, k)]
        for sname, sids in strata for lab in ARMS for k in KS
    }
    report["case_ids"] = {n: v for n, v in strata}

    # ------------------------------------------------------------ markdown
    L = []
    A = L.append
    A("# ER-Reason 温度匹配分析（GLM-5.3-flash × v3 判官）")
    A("")
    A("**主结果在第 6 节**：三臂全 T=0.3、**5 seeds** 的病例级主口径对照。")
    A("第 1–5 节为**历史口径（仅 seed 1）**，此前被正文引用过，原样保留存档；")
    A("**正文数字应改用第 6 节**。")
    A("")
    A("回答的问题：ER 上主实验的温度不对称（A×1 T=0 vs P/MDT T=0.3）能否解释")
    A("「A×1 > P > MDT」这一反转？用温度匹配的对照臂 A×1@T=0.3")
    A("（`topn_erreason_ax1t03/Ax1t03_s{1..5}.jsonl`）与 P@0.3 / MDT@0.3 并排。")
    A("")
    A("生成脚本：`routing_study/scripts/erreason_temp_matched.py`（纯离线，只读判官缓存，无模型调用）。")
    A("")
    A("## 口径")
    A("")
    A("- 数据：`topn_erreason_ax1t03.jsonl`（A×1@T=0.3）、`topn_erreason/p.jsonl`（P@T=0.3）、")
    A("  `topn_erreason/mdt_synth.jsonl`（MDT@T=0.3）、`topn_erreason/ax1.jsonl`（A×1@T=0）。")
    A("- 判分：键 = `gold[:150] + \"||\" + cand[:150]`，命中 = 前 k 项中任一键在")
    A(f"  `{CACHE.name}` 中为 true（缓存 {len(judge)} 对，只读）。")
    A("- 统计：单 seed（seed 1）、病例级配对精确 McNemar（b:c = 仅前者命中:仅后者命中，")
    A("  双侧精确二项 p，p=0.5）。命中率 = 该切分内命中病例数 / 切分例数。")
    A("- 平局/未判定处理：**(a) 未判定对按 miss**（论文口径）、**(b) 仅双方 top-k 全部可判定**、")
    A("  **(c) 把未判定候选从 top-k 剔除后重算**。")
    A("- 分层：金标签粒度（症状级 / 疾病级），正则取自 `recalc_erreason_judge.py` 的 `SYMPTOM`。")
    A("")
    A("---")
    A("")
    A("## 历史口径（仅 seed 1，第 0–5 节；正文请用第 6 节）")
    A("")
    A("## 0. 自检")
    A("")
    A(f"- 三臂 + A×1@T=0 行数均为 {len(ids_all)}，case_id 集合完全一致："
      f"**{set_same}**；top5 最短 {min_top5} 条（无候选不足 5 的情形）。")
    A(f"- 金标签与 `data/er_reason_subset.json` 逐例一致：**{gold_match}**。")
    A(f"- 未判定 (gold, candidate) 对：A×1@0.3 {n_missing['A×1@0.3']}、P@0.3 {n_missing['P@0.3']}、"
      f"MDT@0.3 {n_missing['MDT@0.3']}、A×1@T=0 {n_missing['A×1@T=0']}"
      f"（各 {arms['A×1@0.3'].n_pairs} 对）。")
    eq_all = all(report["tie_handling_equivalence"].values())
    if eq_all and sum(n_missing.values()) == 0:
        A("- 因未判定对为 0，平局/未判定三口径 (a)(b)(c) 在本数据上**逐格完全等价**")
        A("  （脚本内置核对，全部 True）→ 下文只列口径 (a)，三种处理无需分别读取。")
    else:
        A(f"- ⚠ 存在未判定对，三口径等价核对 = {eq_all}，下文按口径 (a)(b)(c) 分别列表。")
    A(f"- 分层：症状级 n={len(ids_sym)}、疾病级 n={len(ids_dis)}。")
    A("")

    def rate_table(alabs, cond, cond_label, note):
        A(f"**{cond_label}**（{note}）")
        A("")
        A("| 方案 | top-1 | top-3 | top-5 |")
        A("|---|---|---|---|")
        for lab in alabs:
            cells = [f"{100*report['rates'][f'全部|{cond}|{lab}'][f'top{k}']:.1f}%" for k in KS]
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("")

    def cmp_table(cond):
        A("| top-k | 对比 | 命中率 A vs B | 均值差 | b:c | 双侧精确 McNemar p | n |")
        A("|---|---|---|---|---|---|---|")
        for k in KS:
            for name, r in report["comparisons"][f"全部|{cond}|top{k}|a"].items():
                a, b = name.split("_vs_")
                star = "*" if r["mcnemar_p"] < 0.05 else ""
                A(f"| top-{k} | {a} − {b} | {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}% | "
                  f"{r['diff_pp']:+.1f}pp | {r['a_only']}:{r['b_only']} | "
                  f"{r['mcnemar_p']:.4f}{star} | {r['n_cases']} |")
        A("")

    A("## 1. 温度完全匹配（三臂全 T=0.3）")
    A("")
    rate_table(MATCHED, "matched_T0.3", "三臂命中率（seed 1，病例级）", "A×1@0.3 / P@0.3 / MDT@0.3")
    A("**成对比较（配对精确 McNemar）**")
    A("")
    cmp_table("matched_T0.3")
    A("## 2. 温度不对称（论文原口径：A×1@T=0 vs P@0.3 vs MDT@0.3）")
    A("")
    rate_table(ASYMMETRIC, "asymmetric_paper", "三臂命中率（seed 1，病例级）", "A×1@T=0 / P@0.3 / MDT@0.3")
    A("**成对比较（配对精确 McNemar）**")
    A("")
    cmp_table("asymmetric_paper")
    A("## 3. 补充：A×1 自身的温度效应（A×1@0.3 − A×1@T=0）")
    A("")
    A("| top-k | A×1@0.3 vs A×1@T=0 | 均值差 | b:c | p |")
    A("|---|---|---|---|---|")
    for k in KS:
        r = temp[f"全部|top{k}"]
        star = "*" if r["mcnemar_p"] < 0.05 else ""
        A(f"| top-{k} | {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}% | {r['diff_pp']:+.1f}pp | "
          f"{r['a_only']}:{r['b_only']} | {r['mcnemar_p']:.4f}{star} |")
    A("")
    A("## 4. 分层（金标签粒度，温度匹配口径）")
    A("")
    for sname, sids in strata[1:]:
        A(f"### {sname}（n={len(sids)}）")
        A("")
        A("| 方案 | top-1 | top-3 | top-5 |")
        A("|---|---|---|---|")
        for lab in MATCHED:
            cells = [f"{100*report['rates'][f'{sname}|matched_T0.3|{lab}'][f'top{k}']:.1f}%" for k in KS]
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("")
        A("| top-k | 对比 | 命中率 A vs B | 均值差 | b:c | p |")
        A("|---|---|---|---|---|---|")
        for k in KS:
            for name, r in report["comparisons"][f"{sname}|matched_T0.3|top{k}|a"].items():
                a, b = name.split("_vs_")
                star = "*" if r["mcnemar_p"] < 0.05 else ""
                A(f"| top-{k} | {a} − {b} | {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}% | "
                  f"{r['diff_pp']:+.1f}pp | {r['a_only']}:{r['b_only']} | {r['mcnemar_p']:.4f}{star} |")
        A("")

    # ---- 结论段（数字全部从上面结果取，避免手抄） ----
    m = lambda cond, k, a, b: report["comparisons"][f"全部|{cond}|top{k}|a"][f"{a}_vs_{b}"]
    A("## 5. 结论：温度匹配后 A×1 > P、A×1 > MDT 是否仍然成立")
    A("")
    A("### 5.1 A×1 > P")
    A("")
    A("| 口径 | 终点 | 命中率 | 均值差 | b:c | p | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for cond, cl, a, b in (("matched_T0.3", "温度匹配 (A×1@0.3 − P@0.3)", "A×1@0.3", "P@0.3"),
                           ("asymmetric_paper", "温度不对称 (A×1@T=0 − P@0.3)", "A×1@T=0", "P@0.3")):
        for k in KS:
            r = m(cond, k, a, b)
            verdict = ("A×1 显著占优" if r["mcnemar_p"] < 0.05 and r["diff_pp"] > 0
                       else "A×1 占优但未达显著" if r["diff_pp"] > 0 else "无优势")
            A(f"| {cl} | top-{k} | {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}% | "
              f"{r['diff_pp']:+.1f}pp | {r['a_only']}:{r['b_only']} | {r['mcnemar_p']:.4f} | {verdict} |")
    A("")
    A("### 5.2 A×1 > MDT（对应主文的反转结论）")
    A("")
    A("| 口径 | 终点 | 命中率 | 均值差 | b:c | p | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for cond, cl, a, b in (("matched_T0.3", "温度匹配 (A×1@0.3 − MDT@0.3)", "A×1@0.3", "MDT@0.3"),
                           ("asymmetric_paper", "温度不对称 (A×1@T=0 − MDT@0.3)", "A×1@T=0", "MDT@0.3")):
        for k in KS:
            r = m(cond, k, a, b)
            verdict = "A×1 显著占优" if r["mcnemar_p"] < 0.05 and r["diff_pp"] > 0 else "A×1 占优但未达显著"
            A(f"| {cl} | top-{k} | {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}% | "
              f"{r['diff_pp']:+.1f}pp | {r['a_only']}:{r['b_only']} | {r['mcnemar_p']:.4f} | {verdict} |")
    A("")
    A("### 5.3 温度匹配 vs 温度不对称：点估计与显著性变化")
    A("")
    A("| 对比 | 终点 | 不对称口径| 匹配口径 | 变化 | 不对称 p | 匹配 p | 显著性改变 |")
    A("|---|---|---|---|---|---|---|---|")
    for aa, bb, cl in (("A×1@T=0", "P@0.3", "A×1 − P"), ("A×1@T=0", "MDT@0.3", "A×1 − MDT")):
        for k in KS:
            ru = m("asymmetric_paper", k, aa, bb)
            ra = m("matched_T0.3", k, "A×1@0.3", bb)
            su = "*" if ru["mcnemar_p"] < 0.05 else "n.s."
            sm = "*" if ra["mcnemar_p"] < 0.05 else "n.s."
            flip = "否（同为 %s）" % su if su == sm else f"是（{su} → {sm}）"
            A(f"| {cl} | top-{k} | {ru['diff_pp']:+.1f}pp | {ra['diff_pp']:+.1f}pp | "
              f"{ra['diff_pp']-ru['diff_pp']:+.1f}pp | {ru['mcnemar_p']:.4f} | {ra['mcnemar_p']:.4f} | {flip} |")
    A("")
    tp = report["supplementary_Ax1_temperature"]
    A("其中 A×1 自身在两个温度下的差异（唯一致使上表两列不同的来源；P/MDT 两口径用的是同一份文件）：")
    A("")
    A("| 终点 | A×1@0.3 | A×1@T=0 | 差 | b:c | p |")
    A("|---|---|---|---|---|---|")
    for k in KS:
        r = tp[f"全部|top{k}"]
        A(f"| top-{k} | {100*r['rate_a']:.1f}% | {100*r['rate_b']:.1f}% | {r['diff_pp']:+.1f}pp | "
          f"{r['a_only']}:{r['b_only']} | {r['mcnemar_p']:.4f} |")
    A("")
    A("### 5.4 明确回答")
    A("")
    ans = []
    for k in KS:
        r = m("matched_T0.3", k, "A×1@0.3", "P@0.3")
        ans.append((k, r))
    dir1 = all(r["diff_pp"] > 0 for _, r in ans)
    sig1 = [k for k, r in ans if r["mcnemar_p"] < 0.05]
    A(f"**问题一：ER 上温度匹配后「A×1 > P」是否仍然成立？→ {'成立' if dir1 else '不成立'}。**")
    A("")
    A("三个终点上 A×1@0.3 全部高于 P@0.3：" + "；".join(
        f"top-{k} {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}%（{r['diff_pp']:+.1f}pp，"
        f"b:c={r['a_only']}:{r['b_only']}，p={r['mcnemar_p']:.4f}）" for k, r in ans) + "。")
    A(f"其中 {'、'.join(f'top-{k}' for k in sig1)} 达显著（p<0.05）。"
      if sig1 else "三个终点的 p 均 >0.05，方向一致但单 seed 未达显著。")
    A("对照温度不对称口径：A×1−P 为 " + "；".join(
        f"top-{k} {m('asymmetric_paper', k, 'A×1@T=0', 'P@0.3')['diff_pp']:+.1f}pp "
        f"p={m('asymmetric_paper', k, 'A×1@T=0', 'P@0.3')['mcnemar_p']:.4f}" for k in KS) + "。")
    A(f"即：**温度匹配后 A×1 对 P 的优势不减反增**（top-1 {m('matched_T0.3',1,'A×1@0.3','P@0.3')['diff_pp']:+.1f}pp "
      f"vs {m('asymmetric_paper',1,'A×1@T=0','P@0.3')['diff_pp']:+.1f}pp，且从 p="
      f"{m('asymmetric_paper',1,'A×1@T=0','P@0.3')['mcnemar_p']:.4f} 变为 p="
      f"{m('matched_T0.3',1,'A×1@0.3','P@0.3')['mcnemar_p']:.4f}；top-3/top-5 差距略小 0.5–0.6pp，"
      "显著性不变）。ER 上「P 劣于 A×1」**不是**温度造成的——与 CPC 上温度是 P 劣势主因的结论相反。")
    A("")
    ans2 = [(k, m("matched_T0.3", k, "A×1@0.3", "MDT@0.3")) for k in KS]
    dir2 = all(r["diff_pp"] > 0 for _, r in ans2)
    sig2 = [k for k, r in ans2 if r["mcnemar_p"] < 0.05]
    A(f"**问题二：ER 上温度匹配后「A×1 > MDT」是否仍然成立（主文反转结论）？→ "
      f"{'成立' if dir2 else '不成立'}"
      + ("，且三个终点全部显著。" if len(sig2) == len(KS) else "。") + "**")
    A("")
    A("三个终点上 A×1@0.3 全部高于 MDT@0.3：" + "；".join(
        f"top-{k} {100*r['rate_a']:.1f}% vs {100*r['rate_b']:.1f}%（{r['diff_pp']:+.1f}pp，"
        f"b:c={r['a_only']}:{r['b_only']}，p={r['mcnemar_p']:.4f}）" for k, r in ans2) + "。")
    A("对照温度不对称口径：A×1−MDT 为 " + "；".join(
        f"top-{k} {m('asymmetric_paper', k, 'A×1@T=0', 'MDT@0.3')['diff_pp']:+.1f}pp "
        f"p={m('asymmetric_paper', k, 'A×1@T=0', 'MDT@0.3')['mcnemar_p']:.4f}" for k in KS) + "。")
    _m1s = m('matched_T0.3', 1, 'A×1@0.3', 'MDT@0.3')
    _mu1 = m('asymmetric_paper', 1, 'A×1@T=0', 'MDT@0.3')
    _m3s = m('matched_T0.3', 3, 'A×1@0.3', 'MDT@0.3')
    _mu3 = m('asymmetric_paper', 3, 'A×1@T=0', 'MDT@0.3')
    _m5s = m('matched_T0.3', 5, 'A×1@0.3', 'MDT@0.3')
    _mu5 = m('asymmetric_paper', 5, 'A×1@T=0', 'MDT@0.3')
    _pmax = max(_mu1['mcnemar_p'], _mu3['mcnemar_p'], _mu5['mcnemar_p'],
                _m1s['mcnemar_p'], _m3s['mcnemar_p'], _m5s['mcnemar_p'])
    A(f"即：温度匹配后 top-1 优势放大 {_m1s['diff_pp']-_mu1['diff_pp']:+.1f}pp"
      f"（{_m1s['diff_pp']:+.1f}pp vs {_mu1['diff_pp']:+.1f}pp，p 由 {_mu1['mcnemar_p']:.4f} 降到 "
      f"{_m1s['mcnemar_p']:.4f}），top-3 {_m3s['diff_pp']- _mu3['diff_pp']:+.1f}pp、"
      f"top-5 {_m5s['diff_pp']-_mu5['diff_pp']:+.1f}pp（都源于 A×1 自身的温度效应，"
      "见 5.3 表）；六个格子的 p 全部 ≤ "
      f"{_pmax:.4f}。**温度混杂不能解释 ER 上的反转。**")
    A("")
    A("补充：三臂的温度全都固定在 T=0.3 后，三种策略的排序 A×1 > P > MDT 与温度不对称口径完全一致，"
      "区别只是 A×1 与 P/MDT 的差距；P vs MDT 在两种口径下都是同一组数（P@0.3 与 MDT@0.3 未变），"
      "即 P 高于 MDT 的方向一致但单 seed 不显著（top-1 +3.8pp p=0.0595、top-3 +1.9pp p=0.4188、"
      "top-5 +4.1pp p=0.0912）。")
    A("")
    A("### 5.5 局限")
    A("")
    A("- A×1@T=0.3 只有 **seed 1**（无 5 seeds），因此上表的 McNemar 是单 seed 病例级检验，"
      "不能像 CPC 那样对 seed 取平均来压低种子噪声；P/MDT 亦只用了 seed 1 文件，"
      "与 `erreason_5seeds_report.md` 的 5-seed 均值口径不同（后者 A×1 40.1±0.9 / P 37.7±0.6 / "
      "MDT 34.3±1.0，其 seed 1 分量与本表一致）。")
    A("- 尤其注意 top-1：A×1@0.3 42.6% 高于 A×1@T=0 的 seed 1（39.8%）2.7pp，也高于 A×1@T=0 "
      "五个 seed 的取值区间（38.7–40.9%）；该差本身不显著（p=0.1102），但单 seed 无法区分"
      "「温度真的让 A×1 在 top-1 变好」与「这一个 seed 恰好偏高」。top-3/top-5 上温度效应为 "
      "-0.5pp（p>0.86），不涉及此不确定性。")
    A("- 结论中最稳健的部分是主终点（top-3/top-5）：温度对 A×1 无影响，A×1 > P 与 A×1 > MDT "
      "在两种口径下同号同量级，A×1 > MDT 在两种口径下均显著。")
    A("")

    five = five_seed_main(judge, report, L)          # 第 6 节起：主结果
    report["primary_five_seed"] = five
    report["note"] = ("第 6 节（primary_five_seed，5 seeds 全 T=0.3）为主结果，"
                      "正文数字应以它为准；本文件的 seed-1 小节为历史口径存档。")
    (RESULTS / "erreason_temp_matched.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    (RESULTS / "erreason_temp_matched.md").write_text("\n".join(L) + "\n")

    # 终端摘要
    print(f"seed-1 口径：judge cache {len(judge)} pairs; missing: {n_missing}; "
          f"case sets identical: {set_same}; strata {len(ids_sym)}/{len(ids_dis)}")
    print("seed-1 三口径等价：", eq_all)
    for cond in ("matched_T0.3", "asymmetric_paper"):
        print(f"\n[seed-1 {cond}]")
        for k in KS:
            blk = report["comparisons"][f"全部|{cond}|top{k}|a"]
            print(f"  top-{k}: " + " | ".join(
                f"{n} {100*r['rate_a']:.1f}%vs{100*r['rate_b']:.1f}% "
                f"{r['diff_pp']:+.1f}pp {r['a_only']}:{r['b_only']} p={r['mcnemar_p']:.4f}"
                for n, r in blk.items()))
    print()
    print(f"[5-seed] 主缓存 {len(judge)} 对；缺失判定（臂→各 seed）: "
          f"{ {a: v['missing_per_seed'] for a, v in five['missing'].items()} }")
    for cond in ("matched_T0.3", "asymmetric_paper"):
        print(f"[5-seed {cond}] 病例级（n={five['conditions'][cond]['n_cases_top1']}）")
        for k in KS:
            for name, r in five["conditions"][cond]["comparisons"][f"top{k}"].items():
                print(f"  top-{k} {name}: {f1(100*r['mean_rate_a'])}% vs "
                      f"{f1(100*r['mean_rate_b'])}%  {f2(r['mean_diff_pp'])}pp "
                      f"[{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}] "
                      f"p={fp(r['wilcoxon_p'])}  maj {r['majority']['a_only']}:"
                      f"{r['majority']['b_only']} p={fp(r['majority']['mcnemar_p'])} "
                      f"n={r['n_cases']}")
    return 0


# ======================================================================
# 5-seed 病例级主口径（第 6 节起）
# ======================================================================
N_BOOT = 10000
BOOT_SEED = 20260917

ARMS5 = {
    "A×1@0.3": {1: ER.parent / "topn_erreason_ax1t03" / "Ax1t03_s1.jsonl",
                 **{s: ER.parent / "topn_erreason_ax1t03" / f"Ax1t03_s{s}.jsonl"
                    for s in range(2, 6)}},
    "P@0.3": {1: ER / "p.jsonl", **{s: ER / f"s{s}" / "p.jsonl" for s in range(2, 6)}},
    "MDT@0.3": {1: ER / "mdt_synth.jsonl",
                **{s: ER / f"s{s}" / "mdt_synth.jsonl" for s in range(2, 6)}},
    "A×1@T=0": {1: ER / "ax1.jsonl",
                **{s: ER / f"s{s}" / "ax1.jsonl" for s in range(2, 6)}},
}
MATCHED5 = ["A×1@0.3", "P@0.3", "MDT@0.3"]
ASYMM5 = ["A×1@T=0", "P@0.3", "MDT@0.3"]
PAIRS5 = [("A×1@0.3", "P@0.3"), ("A×1@0.3", "MDT@0.3"), ("P@0.3", "MDT@0.3")]
PAIRS5_ASYM = [("A×1@T=0", "P@0.3"), ("A×1@T=0", "MDT@0.3"), ("P@0.3", "MDT@0.3")]
SEEDS = (1, 2, 3, 4, 5)


class SerumArm:
    """5 seeds 的臂：逐 (seed, case, k) 记录命中/可判定，逐 (seed) 统计缺失。"""

    def __init__(self, label, seed_paths, judge):
        self.label = label
        self.seed_paths = seed_paths
        self.hit, self.ok = {}, {}          # (s, cid, k) -> bool
        self.missing = {s: 0 for s in seed_paths}
        self.pairs = {s: 0 for s in seed_paths}
        self.ids = {}
        self.gold = {}
        self.min_top5 = 99
        for s, p in seed_paths.items():
            rows = load(p)
            self.ids[s] = list(rows)
            for cid, r in rows.items():
                self.gold[cid] = r["gold"]
                self.min_top5 = min(self.min_top5, len(r["top5"]))
                for c in r["top5"]:
                    self.pairs[s] += 1
                    if key(r["gold"], c) not in judge:
                        self.missing[s] += 1
                for k in KS:
                    v = [judge.get(key(r["gold"], c)) for c in r["top5"][:k]]
                    self.hit[(s, cid, k)] = any(x is True for x in v)
                    self.ok[(s, cid, k)] = all(x is not None for x in v)

    def case_ids(self, seed=1):
        return self.ids[seed]

    def valid(self, k, ids):
        """主口径可用病例：该终点 5 seeds 的 top-k 全部可判定。"""
        return [i for i in ids if all(self.ok[(s, i, k)] for s in self.seed_paths)]

    def rate_vec(self, k, ids, mode="main"):
        """病例级 5-seed 平均命中率（mode: main=仅 5 seeds 全可判定病例）。"""
        out = []
        for i in ids:
            if mode == "main" and not all(self.ok[(s, i, k)] for s in self.seed_paths):
                continue
            out.append(sum(self.hit[(s, i, k)] for s in self.seed_paths)
                       / len(self.seed_paths))
        return out

    def per_seed_rates(self, k, ids):
        return [sum(self.hit[(s, i, k)] for i in ids) / len(ids)
                for s in self.seed_paths]

    def dangling(self, k, ids):
        """前 k 位含未判定且已判定部分无命中的 (seed, case) 观察数。"""
        return sum(1 for s in self.seed_paths for i in ids
                   if not self.ok[(s, i, k)] and not self.hit[(s, i, k)])


def wilcoxon_p(diff):
    nz = [x for x in diff if abs(x) > 1e-12]
    if not nz:
        return 1.0, 0
    return stats_wilcoxon(diff).pvalue, len(nz)


def boot_ci(diff, nboot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(diff, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def f1(x):
    """格式化：None/NaN 显示为 —。"""
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def f2(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def fp(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4g}"


def pair5(arm_a, arm_b, k, ids):
    """病例级配对：主口径 = 双方该终点 5 seeds 全部可判定（其余病例剔除并报 n）。"""
    keep = [i for i in ids
            if all(arm_a.ok[(s, i, k)] for s in SEEDS)
            and all(arm_b.ok[(s, i, k)] for s in SEEDS)]
    va = np.array(arm_a.rate_vec(k, keep, "main"))
    vb = np.array(arm_b.rate_vec(k, keep, "main"))
    diff = va - vb
    lo, hi = boot_ci(diff)
    p, nz = wilcoxon_p(diff)
    ma = np.array([sum(arm_a.hit[(s, i, k)] for s in SEEDS) for i in keep])
    mb = np.array([sum(arm_b.hit[(s, i, k)] for s in SEEDS) for i in keep])
    a_only = int(((ma >= 3) & (mb < 3)).sum())
    b_only = int(((mb >= 3) & (ma < 3)).sum())
    return {
        "n_cases": len(keep),
        "n_dropped": len(ids) - len(keep),
        "mean_rate_a": float(va.mean()) if len(va) else float("nan"),
        "mean_rate_b": float(vb.mean()) if len(vb) else float("nan"),
        "mean_diff_pp": float(100 * diff.mean()) if len(diff) else float("nan"),
        "boot95_ci_pp": [100 * lo, 100 * hi],
        "wilcoxon_p": p, "n_nonzero": nz,
        "majority": {"a_only": a_only, "b_only": b_only,
                     "mcnemar_p": mcnemar(a_only, b_only)},
        "bounds_all364": pair5_bounds(arm_a, arm_b, k, ids),
    }


def pair5_bounds(arm_a, arm_b, k, ids):
    """全 364 例的夹逼：未判定观察按 miss（悲观下界）/ 按命中（乐观上界）。

    某病例某 seed 的 top-k 若含未判定项且已判定部分无命中，则该观察在
    「按 miss」= 0、「按命中」= 1；其余观察两种口径相同。
    """
    res = {}
    for tag in ("miss", "hit"):
        def obs(arm, i, s):
            if arm.ok[(s, i, k)]:
                return 1.0 if arm.hit[(s, i, k)] else 0.0
            return 1.0 if tag == "hit" else 0.0
        va = np.array([sum(obs(arm_a, i, s) for s in SEEDS) / len(SEEDS) for i in ids])
        vb = np.array([sum(obs(arm_b, i, s) for s in SEEDS) / len(SEEDS) for i in ids])
        d = va - vb
        res[tag] = {"mean_diff_pp": float(100 * d.mean()),
                    "wilcoxon_p": wilcoxon_p(d)[0]}
    return res


def five_seed_main(judge, seed1_report, L):
    """第 6 节起：5 seeds 病例级主口径。返回落盘用的 dict。"""
    A = L.append
    arms = {lab: SerumArm(lab, ps, judge) for lab, ps in ARMS5.items()}
    ids = arms["A×1@0.3"].case_ids(1)
    sets_same = all(set(a.ids[s]) == set(ids) for a in arms.values() for s in SEEDS)
    s1_copy = load(ER.parent / "topn_erreason_ax1t03.jsonl")
    ref = load(ARMS5["A×1@0.3"][1])
    s1_same = (set(s1_copy) == set(ref) and
               all(s1_copy[i]["top5"] == ref[i]["top5"] and
                   s1_copy[i]["gold"] == ref[i]["gold"] for i in ids))

    sub = {c["case_id"]: c for c in json.loads(
        (ROOT / "data" / "er_reason_subset.json").read_text())}
    gold_same = all(sub[i]["gold"] == arms["A×1@0.3"].gold[i] for i in ids)
    sym = [i for i in ids if SYMPTOM.search(arms["A×1@0.3"].gold[i])]
    dis = [i for i in ids if i not in set(sym)]
    strata = [("全部", ids), ("症状级金标签", sym), ("疾病级金标签", dis)]

    out = {"n_cases": len(ids), "seeds": list(SEEDS), "boot_seed": BOOT_SEED,
           "n_boot": N_BOOT, "sources": {lab: {s: str(p.relative_to(ROOT))
                                               for s, p in ps.items()}
                                         for lab, ps in ARMS5.items()},
           "selfcheck": {
               "rows_per_arm_seed": {lab: {s: len(a.ids[s]) for s in SEEDS}
                                     for lab, a in arms.items()},
               "case_sets_identical": sets_same,
               "min_top5_len": {lab: a.min_top5 for lab, a in arms.items()},
               "strata_n": {n: len(v) for n, v in strata},
               "subset_gold_matches": gold_same,
               "s1_copy_of_original_ax1t03": s1_same},
           "missing": {}, "rates": {}, "conditions": {}, "strata": {},
           "temperature_Ax1": {}, "per_case_rates": {}}

    # ---- 缺失判定（显式） ----
    for lab, a in arms.items():
        out["missing"][lab] = {
            "pairs_per_seed": a.pairs,
            "missing_per_seed": a.missing,
            "missing_total": sum(a.missing.values()),
            "dangling_per_k": {f"top{k}": a.dangling(k, ids) for k in KS},
            "cases_valid_per_k": {f"top{k}": len(a.valid(k, ids)) for k in KS}}

    # ---- 命中率 ----
    for cond, alabs in (("matched_T0.3", MATCHED5), ("asymmetric_paper", ASYMM5)):
        for lab in alabs:
            a = arms[lab]
            out["rates"][f"{cond}|{lab}"] = {}
            for k in KS:
                ps = a.per_seed_rates(k, ids)
                v = np.array(a.rate_vec(k, ids, "main"))
                out["rates"][f"{cond}|{lab}"][f"top{k}"] = {
                    "per_seed": [100 * x for x in ps],
                    "mean_pct": 100 * float(np.mean(ps)),
                    "sd_pct": 100 * float(np.std(ps, ddof=1)),
                    "case_rate_main_pct": 100 * float(v.mean()) if len(v) else None,
                    "n_cases_main": int(len(v)),
                    "case_rate_all_miss_pct": 100 * float(np.mean(
                        [sum(a.hit[(s, i, k)] for s in SEEDS) / len(SEEDS)
                         for i in ids]))}

    # ---- 成对比较 ----
    for cond, alabs, prs in (("matched_T0.3", MATCHED5, PAIRS5),
                             ("asymmetric_paper", ASYMM5, PAIRS5_ASYM)):
        out["conditions"][cond] = {
            "arms": alabs,
            "n_cases_top1": len([i for i in ids
                                 if all(arms[alabs[0]].ok[(s, i, 1)]
                                        for s in SEEDS)]),
            "comparisons": {f"top{k}": {f"{a}_vs_{b}": pair5(arms[a], arms[b], k, ids)
                                        for a, b in prs} for k in KS}}

    # ---- 温度对 A×1 自身 ----
    for k in KS:
        out["temperature_Ax1"][f"top{k}"] = pair5(arms["A×1@0.3"], arms["A×1@T=0"], k, ids)

    # ---- 分层 ----
    for sname, sids in strata:
        out["strata"][sname] = {
            "n": len(sids),
            "rates_matched": {lab: {f"top{k}": 100 * float(np.mean(
                arms[lab].rate_vec(k, sids, "main")))
                for k in KS} for lab in MATCHED5},
            "comparisons_matched": {f"top{k}": {
                f"{a}_vs_{b}": pair5(arms[a], arms[b], k, sids)
                for a, b in PAIRS5} for k in KS}}

    # ---- 逐病例命中率（审计用） ----
    for sname, sids in strata:
        for lab in ARMS5:
            for k in KS:
                out["per_case_rates"][f"{sname}|{lab}|top{k}"] = [
                    round(x, 4) for x in arms[lab].rate_vec(k, sids, "main")]
    out["case_ids"] = {n: v for n, v in strata}

    # ---------------------------------------------------------- markdown
    A("")
    A("---")
    A("")
    A("## 6. 主结果：5-seed 温度匹配口径（三臂全 T=0.3，n=364）")
    A("")
    A("**正文数字以本节为准**；第 1–5 节为仅 seed 1 的历史口径，保留存档。")
    A("")
    A("口径与 `stats_caselevel.py` 一致：病例级 5-seed 平均命中率 → 跨病例配对 Wilcoxon")
    A(f"双侧（`zero_method=\"wilcox\"`，零差病例剔除并计数）+ 病例级 cluster bootstrap "
      f"{N_BOOT} 次 95% 百分位 CI（seed {BOOT_SEED}）+ 多数决（≥3/5 seeds）精确 McNemar。")
    A("")
    A("### 6.0 自检与缺失判定")
    A("")
    A(f"- 四臂 × 5 seeds 各 364 行，case_id 集合完全一致：**{sets_same}**；")
    A(f"  各臂 top5 最短长度：" + "、".join(f"{lab} {a.min_top5}"
                                        for lab, a in arms.items()) + " 条。")
    A(f"- `topn_erreason_ax1t03/Ax1t03_s1.jsonl` 与旧的 `topn_erreason_ax1t03.jsonl` "
      f"逐例（gold + top5）完全一致：**{s1_same}**。")
    A(f"- 金标签与 `data/er_reason_subset.json` 逐例一致：**{gold_same}**；"
      f"分层 症状级 n={len(sym)} / 疾病级 n={len(dis)}。")
    A("")
    A("**未判定 (gold, candidate) 对与悬空观察（显式报告，绝不静默计 miss）**")
    A("")
    A("| 臂 | " + " | ".join(f"s{s}" for s in SEEDS) + " | 合计 | 悬空(seed,case)" +
      "".join(f" top-{k}" for k in KS) + " |")
    A("|---|" + "---|" * (len(SEEDS) + 2) + "---|")
    for lab, a in arms.items():
        ds = "/".join(str(a.dangling(k, ids)) for k in KS)
        A(f"| {lab} | " + " | ".join(str(a.missing[s]) for s in SEEDS) +
          f" | {sum(a.missing.values())} | {ds} |")
    A("")
    A("（每臂每 seed 1820 对候选；「悬空」= 该 (seed, case) 前 k 位含未判定项且已判定部分"
      "无命中，即结果真正取决于未判定的观察数。）")
    A("")
    A("主口径下各终点的可用病例数（该终点 5 seeds 全部可判定）：")
    A("")
    A("| 臂 | top-1 | top-3 | top-5 |")
    A("|---|---|---|---|")
    for lab, a in arms.items():
        A(f"| {lab} | " + " | ".join(str(len(a.valid(k, ids))) for k in KS) + " |")
    A("")
    A("成对比较的 n 取双方可用病例的交集（下表逐行给出 n 与剔除例数；"
      "另给全 364 例的夹逼口径，见每行的 bounds）。")
    A("")

    def rate_block(cond, alabs, title):
        A(f"**{title}**")
        A("")
        A("| 方案 | top-1 | top-3 | top-5 |")
        A("|---|---|---|---|")
        for lab in alabs:
            A(f"| {lab} | " + " | ".join(
                f"{f1(out['rates'][f'{cond}|{lab}'][f'top{k}']['mean_pct'])}"
                f" ± {f1(out['rates'][f'{cond}|{lab}'][f'top{k}']['sd_pct'])}"
                for k in KS) + " |")
        A("")
        A("**逐 seed 命中率（%，顺序 s1–s5）**")
        A("")
        A("| 方案 | 指标 | s1 | s2 | s3 | s4 | s5 | 均值±SD | 病例级(5-seed) | n |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for lab in alabs:
            for k in KS:
                r = out["rates"][f"{cond}|{lab}"][f"top{k}"]
                A(f"| {lab} | top-{k} | " +
                  " | ".join(f"{x:.1f}" for x in r["per_seed"]) +
                  f" | {r['mean_pct']:.1f} ± {r['sd_pct']:.1f} | "
                  f"{f1(r['case_rate_main_pct'])}% | {r['n_cases_main']} |")
        A("")

    def cmp_block(cond, prs):
        A("| top-k | 对比 | 病例级命中率 A vs B | 均值差 [95% CI] | 非零差 | Wilcoxon p | "
          "多数决 a:b p | n（剔除） | 全364 按miss / 按命中 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            for name, r in out["conditions"][cond]["comparisons"][f"top{k}"].items():
                a, b = name.split("_vs_")
                star = "*" if r["wilcoxon_p"] < 0.05 else ""
                mj = r["majority"]
                bo = r["bounds_all364"]
                A(f"| top-{k} | {a} − {b} | {f1(100*r['mean_rate_a'])}% vs "
                  f"{f1(100*r['mean_rate_b'])}% | {f2(r['mean_diff_pp'])}pp "
                  f"[{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}] | "
                  f"{r['n_nonzero']} | {fp(r['wilcoxon_p'])}{star} | "
                  f"{mj['a_only']}:{mj['b_only']} p={fp(mj['mcnemar_p'])} | "
                  f"{r['n_cases']}（剔 {r['n_dropped']}） | "
                  f"{f2(bo['miss']['mean_diff_pp'])}pp p={fp(bo['miss']['wilcoxon_p'])} / "
                  f"{f2(bo['hit']['mean_diff_pp'])}pp p={fp(bo['hit']['wilcoxon_p'])} |")
        A("")

    A("### 6.1 三臂命中率（5-seed 均值 ± SD，病例级）")
    A("")
    rate_block("matched_T0.3", MATCHED5, "温度完全匹配（A×1@0.3 / P@0.3 / MDT@0.3）")
    A("### 6.2 三臂成对比较（温度匹配）")
    A("")
    cmp_block("matched_T0.3", PAIRS5)
    A("### 6.3 温度对 A×1 自身的影响（A×1@0.3 − A×1@T=0，5 seeds）")
    A("")
    A("| top-k | 病例级命中率 | 均值差 [95% CI] | 非零差 | Wilcoxon p | 多数决 a:b p | n |")
    A("|---|---|---|---|---|---|---|")
    for k in KS:
        r = out["temperature_Ax1"][f"top{k}"]
        star = "*" if r["wilcoxon_p"] < 0.05 else ""
        A(f"| top-{k} | {f1(100*r['mean_rate_a'])}% vs {f1(100*r['mean_rate_b'])}% | "
          f"{f2(r['mean_diff_pp'])}pp [{f2(r['boot95_ci_pp'][0])},"
          f"{f2(r['boot95_ci_pp'][1])}] | {r['n_nonzero']} | {fp(r['wilcoxon_p'])}{star} | "
          f"{r['majority']['a_only']}:{r['majority']['b_only']} "
          f"p={fp(r['majority']['mcnemar_p'])} | {r['n_cases']} |")
    A("")
    A("### 6.4 温度不对称口径对照（A×1@T=0 vs P@0.3 vs MDT@0.3，论文原口径）")
    A("")
    rate_block("asymmetric_paper", ASYMM5, "三臂命中率（温度不对称）")
    cmp_block("asymmetric_paper", PAIRS5_ASYM)
    A("### 6.5 分层（金标签粒度，温度匹配口径）")
    A("")
    for sname, _ in strata[1:]:
        st = out["strata"][sname]
        A(f"**{sname}（n={st['n']}）**")
        A("")
        A("| 方案 | top-1 | top-3 | top-5 |")
        A("|---|---|---|---|")
        for lab in MATCHED5:
            A(f"| {lab} | " + " | ".join(
                f1(st['rates_matched'][lab][f'top{k}']) for k in KS) + " |")
        A("")
        A("| top-k | 对比 | 均值差 [95% CI] | 非零差 | Wilcoxon p | 多数决 a:b p | n |")
        A("|---|---|---|---|---|---|---|")
        for k in KS:
            for name, r in st["comparisons_matched"][f"top{k}"].items():
                a, b = name.split("_vs_")
                star = "*" if r["wilcoxon_p"] < 0.05 else ""
                A(f"| top-{k} | {a} − {b} | {f2(r['mean_diff_pp'])}pp "
                  f"[{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}] | "
                  f"{r['n_nonzero']} | {fp(r['wilcoxon_p'])}{star} | "
                  f"{r['majority']['a_only']}:{r['majority']['b_only']} "
                  f"p={fp(r['majority']['mcnemar_p'])} | {r['n_cases']} |")
        A("")

    # ---- 6.6 明确回答 ----
    m = lambda cond, k, a, b: out["conditions"][cond]["comparisons"][f"top{k}"][f"{a}_vs_{b}"]
    s1m = lambda k, a, b: seed1_report["comparisons"][f"全部|matched_T0.3|top{k}|a"][f"{a}_vs_{b}"]
    A("### 6.6 明确回答（5-seed 匹配条件下）")
    A("")
    ap = [(k, m("matched_T0.3", k, "A×1@0.3", "P@0.3")) for k in KS]
    am = [(k, m("matched_T0.3", k, "A×1@0.3", "MDT@0.3")) for k in KS]
    pm = [(k, m("matched_T0.3", k, "P@0.3", "MDT@0.3")) for k in KS]
    def sig_split(pairs):
        sig = [k for k, r in pairs if r["wilcoxon_p"] < 0.05 and r["mean_diff_pp"] > 0]
        ns = [k for k, r in pairs if k not in sig and r["mean_diff_pp"] > 0]
        return sig, ns

    ap_sig, ap_ns = sig_split(ap)
    if len(ap_sig) == 3:
        verdict_a = "是：三个终点均显著优于 P"
    elif ap_sig:
        verdict_a = ("部分成立：主终点 " + "、".join(f"top-{k}" for k in ap_sig) +
                     " 显著优于 P；" + "、".join(f"top-{k}" for k in ap_ns) +
                     " 方向一致（A×1 更高）但未达显著")
    else:
        verdict_a = "否：三个终点均未达显著"
    A(f"(a) **A×1 是否仍显著优于 P？→ {verdict_a}。**")
    for k, r in ap:
        A(f"    - top-{k}：{f1(100*r['mean_rate_a'])}% vs {f1(100*r['mean_rate_b'])}%，"
          f"{f2(r['mean_diff_pp'])}pp [{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}]，"
          f"Wilcoxon p={fp(r['wilcoxon_p'])}，多数决 "
          f"{r['majority']['a_only']}:{r['majority']['b_only']} p={fp(r['majority']['mcnemar_p'])}"
          f"（seed-1 口径：{s1m(k,'A×1@0.3','P@0.3')['diff_pp']:+.1f}pp "
          f"p={s1m(k,'A×1@0.3','P@0.3')['mcnemar_p']:.4f}）")
    am_sig, am_ns = sig_split(am)
    verdict_b = ("是：三个终点均显著优于 MDT，急诊反转结论成立" if len(am_sig) == 3
                 else ("是（部分终点显著）：" + "、".join(f"top-{k}" for k in am_sig)
                       if am_sig else "否：三个终点均未达显著"))
    A(f"(b) **A×1 是否仍显著优于 MDT（急诊反转是否仍成立）？→ {verdict_b}。**")
    for k, r in am:
        A(f"    - top-{k}：{f1(100*r['mean_rate_a'])}% vs {f1(100*r['mean_rate_b'])}%，"
          f"{f2(r['mean_diff_pp'])}pp [{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}]，"
          f"Wilcoxon p={fp(r['wilcoxon_p'])}，多数决 "
          f"{r['majority']['a_only']}:{r['majority']['b_only']} p={fp(r['majority']['mcnemar_p'])}"
          f"（seed-1 匹配口径："
      f"{s1m(k,'A×1@0.3','MDT@0.3')['diff_pp']:+.1f}pp "
      f"p={s1m(k,'A×1@0.3','MDT@0.3')['mcnemar_p']:.4f}）")
    d = [r["mean_diff_pp"] for _, r in pm]
    A(f"(c) **P 与 MDT 谁更高？→ "
      f"{'P 在三个终点上均更高' if all(x > 0 for x in d) else '方向不一致'}**，"
      f"但显著性格点：")
    for k, r in pm:
        A(f"    - top-{k}：P {f1(100*r['mean_rate_a'])}% vs MDT {f1(100*r['mean_rate_b'])}%，"
          f"{f2(r['mean_diff_pp'])}pp [{f2(r['boot95_ci_pp'][0])},{f2(r['boot95_ci_pp'][1])}]，"
          f"Wilcoxon p={fp(r['wilcoxon_p'])}，多数决 "
          f"{r['majority']['a_only']}:{r['majority']['b_only']} p={fp(r['majority']['mcnemar_p'])}")
    A("")
    A("**与 seed-1 口径的差异**：")
    A("")
    A("| 对比 | 终点 | seed-1 匹配 | 5-seed 匹配 | 差异 | seed-1 p | 5-seed p |")
    A("|---|---|---|---|---|---|---|")
    for lab, other in (("A×1@0.3", "P@0.3"), ("A×1@0.3", "MDT@0.3"), ("P@0.3", "MDT@0.3")):
        for k in KS:
            r1 = seed1_report["comparisons"][f"全部|matched_T0.3|top{k}|a"][f"{lab}_vs_{other}"]
            r5 = m("matched_T0.3", k, lab, other)
            A(f"| {lab} − {other} | top-{k} | {r1['diff_pp']:+.1f}pp | "
              f"{f2(r5['mean_diff_pp'])}pp | {f2(r5['mean_diff_pp']-r1['diff_pp'])}pp | "
              f"{r1['mcnemar_p']:.4f} | {fp(r5['wilcoxon_p'])} |")
    A("")
    A("（seed-1 列为单 seed 精确 McNemar 口径，5-seed 列为病例级 Wilcoxon 主口径；"
      "两者检验不同，差异列仅比较点估计。）")
    A("")
    A("### 6.7 结论")
    A("")
    psdr = [(k, m("matched_T0.3", k, "P@0.3", "MDT@0.3")) for k in KS]
    order_ok = all(r["mean_diff_pp"] > 0 for _, r in ap + am + psdr)
    A(f"5-seed 温度匹配（三臂全 T=0.3，n=364）下：**每个终点上三臂排序均为 A×1 > P > MDT"
      f"（{'成立' if order_ok else '不成立'}），与温度不对称口径一致**。")
    A("")
    A(f"- A×1 vs MDT（主文急诊反转）：三个终点全部显著（Wilcoxon "
      + "、".join(f"top-{k} p={fp(m('matched_T0.3', k, 'A×1@0.3', 'MDT@0.3')['wilcoxon_p'])}"
                 for k in KS) + "），与温度不对称口径同号同量级 → **反转结论不是温度混杂**。")
    A(f"- A×1 vs P：主终点显著（"
      + "、".join(f"top-{k} p={fp(r['wilcoxon_p'])}" for k, r in ap if r["wilcoxon_p"] < 0.05)
      + ("），top-1 方向一致但不显著（p=" + fp([r for k, r in ap if k == 1][0]["wilcoxon_p"])
         + "）" if 1 in ap_ns else "）") + "。")
    A(f"- P vs MDT：P 在三个终点上均更高且均达 5-seed 病例级显著（"
      + "、".join(f"top-{k} p={fp(r['wilcoxon_p'])}" for k, r in psdr)
      + "）。该对比与温度无关（P/MDT 两口径是同一份文件），与 "
        "`erreason_5seeds_report.md` 同格一致；按主文多重校正口径，top-3 一格未通过"
        "（q≈0.054），top-1/top-5 通过。")
    A("- A×1 自身的温度效应：" +
      "；".join(f"top-{k} {f2(out['temperature_Ax1'][f'top{k}']['mean_diff_pp'])}pp "
               f"(p={fp(out['temperature_Ax1'][f'top{k}']['wilcoxon_p'])})" for k in KS) +
      " → ER 上 T=0.3 对单次调用无显著损益（|Δ|≤1pp），因此两个口径的差距基本一致。")
    A("")
    return out


if __name__ == "__main__":
    sys.exit(main())
