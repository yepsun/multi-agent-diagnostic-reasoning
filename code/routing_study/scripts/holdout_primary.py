#!/usr/bin/env python3
"""留出集 46 例的预注册主终点（top-1/3/5）结果：三个切分 × 三策略 × 三终点，
两判官口径（GLM 主判官 / DS-v3 敏感性判官）。

统计口径**完全复用** routing_study/scripts/stats_caselevel.py（病例级 5-seed 平均命中率 →
配对 Wilcoxon 双侧；病例级 cluster bootstrap 10,000 次（seed 20260917）95% 百分位 CI；
多数决 ≥3/5 精确 McNemar）与 routing_study/scripts/recalc_main_judge.py（切分定义与取数）。
为避免 import 该脚本时触发其模块级写文件（会覆盖既有 stats_caselevel.json/.md），
本脚本把 topk / mcnemar / case_rates / case_majority / boot_ci 逐字复刻为本地函数，
N_BOOT=10000、RNG 种子 20260917、抽样公式（idx = RNG.integers(0, n, size=(N_BOOT, n))）
与百分位法（np.percentile 2.5/97.5）均与原脚本一致。
**唯一的顺序差异**：原脚本 splil 计算顺序是 (CPC87, MCR406)，本脚本先算 full87，
因此 full87（= CPC87）的 27 个 bootstrap 抽样序列与原脚本前 9 次调用**逐位相同**，
可用 stats_caselevel.json 的 CPC87 行做机器校验（见输出中的 stats_caselevel_ci_check）。

切分定义（两处等价，已核对）：held-out 46 = data/mgh_qa_dataset_new_cases.json 的 case_id；
dev 41 = routing_study/results/topn_ablation/ax1.jsonl 的 case_id；全量 87 = data/mgh_qa_dataset_merged.json
（校验：46 ∪ 41 = 87 且不相交）。

用法：./.venv/bin/python routing_study/scripts/holdout_primary.py [judge_cache_path]
默认 judge_cache_path = routing_study/results/judge_cache_glm_v3.json（GLM 主判官）。
每次运行写出 holdout46_primary_<judge_tag>.json，并把 results/ 下已存在的全部 <judge_tag> 文件
合并重写为 holdout46_primary.json（多判官对照）与 holdout46_primary.md。先跑 GLM 再跑 DS-v3，
主文件即同时包含两个判官口径。本脚本只读原始数据，不写任何既有结果文件。
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
DEFAULT_CACHE = B / "judge_cache_glm_v3.json"
OUT_JSON = B / "holdout46_primary.json"
OUT_MD = B / "holdout46_primary.md"

N_BOOT = 10000
RNG = np.random.default_rng(20260917)
KEY = lambda g, c: g[:150] + "||" + c[:150]

METHODS = ["Ax1", "P", "MDT"]
DISPLAY = {"Ax1": "A×1", "P": "P", "MDT": "MDT"}
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]
ENDPOINTS = (1, 3, 5)

# 先算 full87：使 bootstrap 抽样序列与 stats_caselevel.py 的 CPC87 部分一致（可校验）
COMPUTE_ORDER = ["full87", "heldout46", "dev41"]
REPORT_ORDER = ["heldout46", "dev41", "full87"]
SPLIT_TITLE = {
    "heldout46": "留出集 46 例（held-out，未参与提示词/判官规则开发）",
    "dev41": "开发集 41 例（dev，用于提示词 v2 与判官规则）",
    "full87": "全量 CPC 87 例（= 46 留出 + 41 开发，主文 Table 1 口径）",
}

JUDGE_ORDER = ["glm_v3", "dsflash_v3"]
JUDGE_TITLE = {
    "glm_v3": "GLM-5.3-flash × v3（主判官）",
    "dsflash_v3": "deepseek-flash × v3（敏感性判官）",
}


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


def ids_of(p):
    d = json.loads(Path(p).read_text())
    return [r["case_id"] for r in d]


# ---------------------------------------------------------------- 原始数据载入
runs = {}
for s in range(1, 6):
    runs[("Ax1", s)] = load(B / f"topn_seeds/Ax1_s{s}.jsonl")
    runs[("P", s)] = load(B / f"topn_seeds/P_s{s}.jsonl")
    runs[("MDT", s)] = load(
        B / "topn_mdt/synthesis.jsonl" if s == 1
        else B / f"topn_mdt/s{s}/synthesis.jsonl")

heldout_ids = ids_of(ROOT / "data" / "mgh_qa_dataset_new_cases.json")
dev_ids = list(load(B / "topn_ablation/ax1.jsonl"))
full_ids = ids_of(ROOT / "data" / "mgh_qa_dataset_merged.json")
SPLIT_IDS = {"heldout46": heldout_ids, "dev41": dev_ids, "full87": full_ids}
SPLIT_SOURCE = {
    "heldout46": "data/mgh_qa_dataset_new_cases.json",
    "dev41": "routing_study/results/topn_ablation/ax1.jsonl",
    "full87": "data/mgh_qa_dataset_merged.json",
}

split_check = {
    "heldout46_disjoint_from_dev41": not (set(heldout_ids) & set(dev_ids)),
    "heldout46_union_dev41_equals_full87": set(heldout_ids) | set(dev_ids) == set(full_ids),
    "n_heldout46": len(set(heldout_ids)), "n_dev41": len(set(dev_ids)),
    "n_full87": len(set(full_ids)),
    "run_files_cover_full87": set(runs[("Ax1", 1)]) == set(full_ids),
}
short_lists = sum(
    1 for (m, s), cases in runs.items() for rec in cases.values()
    if len(rec["top5"]) < 5)

# ------------------------------------------------- 判官缓存缺失计数（逐 case-seed）
cache_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CACHE
J = json.loads(cache_path.read_text())
stem = cache_path.stem
judge_tag = stem[len("judge_cache_"):] if stem.startswith("judge_cache_") else stem

missing_by_split = {}
for name, ids in SPLIT_IDS.items():
    per_ms = {m: {} for m in METHODS}
    n_missing = 0
    n_pairs = 0
    for m in METHODS:
        for s in range(1, 6):
            miss = tot = 0
            for cid in ids:
                rec = runs[(m, s)][cid]
                for c in rec["top5"][:5]:
                    tot += 1
                    if KEY(rec["gold"], c) not in J:
                        miss += 1
            per_ms[m][str(s)] = miss
            n_missing += miss
            n_pairs += tot
    missing_by_split[name] = {
        "n_candidate_pairs_checked": n_pairs,
        "n_missing_pairs": n_missing,
        "by_method_seed": per_ms,
    }

hits = {}
for (m, s), cases in runs.items():
    for cid, rec in cases.items():
        hits[(m, s, cid)] = [
            bool(J[KEY(rec["gold"], c)]) if KEY(rec["gold"], c) in J else None
            for c in rec["top5"][:5]]


# ------------------------------------------------------- 复刻 stats_caselevel.py
def topk(t, k):
    """前 k 个候选中任一为 True 即命中；全部缺失判官返回 None。"""
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


def case_rates(ids, m, k):
    """dict cid -> 5-seed 命中率（仅 5 seeds 全部可判定的病例）；另返回被丢弃病例。"""
    rates, dropped = {}, []
    for cid in ids:
        vals = [topk(hits[(m, s, cid)], k) for s in range(1, 6)]
        if any(v is None for v in vals):
            dropped.append(cid)
            continue
        rates[cid] = sum(vals) / 5.0
    return rates, dropped


def case_majority(ids, m, k):
    """dict cid -> 多数决（>=3/5 seeds 命中为 1；仅 5 seeds 全部可判定）。"""
    out = {}
    for cid in ids:
        vals = [topk(hits[(m, s, cid)], k) for s in range(1, 6)]
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


# ------------------------------------------------------------------ 主计算
results = {}
for name in COMPUTE_ORDER:
    ids = SPLIT_IDS[name]
    res = {
        "n_cases": len(set(ids)),
        "split_source": SPLIT_SOURCE[name],
        "judge_missing": missing_by_split[name],
        "endpoints": {},
    }
    for k in ENDPOINTS:
        rk = {m: case_rates(ids, m, k) for m in METHODS}
        maj = {m: case_majority(ids, m, k) for m in METHODS}
        end = {"strategies": {}, "comparisons": {}}
        for m in METHODS:
            rates = rk[m][0]
            evaluable = sorted(rates)
            per_seed = [
                sum(topk(hits[(m, s, c)], k) for c in evaluable) / len(evaluable)
                for s in range(1, 6)]
            end["strategies"][m] = {
                "n_evaluable_cases": len(evaluable),
                "n_dropped_cases": len(rk[m][1]),
                "dropped_cases": rk[m][1],
                "per_seed_rate": per_seed,
                "per_seed_pct": [round(x * 100, 4) for x in per_seed],
                "mean_pct": st.mean(per_seed) * 100,
                "sd_pct": st.stdev(per_seed) * 100,
                "case_level_mean_rate": st.mean(rates.values()),
                "majority_n_correct": sum(maj[m].values()),
            }
        for a, b in PAIRS:
            ra_d, rb_d = rk[a][0], rk[b][0]
            common = sorted(set(ra_d) & set(rb_d))
            ra = np.array([ra_d[c] for c in common])
            rb = np.array([rb_d[c] for c in common])
            diff = ra - rb
            wp = 1.0 if np.all(diff == 0) else float(wilcoxon(ra, rb, zero_method="wilcox").pvalue)
            md, lo, hi = boot_ci(diff)

            cm = sorted(set(maj[a]) & set(maj[b]))
            ao = sum(1 for c in cm if maj[a][c] and not maj[b][c])
            bo = sum(1 for c in cm if maj[b][c] and not maj[a][c])
            mp = mcnemar(ao, bo)

            pao = pbo = 0
            for s in range(1, 6):
                for c in ids:
                    ha = topk(hits[(a, s, c)], k)
                    hb = topk(hits[(b, s, c)], k)
                    if ha and not hb:
                        pao += 1
                    elif hb and not ha:
                        pbo += 1

            end["comparisons"][f"{a}_vs_{b}"] = {
                "n_cases_paired": len(common),
                "n_cases_nonzero_diff": int(np.count_nonzero(diff)),
                "mean_rate_a": float(ra.mean()) if len(ra) else None,
                "mean_rate_b": float(rb.mean()) if len(rb) else None,
                "mean_diff_pct": md * 100,
                "boot95_ci_pct": [lo * 100, hi * 100],
                "wilcoxon_p": wp,
                "majority": {"a_only": ao, "b_only": bo, "n_cases": len(cm), "mcnemar_p": mp},
                "pooled_mcnemar": {"a_only": pao, "b_only": pbo, "p": mcnemar(pao, pbo)},
            }
        res["endpoints"][str(k)] = end
    results[name] = res


# ------------------------------------------------------------------ 校验
checks = {"split_definition": split_check,
          "cases_with_fewer_than_5_candidate_lists": short_lists,
          "notes": []}

# 1) 与 stats_caselevel.json 的 CPC87（GLM）逐字段比对：验证 bootstrap/Wilcoxon 口径一致
sc_path = B / "stats_caselevel.json"
if judge_tag == "glm_v3" and sc_path.exists():
    sc = json.loads(sc_path.read_text())
    ref = sc.get("CPC87", {}).get("topk", {})
    max_boot = 0.0
    max_rate = 0.0
    max_wil = 0.0
    for k in ENDPOINTS:
        mine = results["full87"]["endpoints"][str(k)]
        for m in METHODS:
            max_rate = max(max_rate, abs(
                mine["strategies"][m]["case_level_mean_rate"]
                - ref[str(k)]["case_rate_mean"][m]))
        for pair in [f"{a}_vs_{b}" for a, b in PAIRS]:
            c1 = mine["comparisons"][pair]
            c2 = ref[str(k)]["comparisons"][pair]
            max_boot = max(max_boot, max(abs(x - y) for x, y in
                                         zip(c1["boot95_ci_pct"], [v * 100 for v in c2["boot95_ci"]])))
            max_wil = max(max_wil, abs(c1["wilcoxon_p"] - c2["wilcoxon_p"]))
    checks["stats_caselevel_ci_check"] = {
        "reference": "routing_study/results/stats_caselevel.json 的 CPC87 行（GLM 缓存）",
        "max_abs_diff_boot95_ci_pp": max_boot,
        "max_abs_diff_wilcoxon_p": max_wil,
        "max_abs_diff_case_rate_mean": max_rate,
        "identical": bool(max(max_boot, max_wil, max_rate) < 1e-9),
    }

# 2) 与 paper/evidence_bundle.md / 主文 Table 1 的已发表数字比对（仅 GLM）
if judge_tag == "glm_v3":
    ev = {}
    exp_held_top1 = {"Ax1": (55.7, 1.2), "P": (56.1, 3.2), "MDT": (60.0, 3.3)}
    for m, (mu, sd) in exp_held_top1.items():
        got = results["heldout46"]["endpoints"]["1"]["strategies"][m]
        ev[f"heldout46_top1_{m}"] = {
            "evidence_bundle": f"{mu}±{sd}",
            "recomputed": f"{got['mean_pct']:.1f}±{got['sd_pct']:.1f}",
            "match_1dp": (round(got["mean_pct"], 1) == mu and round(got["sd_pct"], 1) == sd),
        }
    exp_full = {"Ax1": [(57.9, 1.9), (76.1, 4.1), (81.1, 2.6)],
                "P": [(53.8, 4.0), (72.2, 2.6), (76.8, 2.1)],
                "MDT": [(60.5, 3.1), (81.6, 2.4), (85.5, 2.1)]}
    for m, vals in exp_full.items():
        for i, k in enumerate(ENDPOINTS):
            got = results["full87"]["endpoints"][str(k)]["strategies"][m]
            mu, sd = vals[i]
            ev[f"full87_top{k}_{m}"] = {
                "manuscript_table1": f"{mu}±{sd}",
                "recomputed": f"{got['mean_pct']:.1f}±{got['sd_pct']:.1f}",
                "match_1dp": (round(got["mean_pct"], 1) == mu and round(got["sd_pct"], 1) == sd),
            }
    checks["published_number_check"] = ev

# 3) 旧文件 holdout46_metrics.json（DS-v3 口径过时产物）一致性核对
old_path = B / "holdout46_metrics.json"
if judge_tag == "dsflash_v3" and old_path.exists():
    old = json.loads(old_path.read_text())
    cmp_res = {}
    max_raw = 0.0
    for m in METHODS:
        for k in ENDPOINTS:
            per = results["heldout46"]["endpoints"][str(k)]["strategies"][m]["per_seed_rate"]
            ref = old[DISPLAY[m]][str(k)]
            d_raw = max(abs(a - b) for a, b in zip(per, ref))
            max_raw = max(max_raw, d_raw)
            cmp_res[f"{m}_top{k}"] = {
                "holdout46_metrics_json": [round(x, 6) for x in ref],
                "recomputed_6dp": [round(x, 6) for x in per],
                "max_abs_diff_raw": d_raw,
                "identical_at_6dp": d_raw < 1e-6,
            }
    checks["holdout46_metrics_json_check"] = {
        "file": "routing_study/results/holdout46_metrics.json",
        "judge": "DS-v3（该文件为旧口径产物，本脚本不修改它）",
        "tolerance": "旧文件按 6 位小数存储，逐格比较容差 1e-6",
        "max_abs_diff_raw": max_raw,
        "all_identical": all(v["identical_at_6dp"] for v in cmp_res.values()),
        "per_cell": cmp_res,
    }

payload = {
    "judge_tag": judge_tag,
    "judge_cache": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
    "n_cache_entries": len(J),
    "split_check": split_check,
    "protocol": {
        "endpoint_metric": "top-k 召回：金诊断出现在策略输出的前 k 个候选中",
        "per_seed_rate": "单 seed 的 top-k 命中率（跨该切分全部病例）",
        "mean_sd_across_seeds": "5 个 seed 命中率的算术均值与样本标准差（ddof=1），单位为百分点",
        "case_level_primary": "每病例先取 5-seed 平均命中率（0-1），再跨病例做配对 Wilcoxon 符号秩检验（双侧，zero_method='wilcox'）",
        "bootstrap": "病例级 cluster bootstrap 10,000 次（numpy default_rng seed 20260917），均值差 95% 百分位 CI",
        "majority_mcnemar": "病例在 >=3/5 seeds 命中记为对，精确二项 McNemar（双侧）",
        "pooled_mcnemar": "旧口径：跨 5 seeds 合并的配对 McNemar（违反独立性，仅作对照）",
        "reference_implementation": "routing_study/scripts/stats_caselevel.py（topk/mcnemar/case_rates/case_majority/boot_ci 逐字复刻）；切分与取数同 routing_study/scripts/recalc_main_judge.py",
    },
    "checks": checks,
    "splits": results,
}

tag_file = B / f"holdout46_primary_{judge_tag}.json"
tag_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
print(f"写出 {tag_file}")

# --------------------------------------------- 合并磁盘上已有的全部判官结果
merged = {}
for f in B.glob("holdout46_primary_*.json"):
    d = json.loads(f.read_text())
    merged[d["judge_tag"]] = d
order = [t for t in JUDGE_ORDER if t in merged] + sorted(t for t in merged if t not in JUDGE_ORDER)
merged = {t: merged[t] for t in order}
OUT_JSON.write_text(json.dumps({"judges": merged}, ensure_ascii=False, indent=2))
print(f"写出 {OUT_JSON}（判官：{', '.join(merged)}）")


# ------------------------------------------------------------------ Markdown
def fmt_p(p):
    if p is None:
        return "NA"
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


date = "2026-09-19"
L = []
L.append("# 留出集 46 例主终点结果（top-1/3/5，两判官口径）")
L.append("")
L.append(f"生成脚本：`routing_study/scripts/holdout_primary.py`（{date}）。"
         f"包含判官口径：{'、'.join(JUDGE_TITLE.get(t, t) for t in merged)}。")
L.append("")
L.append("> **关于 `routing_study/results/holdout46_metrics.json`**：该文件是 **DS-v3 判官口径的过时产物**"
         "（仅覆盖留出集、仅逐 seed 命中率，无 top-3/top-5 的统计检验）。本文件保留它不修改，")
L.append("> 并在下文的 DS-v3 一节给出逐格一致性核对；论文正文与补充材料以本文件为准。")
L.append("")
L.append("## 口径说明")
L.append("")
L.append("本文目的：补齐主文缺口——**未参与提示词/判官规则开发的留出集 46 例**上，"
         "预注册主终点 top-3/top-5 的完整数字与配对检验；同时给出开发集 41 例、"
         "全量 87 例的同口径数字，以及 GLM 主判官与 DS-v3 敏感性判官两套口径。")
L.append("")
L.append("- **主终点（预注册）**：top-3 / top-5 召回；top-1 为次要终点。")
L.append("- **`×5 seeds 均值±SD`**：单 seed 命中率（跨该切分全部病例），5 个 seed 的均值与样本 SD，单位百分点。")
L.append("- **病例级主口径**：每病例先按 5 seeds 取平均命中率（0–1 连续值），再跨病例配对 Wilcoxon 双侧符号秩检验"
         "（`zero_method='wilcox'`，即差值为 0 的病例不计入检验；「非零差例数」列为参与检验的病例数）。")
L.append("- **CI**：病例级 cluster bootstrap（按病例重抽样 10,000 次，seed 20260917）均值差 95% 百分位 CI。")
L.append("- **多数决**：病例在 ≥3/5 seeds 命中记为对，精确二项 McNemar 双侧检验；`a:b` = 仅 a 对:仅 b 对(例数)。")
L.append("- 统计函数逐字复刻 `routing_study/scripts/stats_caselevel.py`（`topk`/`mcnemar`/`case_rates`/"
         "`case_majority`/`boot_ci`），未另创口径。")
L.append("")

for tag in order:
    p = merged[tag]
    L.append(f"## 判官：{JUDGE_TITLE.get(tag, tag)}")
    L.append("")
    L.append(f"缓存：`{p['judge_cache']}`（{p['n_cache_entries']} 对）；"
             f"切分校验：46 ∪ 41 = 87 且不相交 = "
             f"{p['split_check']['heldout46_union_dev41_equals_full87'] and p['split_check']['heldout46_disjoint_from_dev41']}。")
    L.append("")
    for name in REPORT_ORDER:
        res = p["splits"][name]
        L.append(f"### {SPLIT_TITLE[name]}（n={res['n_cases']}）")
        L.append("")
        mm = res["judge_missing"]
        L.append(f"判官缓存缺失候选对：**{mm['n_missing_pairs']} / {mm['n_candidate_pairs_checked']}**"
                 f"（逐 (方案, seed) 明细见 JSON；缺失对不作 miss 处理，若某 case-seed 的 top-k "
                 f"全部不可判定则该病例从病例级分析中剔除并在此计数）。")
        n_drop = {m: res["endpoints"]["5"]["strategies"][m]["n_dropped_cases"] for m in METHODS}
        L.append("")
        L.append(f"病例级分析可用病例数（top-5）：A×1 {res['endpoints']['5']['strategies']['Ax1']['n_evaluable_cases']}、"
                 f"P {res['endpoints']['5']['strategies']['P']['n_evaluable_cases']}、"
                 f"MDT {res['endpoints']['5']['strategies']['MDT']['n_evaluable_cases']}"
                 f"（因判官缺失被剔除的病例数：{n_drop['Ax1']}/{n_drop['P']}/{n_drop['MDT']}）。")
        L.append("")
        L.append("**三策略 × 三终点（5-seed 均值 ± SD，百分点）**")
        L.append("")
        L.append("| top-k | A×1 | P | MDT |")
        L.append("|---|---|---|---|")
        for k in ENDPOINTS:
            row = [f"top-{k}"]
            for m in METHODS:
                s = res["endpoints"][str(k)]["strategies"][m]
                row.append(f"{s['mean_pct']:.1f}±{s['sd_pct']:.1f}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L.append("**逐 seed 命中率（%，顺序 s1–s5）**")
        L.append("")
        L.append("| top-k | A×1 | P | MDT |")
        L.append("|---|---|---|---|")
        for k in ENDPOINTS:
            row = [f"top-{k}"]
            for m in METHODS:
                row.append(" / ".join(f"{v:.1f}" for v in
                                      res["endpoints"][str(k)]["strategies"][m]["per_seed_pct"]))
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L.append("**配对比较（病例级：均值差 [95% bootstrap CI]，Wilcoxon p；多数决 McNemar）**")
        L.append("")
        L.append("| top-k | 对比 | 命中率 A vs B | 均值差 [95% CI] | 非零差例数 | Wilcoxon p | 多数决 a:b p | 合并McNemar a:b p |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in ENDPOINTS:
            for a, b in PAIRS:
                c = res["endpoints"][str(k)]["comparisons"][f"{a}_vs_{b}"]
                lo, hi = c["boot95_ci_pct"]
                L.append(
                    f"| top-{k} | {DISPLAY[a]} vs {DISPLAY[b]} | "
                    f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                    f"{c['mean_diff_pct']:+.1f}pp [{lo:+.1f}, {hi:+.1f}] | "
                    f"{c['n_cases_nonzero_diff']}/{c['n_cases_paired']} | "
                    f"{fmt_p(c['wilcoxon_p'])}{sig(c['wilcoxon_p'])} | "
                    f"{c['majority']['a_only']}:{c['majority']['b_only']} "
                    f"p={fmt_p(c['majority']['mcnemar_p'])}{sig(c['majority']['mcnemar_p'])} | "
                    f"{c['pooled_mcnemar']['a_only']}:{c['pooled_mcnemar']['b_only']} "
                    f"p={fmt_p(c['pooled_mcnemar']['p'])}{sig(c['pooled_mcnemar']['p'])} |")
        L.append("")
        L.append("（`*` p<0.05；「命中率 A vs B」为病例级 5-seed 均值命中率，仅作描述；"
                 "「非零差例数」= 该比较中两方案 5-seed 平均命中率不相等的病例数，"
                 "即 Wilcoxon 实际使用的事件数。）")
        L.append("")

    # 校验
    ck = p["checks"]
    L.append("### 校验")
    L.append("")
    if "stats_caselevel_ci_check" in ck:
        c = ck["stats_caselevel_ci_check"]
        L.append(f"- 与 `stats_caselevel.json` 的 CPC87 行逐字段比对："
                 f"bootstrap CI 最大绝对差 {c['max_abs_diff_boot95_ci_pp']:.2e}pp、"
                 f"Wilcoxon p 最大绝对差 {c['max_abs_diff_wilcoxon_p']:.2e}、"
                 f"病例级命中率最大绝对差 {c['max_abs_diff_case_rate_mean']:.2e} → "
                 f"**{'完全一致' if c['identical'] else '存在差异（需排查）'}**。")
    if "published_number_check" in ck:
        bad = [k for k, v in ck["published_number_check"].items() if not v["match_1dp"]]
        L.append(f"- 与 `paper/evidence_bundle.md` / 主文 Table 1 已发表数字比对（保留 1 位小数）："
                 f"共 {len(ck['published_number_check'])} 格，"
                 f"{'全部一致' if not bad else '不一致 ' + ', '.join(bad)}。")
    if "holdout46_metrics_json_check" in ck:
        c = ck["holdout46_metrics_json_check"]
        L.append(f"- 与旧文件 `holdout46_metrics.json`（DS-v3 口径）逐格比对："
                 f"**{'9/9 逐 seed 数值一致（6 位小数）' if c['all_identical'] else '存在差异（见 JSON）'}**"
                 f"（最大原始绝对差 {c['max_abs_diff_raw']:.2e}，旧文件按 6 位小数存储）"
                 f"——确认该旧文件即 DS-v3 留出集口径，本文件不改动它。")
    L.append(f"- 病例候选列表不足 5 条的情形：{ck['cases_with_fewer_than_5_candidate_lists']} 例。")
    L.append("")

# 两判官对照
if len(order) > 1 and all("heldout46" in merged[t]["splits"] for t in order):
    L.append("## 两判官在留出集 46 例上的一致性对照")
    L.append("")
    L.append("| 口径 | 指标 | " + " | ".join(DISPLAY[m] for m in METHODS) + " |")
    L.append("|" + "---|" * (2 + len(METHODS)))
    for tag in order:
        for k in ENDPOINTS:
            s = merged[tag]["splits"]["heldout46"]["endpoints"][str(k)]["strategies"]
            L.append(f"| {tag} | top-{k} | " + " | ".join(
                f"{s[m]['mean_pct']:.1f}±{s[m]['sd_pct']:.1f}" for m in METHODS) + " |")
    L.append("")
    L.append("| 对比 | top-k | " + " | ".join(
        f"{t}: 均值差 [95% CI], Wilcoxon p（合并McNemar p）" for t in order) + " |")
    L.append("|" + "---|" * (2 + len(order)))
    for a, b in PAIRS:
        for k in ENDPOINTS:
            cells = []
            for tag in order:
                c = merged[tag]["splits"]["heldout46"]["endpoints"][str(k)]["comparisons"][f"{a}_vs_{b}"]
                lo, hi = c["boot95_ci_pct"]
                cells.append(f"{c['mean_diff_pct']:+.1f}pp [{lo:+.1f}, {hi:+.1f}], "
                             f"p={fmt_p(c['wilcoxon_p'])}{sig(c['wilcoxon_p'])} "
                             f"（p={fmt_p(c['pooled_mcnemar']['p'])}{sig(c['pooled_mcnemar']['p'])}）")
            L.append(f"| {DISPLAY[a]} vs {DISPLAY[b]} | top-{k} | " + " | ".join(cells) + " |")
    L.append("")
    if len(order) > 1:
        L.append("**方向与显著性一致性（逐格）**：")
        L.append("")
        L.append("| 对比 | top-k | 方向一致 | 病例级 Wilcoxon 显著性结论一致 | 合并 McNemar 显著性结论一致 |")
        L.append("|---|---|---|---|---|")
        for a, b in PAIRS:
            for k in ENDPOINTS:
                cells = {t: merged[t]["splits"]["heldout46"]["endpoints"][str(k)]["comparisons"][f"{a}_vs_{b}"]
                         for t in order}
                signs = {t: (cells[t]["mean_diff_pct"] > 0) for t in order}
                sigs = {t: (cells[t]["wilcoxon_p"] < 0.05) for t in order}
                psigs = {t: (cells[t]["pooled_mcnemar"]["p"] < 0.05) for t in order}
                L.append(f"| {DISPLAY[a]} vs {DISPLAY[b]} | top-{k} | "
                         f"{'是' if len(set(signs.values())) == 1 else '否'} | "
                         f"{'是' if len(set(sigs.values())) == 1 else '否'} | "
                         f"{'是' if len(set(psigs.values())) == 1 else '否'} |")
        L.append("")
        a, b = "MDT", "Ax1"
        L.append("**MDT vs A×1 主终点（预注册 top-3/top-5）在留出集上的口径依赖**：")
        L.append("")
        for k in (3, 5):
            cells = {t: merged[t]["splits"]["heldout46"]["endpoints"][str(k)]["comparisons"][f"{a}_vs_{b}"]
                     for t in order}
            sig_txt = ("两判官均未达 0.05" if all(cells[t]["wilcoxon_p"] >= 0.05 for t in order)
                       else "至少一个判官 p<0.05")
            L.append(f"- top-{k}：病例级主口径下 " + "；".join(
                f"{t} 均值差 {cells[t]['mean_diff_pct']:+.1f}pp "
                f"[{cells[t]['boot95_ci_pct'][0]:+.1f}, {cells[t]['boot95_ci_pct'][1]:+.1f}]，"
                f"Wilcoxon p={fmt_p(cells[t]['wilcoxon_p'])}" for t in order)
                + f"（{sig_txt}）；"
                + "旧合并 McNemar 口径下 " + "；".join(
                    f"{t} {cells[t]['pooled_mcnemar']['a_only']}:{cells[t]['pooled_mcnemar']['b_only']} "
                    f"p={fmt_p(cells[t]['pooled_mcnemar']['p'])}{sig(cells[t]['pooled_mcnemar']['p'])}"
                    for t in order) + "。")
        L.append("")
    L.append("解读要点：留出集上两判官对 **同一次比较的方向与显著性** 是否一致，"
             "决定该格能否作为稳健结论写入正文；涉及判官结论不一致的格，正文只报数值、不下优越性断言。"
             "「合并 McNemar」为旧口径（跨 5 seeds 合并，违反独立性），"
             "仅用于与既有补充材料保持对照，结论以病例级主口径为准。")
    L.append("")

OUT_MD.write_text("\n".join(L) + "\n")
print(f"写出 {OUT_MD}")
