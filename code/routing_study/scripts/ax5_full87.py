#!/usr/bin/env python3
"""A×5 预算匹配对照（87 例 × 5 seeds）+ 自洽性变体，GLM-5.3-flash × v3 判官计分。

回应评审 (a)：原 A×5 对照只跑了 41 例开发集、且用旧版 v2 判官（deepseek-flash），
无法支撑"结构而非算力带来收益"的核心归因。本脚本把它扩到全部 87 例 × 5 seeds，
并用主判官口径（GLM-5.3-flash × judge_v3.V3_PROMPT，共享缓存
results/judge_cache_glm_v3.json）重新计分。

两条臂（同一批样本）：
- Ax5   ：每例每 seed 采 5 份独立样本（T=0.7），用 topn_cpc.aggregate_top5
          （Borda，rank1 权重 1.0 → rank5 0.2，same_disease 同病簇合并）聚合。
- Ax5_sc：自洽性（self-consistency）变体——rank-1 做疾病聚类多数投票定 top-1
          （平票按 Borda 分数），其余位置按 Borda 序补齐。多智能体文献里
          最常见的预算匹配基线正是 self-consistency voting。

推理：DashScope qwen3.8-flash（provider="qwen"，disable_thinking，max_tokens=2048），
提示词与主实验逐字一致（topn_cpc_promptv2.A_TOPN_PROMPT），病例文本
topn_cpc_promptv2_87.load_merged()。

阶段：PHASE=infer|judge|analyze|all（默认 all）。各阶段断点续跑。
  infer   → results/topn_seeds_ax5/Ax5_s{1..5}.jsonl
            （每行 {"case_id","gold","top5","top5_sc","samples","total_tokens"}）
  judge   → 只补 GLM v3 缓存中缺失的 (gold, cand) 对，两个变体都判
  analyze → results/ax5_full87.json + .md：5-seed 均值±SD 的 top-1/3/5，
            与 A×1（topn_seeds）、P、MDT（topn_mdt）做病例级配对检验
            （配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10000 次 95% CI
            + 多数决精确 McNemar），口径同 stats_caselevel.py；
            另给 held-out 46 例与 dev 41 例子集。

环境变量：AX5_OUTDIR（输出目录，默认 results/topn_seeds_ax5）、
AX5_SEEDS（默认 1,2,3,4,5）、AX5_LIMIT（只跑前 N 例，冒烟用）、
GLM_CACHE（判官缓存路径，默认共享的 judge_cache_glm_v3.json，正式运行勿改）。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import (load_done, append_row, call_top5, aggregate_top5,  # noqa: E402
                      MAX_WORKERS)
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from webapp.clustering import same_disease  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("AX5_OUTDIR", RESULTS / "topn_seeds_ax5"))
SEEDS = [int(s) for s in os.environ.get("AX5_SEEDS", "1,2,3,4,5").split(",")
         if s.strip()]
LIMIT = int(os.environ.get("AX5_LIMIT", "0"))
A5_SAMPLES = 5
A5_TEMP = 0.7
MAX_ATTEMPTS = 3  # 空输出重试次数（参照 ax1_t03_sensitivity.py）
OUT_JSON = RESULTS / "ax5_full87.json"
OUT_MD = RESULTS / "ax5_full87.md"
PAIRS = [("Ax5", "Ax1"), ("Ax5", "P"), ("Ax5", "MDT"),
         ("Ax5_sc", "Ax1"), ("Ax5_sc", "P"), ("Ax5_sc", "MDT"),
         ("Ax5_sc", "Ax5")]
ARM_LABEL = {"Ax5": "A×5 (Borda 聚合)", "Ax5_sc": "A×5 (自洽性多数投票)",
             "Ax1": "A×1 (T=0)", "P": "P (T=0.3)", "MDT": "MDT (T=0.3)"}


def row_path(seed):
    return OUTDIR / f"Ax5_s{seed}.jsonl"


# ---------- 聚合：Borda（复用 topn_cpc.aggregate_top5）+ 自洽性变体 ----------

def borda_clusters(sample_lists):
    """与 topn_cpc.aggregate_top5 逐行一致的贪心同病簇聚合，返回簇列表。

    仅把 aggregate_top5 的簇构造抽出来（供自洽性变体读分数/票数用），
    top5 本身仍调用 aggregate_top5，保证与既有 A×5 口径完全一致。"""
    clusters = []
    for lst in sample_lists:
        for rank, dx in enumerate(lst[:5], start=1):
            w = (6 - rank) / 5.0
            placed = False
            for c in clusters:
                if same_disease(c["rep"], dx):
                    c["score"] += w
                    c["names"].append(dx)
                    c["primaries"] += 1 if rank == 1 else 0
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": dx, "names": [dx], "score": w,
                                 "primaries": 1 if rank == 1 else 0})
    return clusters


def cluster_rep(cluster):
    """簇代表名 = 簇内出现次数最多的原始写法（同 aggregate_top5）。"""
    counts = {}
    for n in cluster["names"]:
        counts[n] = counts.get(n, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def borda_order(clusters):
    return sorted(clusters, key=lambda c: (c["score"], c["primaries"]),
                  reverse=True)


def aggregate_top5_sc(sample_lists):
    """自洽性变体：rank-1 疾病聚类多数投票 → top-1；其余按 Borda 序补齐。"""
    order = borda_order(borda_clusters(sample_lists))
    idx_of_name = {}
    for i, c in enumerate(order):
        for nm in c["names"]:
            idx_of_name.setdefault(nm, i)

    r1 = []
    for lst in sample_lists:
        if not lst:
            continue
        dx = lst[0]
        for c in r1:
            if same_disease(c["rep"], dx):
                c["names"].append(dx)
                c["votes"] += 1
                break
        else:
            r1.append({"rep": dx, "names": [dx], "votes": 1})
    if not r1:
        return []
    for c in r1:
        idx = min((idx_of_name.get(nm, len(order)) for nm in c["names"]),
                  default=len(order))
        c["borda_idx"] = idx
        c["borda_score"] = order[idx]["score"] if idx < len(order) else 0.0

    win = max(r1, key=lambda c: (c["votes"], c["borda_score"]))
    out = [cluster_rep(win)]
    for i, c in enumerate(order):
        rep = cluster_rep(c)
        if i == win["borda_idx"] or same_disease(rep, win["rep"]):
            continue
        if rep.lower() in [x.lower() for x in out]:
            continue
        out.append(rep)
        if len(out) == 5:
            break
    return out


# ---------- 阶段 1：推理 ----------

def run_seed(seed, cases):
    path = row_path(seed)
    done = load_done(path)
    todo = [c for c in cases
            if len(done.get(c["case_id"], {}).get("samples", [])) < A5_SAMPLES]
    print(f"[Ax5 s{seed}] 已完成 "
          f"{len(cases) - len(todo)}/{len(cases)}，待跑 {len(todo)}", flush=True)

    def work(c):
        samples = list(done.get(c["case_id"], {}).get("samples", []))
        tokens_total = 0
        while len(samples) < A5_SAMPLES:
            got = None
            for attempt in range(MAX_ATTEMPTS):
                top5, tokens = call_top5(
                    A_TOPN_PROMPT.format(case_text=c["text"]),
                    temperature=A5_TEMP)
                tokens_total += tokens
                if top5:
                    got = top5
                    break
                print(f"[Ax5 s{seed}] {c['case_id'][:40]} 空输出，"
                      f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
            if got is None:
                raise RuntimeError(f"{c['case_id']} 连续空输出")
            samples.append(got)
        top5 = aggregate_top5(samples)
        check = [cluster_rep(c)
                 for c in borda_order(borda_clusters(samples))[:5]]
        if check != top5:
            print(f"[Ax5 s{seed}] 警告：Borda 复算与 aggregate_top5 不一致 "
                  f"{top5} vs {check}", flush=True)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "top5_sc": aggregate_top5_sc(samples), "samples": samples,
                "total_tokens": tokens_total}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"[Ax5 s{seed}] {c['case_id'][:40]} 失败: {e}", flush=True)
                continue
            append_row(path, row)
            print(f"[Ax5 s{seed}] {i}/{len(todo)} {row['case_id'][:40]}: "
                  f"{row['top5'][:1]} | sc {row['top5_sc'][:1]}", flush=True)


# ---------- 阶段 2：判分（GLM v3，只补缺失对，两个变体都判） ----------

def load_arm(seed):
    return load_done(row_path(seed))


def run_judge():
    rows = []
    for seed in SEEDS:
        for row in load_arm(seed).values():
            rows.append(row)
            rows.append({**row, "top5": row.get("top5_sc") or []})
    n_before = len(json.loads(cs.GLM_CACHE.read_text())) if cs.GLM_CACHE.exists() else 0
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)
    return cache


# ---------- 阶段 3：分析 ----------

def build_arms():
    ax5 = {s: load_arm(s) for s in SEEDS}
    refs = {
        "Ax1": {s: cs.load(RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl")
                for s in SEEDS},
        "P": {s: cs.load(RESULTS / "topn_seeds" / f"P_s{s}.jsonl")
              for s in SEEDS},
        "MDT": {s: cs.load(RESULTS / "topn_mdt"
                           / ("synthesis.jsonl" if s == 1
                              else f"s{s}/synthesis.jsonl"))
                for s in SEEDS},
    }
    arms = {"Ax5": ax5, "Ax5_sc": cs.with_variant(ax5, "top5_sc"), **refs}
    return arms


def render_md(meta, stats, splits_order):
    L = []
    L.append("# A×5 预算匹配对照（87 例 × 5 seeds，GLM-5.3-flash × v3 判官）\n")
    L.append("回应评审 (a)：原 A×5 只跑 41 例 dev 集、用旧版 v2 判官，不能支撑"
             "“结构而非算力带来收益”的核心归因。本表把 A×5 扩到全部 87 例 × 5 "
             "seeds，并用主判官（GLM-5.3-flash × v3 规则，共享缓存 "
             "`results/judge_cache_glm_v3.json`）计分；同时给出预算匹配的 "
             "self-consistency 变体（rank-1 疾病聚类多数投票）。\n")
    L.append(f"- 推理：qwen3.8-flash，A×5 每例每 seed 5 份 T=0.7 独立样本，"
             f"共 {meta['n_cases']} 例 × {meta['n_seeds']} seeds × 5 = "
             f"{meta['n_infer_calls']} 次调用")
    L.append("- 聚合：Borda（rank-1 权重 1.0 → rank-5 0.2，同病簇合并，"
             "函数复用 `topn_cpc.aggregate_top5`）；`Ax5_sc` 的 top-1 由 rank-1 "
             "疾病聚类多数投票决定（平票取 Borda 分数高者），其余位置按 Borda 序补齐")
    L.append(f"- 判官缺失对（按 miss 计入 top-k 失败）："
             f"{ {k: v['judge_missing_pairs'] for k, v in stats.items()} }")
    L.append(f"- 样本路径：`{meta['outdir']}/Ax5_s{{1..5}}.jsonl`；"
             f"生成脚本 `routing_study/scripts/ax5_full87.py`\n")

    for name in splits_order:
        res = stats[name]
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 | 病例级命中率 top-1/3/5 |")
        L.append("|---|---|---|---|---|")
        for arm in ("Ax5", "Ax5_sc", "Ax1", "P", "MDT"):
            if arm not in res["arms"]:
                continue
            a = res["arms"][arm]
            cr = a["case_rate_mean"]
            crm = "/".join(f"{cr[f'top{k}']*100:.1f}%"
                           if cr[f"top{k}"] is not None else "NA"
                           for k in (1, 3, 5))
            L.append(f"| {ARM_LABEL[arm]} | {cs.acc_cell(a['mean_sd'], 1)} | "
                     f"{cs.acc_cell(a['mean_sd'], 3)} | "
                     f"{cs.acc_cell(a['mean_sd'], 5)} | {crm} |")
        L.append("\n（百分比为 5 seeds 准确率均值 ± SD；其余为病例级 5-seed 均值）\n")

        L.append("| 对比 | top-k | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
                 "多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p |")
        L.append("|---|---|---|---|---|---|---|")
        for pair, by_k in res["comparisons"].items():
            a, b = pair.split("_vs_")
            for k in (1, 3, 5):
                c = by_k[f"top{k}"]
                if c["mean_rate_a"] is None:
                    continue
                lo, hi = c["boot95_ci"]
                maj, old = c["majority"], c["pooled_mcnemar"]
                L.append(
                    f"| {ARM_LABEL[a]} vs {ARM_LABEL[b]} | top-{k} | "
                    f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                    f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                    f"{maj['a_only']}:{maj['b_only']} p={cs.fmt_p(maj['mcnemar_p'])}"
                    f"{cs.sig(maj['mcnemar_p'])} | "
                    f"{old['a_only']}:{old['b_only']} p={cs.fmt_p(old['p'])}"
                    f"{cs.sig(old['p'])} |")
        L.append("")

    full = stats.get("full87")
    if full:
        def line(pair, k, la=None, lb=None):
            c = full["comparisons"].get(pair, {}).get(f"top{k}")
            if c is None or c["mean_rate_a"] is None:
                return f"{pair} top-{k}: 数据不足"
            a, b = (la, lb) if la else pair.split("_vs_")
            return (f"{a} {c['mean_rate_a']*100:.1f}% vs "
                    f"{b} {c['mean_rate_b']*100:.1f}%（差 "
                    f"{c['mean_diff']*100:+.1f}pp, 95% CI "
                    f"[{c['boot95_ci'][0]*100:+.1f}, {c['boot95_ci'][1]*100:+.1f}], "
                    f"Wilcoxon p={cs.fmt_p(c['wilcoxon_p'])}, 多数决 McNemar "
                    f"p={cs.fmt_p(c['majority']['mcnemar_p'])}）")
        L.append("\n## 结论（全量 87 例，核心归因）\n")
        L.append(f"1. 预算匹配：{line('Ax5_vs_Ax1', 1)}；{line('Ax5_vs_Ax1', 5)}")
        L.append(f"2. 自洽性变体 vs A×1：{line('Ax5_sc_vs_Ax1', 1)}；"
                 f"{line('Ax5_sc_vs_Ax1', 5)}")
        L.append(f"3. 结构 vs 算力（关键，Borda 基线）："
                 f"{line('Ax5_vs_MDT', 1, 'A×5(Borda)', 'MDT')}；"
                 f"{line('Ax5_vs_MDT', 3, 'A×5(Borda)', 'MDT')}；"
                 f"{line('Ax5_vs_MDT', 5, 'A×5(Borda)', 'MDT')}")
        L.append(f"4. 结构 vs 算力（自洽性基线）："
                 f"{line('Ax5_sc_vs_MDT', 1, 'A×5(SC)', 'MDT')}；"
                 f"{line('Ax5_sc_vs_MDT', 3, 'A×5(SC)', 'MDT')}；"
                 f"{line('Ax5_sc_vs_MDT', 5, 'A×5(SC)', 'MDT')}")
        L.append(f"5. 两变体差异：{line('Ax5_sc_vs_Ax5', 1, 'A×5(SC)', 'A×5(Borda)')}；"
                 f"{line('Ax5_sc_vs_Ax5', 5, 'A×5(SC)', 'A×5(Borda)')}")

        def mdt_dir(c):
            """返回 1=MDT 更好、-1=A×5 更好、0=无显著差异。"""
            if c is None or c["mean_rate_a"] is None or c["wilcoxon_p"] is None:
                return 0
            if c["wilcoxon_p"] >= 0.05:
                return 0
            return 1 if c["mean_diff"] < 0 else -1

        cmp5 = full["comparisons"].get("Ax5_vs_MDT", {})
        c3, c5 = cmp5.get("top3"), cmp5.get("top5")
        d3, d5 = mdt_dir(c3), mdt_dir(c5)
        if d3 == 1 or d5 == 1:
            verdict = ("MDT 相对等预算 A×5 在 top-3/top-5 上（部分）显著更优 → "
                       "支持“结构而非算力”的归因。")
        elif d3 == -1 or d5 == -1:
            verdict = ("等预算 A×5 反而显著更优 → “结构带来收益”的归因不成立，"
                       "需据实改写。")
        else:
            verdict = ("MDT 相对等预算 A×5（Borda 与自洽性两版）在 top-3/5 上"
                       "均无显著差异 → 核心归因需按此口径弱化为条件性表述。")
        L.append(f"\n**判定**：{verdict}")
    return "\n".join(L) + "\n"


def run_analyze():
    arms = build_arms()
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    splits = cs.split_ids()
    if LIMIT:
        keep = {name: [c for c in ids if c in
                       {r["case_id"] for r in arms["Ax5"][SEEDS[0]].values()}]
                for name, ids in splits.items()}
        splits = keep
    stats = cs.split_stats(arms, splits, PAIRS)
    meta = {"n_cases": len(arms["Ax5"][SEEDS[0]]), "n_seeds": len(SEEDS),
            "seeds": SEEDS, "n_infer_calls": len(arms["Ax5"][SEEDS[0]])
            * len(SEEDS) * A5_SAMPLES,
            "outdir": str(OUTDIR),
            "judge_cache": str(cs.GLM_CACHE),
            "judge_model": cs.GLM_MODEL, "prompt": "topn_cpc_promptv2.A_TOPN_PROMPT",
            "aggregation": "topn_cpc.aggregate_top5 (Borda, same_disease 聚类)",
            "sc_variant": "rank-1 疾病聚类多数投票 → top-1，其余 Borda 补齐"}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats, ["full87", "heldout46", "dev41"])
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def main():
    phase = os.environ.get("PHASE", "all").lower()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"数据集: {len(cases)} 例 | seeds {SEEDS} | 输出: {OUTDIR} | "
          f"GLM 缓存: {cs.GLM_CACHE}", flush=True)

    if phase in ("all", "infer"):
        t0 = time.time()
        for seed in SEEDS:
            run_seed(seed, cases)
        for seed in SEEDS:
            rows = load_arm(seed)
            assert len(rows) == len(cases), f"s{seed} 缺行: {len(rows)}"
            bad = [r["case_id"] for r in rows.values()
                   if len(r["samples"]) < A5_SAMPLES or not r["top5"]]
            assert not bad, f"s{seed} 样本/聚合不完整: {bad[:3]}"
        print(f"[检查] {len(SEEDS)} seeds 全部完整、无空聚合（{time.time() - t0:.0f}s）",
              flush=True)

    if phase in ("all", "judge"):
        run_judge()

    if phase in ("all", "analyze"):
        run_analyze()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
