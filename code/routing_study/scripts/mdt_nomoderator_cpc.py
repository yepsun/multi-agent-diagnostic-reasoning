#!/usr/bin/env python3
"""主持人消融：MDT 的 5 个隔离角色列表直接 Borda 聚合（无 moderator 主持人调用）。

回应评审 Major 1：MDT = 5 个隔离角色 + LLM 主持人。现有对照（A×5）只排除了
"调用预算"，未能区分"独立视角的列表多样性 + 机械聚合即可"与"主持人综合
本身有增量"。本臂复用主实验 MDT 已产出的角色级输出（topn_mdt/roles.jsonl，
seed 1；topn_mdt/s{2..5}/roles.jsonl），对每例每 seed 的 5 份角色列表做与
A×5 完全相同的 Borda 聚合（topn_cpc.aggregate_top5，rank-1 权重 1.0→rank-5 0.2，
same_disease 同病簇合并），得到 MDT-Borda 臂——零次新增策略调用。

阶段（PHASE=aggregate|judge|analyze|all，默认 all）：
  aggregate → results/topn_mdt_nomoderator/MdtBorda_s{1..5}.jsonl
              （每行 {"case_id","gold","top5"}，另附 roles_sha 等元信息于 meta）
  judge     → 只补 GLM v3 共享缓存缺失的 (gold, cand) 对
              （caselevel_stats.judge_missing，判官/规则与主实验冻结版一致）
  analyze   → results/mdt_nomoderator.{json,md}：5-seed 均值±SD top-1/3/5，
              与 MDT（主持人版）、A×1 的病例级配对检验（Wilcoxon + cluster
              bootstrap 10000 + 多数决精确 McNemar），口径同 stats_caselevel.py；
              另给 held-out 46 / dev 41 子集，并报告主持人改写率
              （MDT 最终列表与 MDT-Borda 列表不同的比例）。

环境变量：GLM_CACHE（默认共享 judge_cache_glm_v3.json，正式运行勿改）。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import aggregate_top5  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = RESULTS / "topn_mdt_nomoderator"
SEEDS = [1, 2, 3, 4, 5]
OUT_JSON = RESULTS / "mdt_nomoderator.json"
OUT_MD = RESULTS / "mdt_nomoderator.md"
PAIRS = [("MdtBorda", "MDT"), ("MdtBorda", "Ax1"), ("MDT", "Ax1")]
ARM_LABEL = {"MdtBorda": "MDT-Borda（角色列表直接聚合，无主持人）",
             "MDT": "MDT（5 角色 + 主持人）", "Ax1": "A×1 (T=0)"}


def roles_path(seed):
    return (RESULTS / "topn_mdt" / "roles.jsonl" if seed == 1
            else RESULTS / "topn_mdt" / f"s{seed}" / "roles.jsonl")


def synth_path(seed):
    return (RESULTS / "topn_mdt" / "synthesis.jsonl" if seed == 1
            else RESULTS / "topn_mdt" / f"s{seed}" / "synthesis.jsonl")


def load_roles_by_case(seed):
    by_case = {}
    for line in open(roles_path(seed)):
        r = json.loads(line)
        by_case.setdefault(r["case_id"], []).append(r)
    return by_case


def run_aggregate(cases):
    gold = {c["case_id"]: c["gold"] for c in cases}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = OUTDIR / f"MdtBorda_s{seed}.jsonl"
        done = cs.load(path)
        by_case = load_roles_by_case(seed)
        todo = [cid for cid in gold
                if cid not in done and cid in by_case]
        missing_cases = [cid for cid in gold if cid not in by_case]
        if missing_cases:
            raise RuntimeError(f"s{seed} 角色文件缺病例: {missing_cases}")
        n_new = 0
        for cid in todo:
            role_lists = [[c["diagnosis"] for c in r["top5"]]
                          for r in by_case[cid]]
            if len(role_lists) != 5:
                raise RuntimeError(f"s{seed} {cid} 角色数 {len(role_lists)} != 5")
            row = {"case_id": cid, "gold": gold[cid],
                   "top5": aggregate_top5(role_lists)}
            with open(path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1
        print(f"[aggregate s{seed}] 已有 {len(done)}，新写 {n_new}，"
              f"共 {len(done) + n_new}", flush=True)


def run_judge():
    rows = []
    for seed in SEEDS:
        rows.extend(cs.load(OUTDIR / f"MdtBorda_s{seed}.jsonl").values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text()))
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}", flush=True)
    return cache


def build_arms():
    mdtb = {s: cs.load(OUTDIR / f"MdtBorda_s{s}.jsonl") for s in SEEDS}
    refs = {
        "Ax1": {s: cs.load(RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl")
                for s in SEEDS},
        "MDT": {s: cs.load(synth_path(s)) for s in SEEDS},
    }
    arms = {"MdtBorda": mdtb, **refs}
    return arms


def moderator_rewrite_stats(arms):
    """主持人相对机械聚合改写列表的程度（列表完全一致率、top-1 一致率）。"""
    out = {}
    for seed in SEEDS:
        same_list = same_top1 = n = 0
        for cid, row_b in arms["MdtBorda"][seed].items():
            row_m = arms["MDT"][seed].get(cid)
            if row_m is None:
                continue
            n += 1
            lb = [d.lower() for d in row_b["top5"][:5]]
            lm = [d.lower() for d in row_m["top5"][:5]]
            same_list += lb == lm
            same_top1 += lb[:1] == lm[:1]
        out[f"s{seed}"] = {"n": n, "identical_top5": same_list,
                           "identical_top1": same_top1}
    return out


def run_analyze():
    arms = build_arms()
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    splits = cs.split_ids()
    stats = cs.split_stats(arms, splits, PAIRS)
    missing = {}
    for name, res in stats.items():
        for arm, a in res["arms"].items():
            m = [0]
            for seed in SEEDS:
                for row in arms[arm][seed].values():
                    cs.hit_flags(row, cache, missing=m)
            missing[arm] = m[0]
    meta = {"n_cases": len(arms["MdtBorda"][SEEDS[0]]), "n_seeds": len(SEEDS),
            "seeds": SEEDS, "n_new_strategy_calls": 0,
            "outdir": str(OUTDIR), "judge_cache": str(cs.GLM_CACHE),
            "judge_model": cs.GLM_MODEL,
            "roles_source": "results/topn_mdt/roles.jsonl (s1), s{2..5}/roles.jsonl",
            "aggregation": "topn_cpc.aggregate_top5 (Borda, same_disease 聚类，与 A×5 同一函数)",
            "unadjudicated_pairs": missing,
            "moderator_rewrite": moderator_rewrite_stats(arms)}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def render_md(meta, stats):
    L = []
    L.append("# 主持人消融：MDT-Borda（角色列表直接 Borda 聚合，无 moderator）\n")
    L.append("- 复用主实验 MDT 的角色级输出（零次新增策略调用），对每例每 seed "
             "的 5 份角色列表做与 A×5 完全相同的 Borda 聚合"
             "（`topn_cpc.aggregate_top5`）；与主持人版 MDT、A×1 在同一病例级"
             "框架下比较（配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10000 "
             "次 95% CI + 多数决精确 McNemar；GLM-5.3-flash × v3 冻结判官，"
             "共享缓存，只补缺失对）。")
    L.append(f"- 未判定对（按 miss 计入失败，不计入命中）：{meta['unadjudicated_pairs']}")
    rw = {k: f"top-1 一致 {v['identical_top1']}/{v['n']}，"
             f"列表全同 {v['identical_top5']}/{v['n']}"
          for k, v in meta["moderator_rewrite"].items()}
    L.append(f"- 主持人改写率（最终列表与机械聚合不同）：{rw}\n")
    for name, res in stats.items():
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for arm in ("MdtBorda", "MDT", "Ax1"):
            a = res["arms"][arm]
            L.append(f"| {ARM_LABEL[arm]} | {cs.acc_cell(a['mean_sd'], 1)} | "
                     f"{cs.acc_cell(a['mean_sd'], 3)} | "
                     f"{cs.acc_cell(a['mean_sd'], 5)} |")
        L.append("\n| 对比 | top-k | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
                 "多数决 McNemar (a:b) p |")
        L.append("|---|---|---|---|---|---|")
        for pair, by_k in res["comparisons"].items():
            a, b = pair.split("_vs_")
            for k in (1, 3, 5):
                c = by_k[f"top{k}"]
                if c["mean_rate_a"] is None:
                    continue
                lo, hi = c["boot95_ci"]
                maj = c["majority"]
                L.append(
                    f"| {ARM_LABEL[a]} vs {ARM_LABEL[b]} | top-{k} | "
                    f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                    f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                    f"{maj['a_only']}:{maj['b_only']} p={cs.fmt_p(maj['mcnemar_p'])}"
                    f"{cs.sig(maj['mcnemar_p'])} |")
    return "\n".join(L) + "\n"


def main():
    phase = os.environ.get("PHASE", "all").lower()
    cases = load_merged()
    print(f"数据集: {len(cases)} 例 | 输出: {OUTDIR} | GLM 缓存: {cs.GLM_CACHE}",
          flush=True)
    t0 = time.time()
    if phase in ("all", "aggregate"):
        run_aggregate(cases)
    if phase in ("all", "judge"):
        run_judge()
    if phase in ("all", "analyze"):
        run_analyze()
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
