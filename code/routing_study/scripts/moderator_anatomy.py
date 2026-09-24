#!/usr/bin/env python3
"""主持人解剖三联实验（CPC，复用已存档 A×5 采样，只新增主持人调用）：

  A4 输入消融  lists-only：主持人不看病例文本（把"对照病例复核"条目改为
               "在列表内部权衡证据"），其余逐字一致 → 检验增益是否依赖
               "重读病例"。5 seeds × 87 = 435 次。
  A3 k 扫描    k∈{1,2,3,4}：取每例存档 5 份采样的前 k 份给主持人
               （提示词数量词如实改为 k），5 seeds × 4 × 87 = 1,740 次 →
               增益-膝点曲线（k=5 即既有 Ax5Mod）。
  A5 顺序置换  2 种置换（逆序 / 固定随机序 seed 20260920）× 87 × 1 seed
               = 174 次 → 呈现顺序稳健性。

判定沿用冻结共享缓存；分析为与 Ax5Mod（同 case-seed 输入）的配对比较。
输出：results/moderator_anatomy.{json,md}
环境变量：ANATOMY_PHASES（默认全部）、ANATOMY_LIMIT。
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from ax5_mod_cpc import call_moderator, opinions_from_samples  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
AX5DIR = RESULTS / "topn_seeds_ax5"
OUTDIR = RESULTS / "moderator_anatomy"
SEEDS = [1, 2, 3, 4, 5]
KS = [1, 2, 3, 4]
LIMIT = int(os.environ.get("ANATOMY_LIMIT", "0"))
PHASES = os.environ.get("ANATOMY_PHASES", "lists_only,k_sweep,order,analyze").split(",")
OUT_JSON = RESULTS / "moderator_anatomy.json"
OUT_MD = RESULTS / "moderator_anatomy.md"

MOD_HEAD_FULL = ("You are the moderator of an MDT panel. Five independent assessments of the "
                 "MGH CPC case below were produced without seeing one another. "
                 "Their ranked candidate lists are given.")
MOD_TAIL = """

Integrate them into the final ranked top-5 for the case:
- Candidates supported by multiple specialists generally rise.
- A unique candidate with specific, case-grounded support must NOT be dropped merely because only one specialist listed it.
- Resolve conflicts by re-checking against the case text.
- Combination diagnoses are allowed.

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, ... exactly 5 items]}}

Case:
{case_text}

Panel opinions:
{opinions_text}
"""


def prompt_variant(case_text, opinions_text, k, with_case_text, head):
    tail = MOD_TAIL
    if not with_case_text:
        tail = tail.replace(
            "- Resolve conflicts by re-checking against the case text.",
            "- Resolve conflicts by weighing the evidence across the lists.")
        tail = tail.replace("\n\nCase:\n{case_text}\n", "\n")
    tail = tail.replace("Candidates supported by multiple specialists generally rise.",
                        "Candidates supported by multiple assessments generally rise.")
    head = head.replace("Five independent assessments", f"{k} independent assessments")
    return (head + tail).format(case_text=case_text, opinions_text=opinions_text)


def moderated_arm(arm, cases, prompt_fn, workers=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{arm}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{arm}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        prompt = prompt_fn(c)
        for attempt in range(3):
            top5, tokens, raw = call_moderator(prompt)
            if len(top5) == 5:
                return {"case_id": c["case_id"], "gold": c["gold"],
                        "top5": top5, "total_tokens": tokens}
        raise RuntimeError(f"{c['case_id']} 连续不足 5 项")

    w = workers or MAX_WORKERS
    t0 = time.time()
    with ThreadPoolExecutor(w) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                append_row(path, fut.result())
            except Exception as e:
                print(f"[失败] {arm} {c['case_id'][:40]}: {e}", flush=True)
                continue
            if i % 25 == 0 or i == len(todo):
                print(f"[{arm}] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    rows = load_done(path)
    missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
    assert not missing, f"{arm} 缺行: {missing[:5]}"
    print(f"[{arm}] 完整 {len(rows)}/{len(cases)}", flush=True)
    return rows


def main():
    cases = load_merged()
    if LIMIT:
        cases = cases[:LIMIT]
    src = {s: load_done(AX5DIR / f"Ax5_s{s}.jsonl") for s in SEEDS}
    out_arms = {}
    rng = random.Random(20260920)

    if "lists_only" in PHASES:
        def fn(c):
            samples = src[1][c["case_id"]]["samples"]
            return prompt_variant(c["text"], opinions_from_samples(samples),
                                  5, with_case_text=False, head=MOD_HEAD_FULL)
        out_arms["lists_only_s1"] = moderated_arm("lists_only_s1", cases, fn)

    if "k_sweep" in PHASES:
        for k in KS:
            def fn(c, k=k):
                samples = src[1][c["case_id"]]["samples"][:k]
                return prompt_variant(c["text"], opinions_from_samples(samples),
                                      k, with_case_text=True, head=MOD_HEAD_FULL)
            out_arms[f"k{k}_s1"] = moderated_arm(f"k{k}_s1", cases, fn)

    if "order" in PHASES:
        perms = {"reverse": lambda xs: list(reversed(xs)),
                 "perm20260920": lambda xs: rng.sample(xs, len(xs))}
        for pname, pf in perms.items():
            def fn(c, pf=pf):
                samples = pf(src[1][c["case_id"]]["samples"])
                return prompt_variant(c["text"], opinions_from_samples(samples),
                                      5, with_case_text=True, head=MOD_HEAD_FULL)
            out_arms[f"order_{pname}_s1"] = moderated_arm(f"order_{pname}_s1", cases, fn)

    if "analyze" in PHASES:
        cache = json.loads((RESULTS / "judge_cache_glm_v3.json").read_text())
        cs.set_cache(cache)
        base = {s: load_done(RESULTS / "topn_ax5_mod" / f"Ax5Mod_s{s}.jsonl") for s in SEEDS}
        # 臂从磁盘发现：gen 与 analyze 可能分属两个进程（监督器即如此拆分）
        prefer = ["lists_only_s1", "k1_s1", "k2_s1", "k3_s1", "k4_s1",
                  "order_reverse_s1", "order_perm20260920_s1"]
        found = {p.stem for p in OUTDIR.glob("*_s1.jsonl")}
        arm_names = [a for a in prefer if a in found] + sorted(found - set(prefer))
        out_arms = {a: load_done(OUTDIR / f"{a}.jsonl") for a in arm_names}
        print(f"[analyze] 磁盘发现 {len(arm_names)} 臂: {arm_names}", flush=True)
        # 判分补缺：未判对按 miss 计会使命中率失真（必须先补到 0 再分析）
        all_rows = [r for rr in out_arms.values() for r in rr.values()]
        cache = cs.judge_missing(all_rows)
        cs.set_cache(cache)
        arms = {"Ax5Mod_s1_ref": {1: load_done(RESULTS / "topn_ax5_mod" / "Ax5Mod_s1.jsonl")},
                **{k: {1: v} for k, v in out_arms.items()}}
        # 全部在 seed 1 上与 Ax5Mod s1 配对
        PAIRS = [(a, "Ax5Mod_s1_ref") for a in out_arms]
        stats = cs.split_stats(arms, cs.split_ids(), PAIRS)
        meta = {"arms": list(out_arms), "seeds": "all arms on seed 1 (87 cases)",
                "prompt_disclosure": "数量词随 k 如实调整；lists_only 将'对照病例复核'条目替换为'在列表内部权衡证据'，其余逐字一致",
                "judge_cache": str(cs.GLM_CACHE)}
        OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        L = ["# 主持人解剖三联：输入消融 / k 扫描 / 顺序置换（CPC，seed 1，n=87）\n",
             "- 与 Ax5Mod seed-1（同 case-seed 输入）配对；判官/口径与主实验一致。\n"]
        res = stats["full87"]
        L.append("| 臂 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        name_map = {"Ax5Mod_s1_ref": "Ax5Mod 基线（完整输入，k=5）",
                    **{f"{a}_s1" if not a.endswith("_s1") else a: a for a in out_arms}}
        for arm in ["Ax5Mod_s1_ref"] + list(out_arms):
            if arm not in res["arms"]:
                continue
            a = res["arms"][arm]
            L.append(f"| {arm} | {cs.acc_cell(a['mean_sd'], 1)} | "
                     f"{cs.acc_cell(a['mean_sd'], 3)} | {cs.acc_cell(a['mean_sd'], 5)} |")
        L.append("\n| 对比 | top-k | 均值差 [95% CI] | Wilcoxon p |")
        L.append("|---|---|---|---|")
        for pair, by_k in res["comparisons"].items():
            a, b = pair.split("_vs_")
            for k in (1, 3, 5):
                c = by_k[f"top{k}"]
                if c["mean_rate_a"] is None:
                    continue
                lo, hi = c["boot95_ci"]
                L.append(f"| {a} vs {b} | top-{k} | "
                         f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                         f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} |")
        OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)
