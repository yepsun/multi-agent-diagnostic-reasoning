#!/usr/bin/env python3
"""87 例 × 5 seeds：A×1 与 P 的重复运行 + 配对统计（McNemar 精确检验）。

seed = 独立重复运行（云端 API 无种子参数，重复度量 run-to-run 方差）。
A×1 temp=0（近确定，重复验证确定性）；P temp=0.3（随机采样方差）。
判定走共享 judge_cache。产出 topn_seeds/seed_summary.json。
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from topn_cpc import load_done, append_row, call_top5, run_judge_phase, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_seeds"
SEEDS = 5


def run_one(scheme, seed, cases):
    path = OUTDIR / f"{scheme}_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{scheme} s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        if scheme == "Ax1":
            top5, tokens = call_top5(
                A_TOPN_PROMPT.format(case_text=c["text"]), temperature=0.0)
        else:
            prompt = PERSPECTIVE_PROMPT.format(
                structured_case=c["text"]) + P_TOPN_SUFFIX
            top5, tokens = call_top5(prompt, temperature=0.3)
        return {"case_id": c["case_id"], "gold": c["gold"],
                "top5": top5, "total_tokens": tokens}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 20 == 0 or i == len(todo):
                print(f"[{scheme} s{seed}] {i}/{len(todo)}", flush=True)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    print(f"数据集: {len(cases)} 例 × {SEEDS} seeds × 2 方案", flush=True)

    for seed in range(1, SEEDS + 1):
        for scheme in ("Ax1", "P"):
            run_one(scheme, seed, cases)

    scheme_rows = {}
    for scheme in ("Ax1", "P"):
        rows = []
        for seed in range(1, SEEDS + 1):
            rows.extend(load_done(OUTDIR / f"{scheme}_s{seed}.jsonl").values())
        scheme_rows[scheme] = rows
    hits_fn = run_judge_phase(cases, scheme_rows)

    def metrics(rows):
        n = len(rows)
        t1 = t3 = t5 = 0
        hits_by_case = {}
        for row in rows:
            verdicts = hits_fn(row)
            hit = next((r for r, v in enumerate(verdicts, start=1) if v), None)
            hits_by_case[(row["case_id"], id(row))] = hit
            if hit:
                t1 += hit == 1
                t3 += hit <= 3
                t5 += 1
        return {"n": n, "top1": t1, "top3": t3, "top5": t5}, hits_by_case

    per_seed = {}
    for scheme in ("Ax1", "P"):
        for seed in range(1, SEEDS + 1):
            rows = list(load_done(OUTDIR / f"{scheme}_s{seed}.jsonl").values())
            m, _ = metrics(rows)
            per_seed[(scheme, seed)] = m
            print(f"{scheme} s{seed}: n={m['n']} top1 {m['top1']} "
                  f"({m['top1']/m['n']:.1%}) top3 {m['top3']} "
                  f"({m['top3']/m['n']:.1%}) top5 {m['top5']} "
                  f"({m['top5']/m['n']:.1%})", flush=True)

    print("\n===== 均值 ± SD =====", flush=True)
    import statistics as st
    stats_summary = {}
    for scheme in ("Ax1", "P"):
        accs = {k: [per_seed[(scheme, s)][f"top{k}"] / per_seed[(scheme, s)]["n"]
                    for s in range(1, SEEDS + 1)] for k in (1, 3, 5)}
        stats_summary[scheme] = {
            f"top{k}": {"mean": round(st.mean(accs[k]), 4),
                        "sd": round(st.stdev(accs[k]), 4) if SEEDS > 1 else 0.0}
            for k in (1, 3, 5)}
        print(scheme, stats_summary[scheme], flush=True)

    print("\n===== 配对 McNemar（top-1，逐 seed + 合并）=====", flush=True)
    pooled = {"both": 0, "a_only": 0, "p_only": 0, "neither": 0}
    per_seed_mcn = []
    for seed in range(1, SEEDS + 1):
        a = load_done(OUTDIR / f"Ax1_s{seed}.jsonl")
        p = load_done(OUTDIR / f"P_s{seed}.jsonl")
        b = cc = both = neither = 0
        for cid, ra in a.items():
            va = hits_fn(ra)
            vp = hits_fn(p[cid])
            ha = any(va[:1])
            hp = any(vp[:1])
            both += ha and hp
            b += ha and not hp
            cc += (not ha) and hp
            neither += (not ha) and (not hp)
        pv = mcnemar_exact(b, cc)
        per_seed_mcn.append({"seed": seed, "a_right_p_wrong": b,
                             "p_right_a_wrong": cc, "p": round(pv, 4)})
        pooled["both"] += both
        pooled["a_only"] += b
        pooled["p_only"] += cc
        pooled["neither"] += neither
        print(f"s{seed}: A独对 {b} | P独对 {cc} | McNemar p={pv:.4f}", flush=True)
    print(f"合并({SEEDS}×87 对): A独对 {pooled['a_only']} | P独对 "
          f"{pooled['p_only']} | McNemar p={mcnemar_exact(pooled['a_only'], pooled['p_only']):.4f}",
          flush=True)

    (OUTDIR / "seed_summary.json").write_text(json.dumps(
        {"per_seed": {f"{s}_s{seed}": per_seed[(s, seed)]
                      for s in ("Ax1", "P") for seed in range(1, SEEDS + 1)},
         "stats": stats_summary,
         "mcnemar_per_seed": per_seed_mcn,
         "mcnemar_pooled": pooled}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已写入 {OUTDIR / 'seed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
