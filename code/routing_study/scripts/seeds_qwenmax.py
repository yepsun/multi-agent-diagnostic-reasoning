#!/usr/bin/env python3
"""Qwen3.8-Max 补 4 个 seed（s2-s5），与既有单次运行（s1）合计 5 seeds，
对齐 qwen3.8-flash 的 seeds_87 协议；输出均值±SD 与配对 McNemar，
并与 flash 的 seed 统计并排。"""
import json
import os
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["QWEN_MODEL"] = "qwen3.8-max"  # must precede imports

from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from topn_cpc_qwenmax import call_top5_qm, judge_rows  # noqa: E402
from seeds_87 import mcnemar_exact  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_seeds_qwenmax"
S1_DIR = ROOT / "routing_study" / "results" / "topn_qwenmax"
JUDGE_CACHE = OUTDIR / "judge_cache_dsflash.json"
FLASH_SUMMARY = ROOT / "routing_study" / "results" / "topn_seeds" / "seed_summary.json"
SEEDS = range(1, 6)  # s1 = 既有运行（复制进来），s2-s5 = 新跑


def run_one(scheme, seed, cases):
    path = OUTDIR / f"{scheme}_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{scheme} s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        if scheme == "Ax1":
            top5, tokens = call_top5_qm(
                A_TOPN_PROMPT.format(case_text=c["text"]), temperature=0.0)
        else:
            prompt = PERSPECTIVE_PROMPT.format(
                structured_case=c["text"]) + P_TOPN_SUFFIX
            top5, tokens = call_top5_qm(prompt, temperature=0.3)
        return {"case_id": c["case_id"], "gold": c["gold"],
                "top5": top5, "total_tokens": tokens}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 20 == 0 or i == len(todo):
                print(f"[{scheme} s{seed}] {i}/{len(todo)}", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # s1 = 既有单次运行
    for scheme in ("Ax1", "P"):
        src = S1_DIR / f"{scheme}.jsonl"
        dst = OUTDIR / f"{scheme}_s1.jsonl"
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text())
    cases = load_merged()

    for seed in range(2, 6):
        for scheme in ("Ax1", "P"):
            run_one(scheme, seed, cases)

    rows_by = {}
    for scheme in ("Ax1", "P"):
        all_rows = []
        for seed in SEEDS:
            all_rows.extend(load_done(OUTDIR / f"{scheme}_s{seed}.jsonl").values())
        rows_by[scheme] = all_rows
    hits_fn = judge_rows(rows_by["Ax1"] + rows_by["P"], JUDGE_CACHE)

    print("\n===== Qwen3.8-Max 逐 seed =====", flush=True)
    per_seed = {}
    for scheme in ("Ax1", "P"):
        for seed in SEEDS:
            rows = list(load_done(OUTDIR / f"{scheme}_s{seed}.jsonl").values())
            n = len(rows)
            t1 = t3 = t5 = 0
            for row in rows:
                verdicts = hits_fn(row)
                hit = next((r for r, v in enumerate(verdicts, start=1) if v), None)
                if hit:
                    t1 += hit == 1
                    t3 += hit <= 3
                    t5 += 1
            per_seed[(scheme, seed)] = {"n": n, "top1": t1, "top3": t3, "top5": t5}
            print(f"{scheme} s{seed}: top1 {t1}/{n} ({t1/n:.1%}) "
                  f"top3 {t3}/{n} ({t3/n:.1%}) top5 {t5}/{n} ({t5/n:.1%})",
                  flush=True)

    print("\n===== Qwen3.8-Max 均值 ± SD =====", flush=True)
    max_stats = {}
    for scheme in ("Ax1", "P"):
        accs = {k: [per_seed[(scheme, s)][f"top{k}"] / per_seed[(scheme, s)]["n"]
                    for s in SEEDS] for k in (1, 3, 5)}
        max_stats[scheme] = {
            f"top{k}": {"mean": round(st.mean(accs[k]), 4),
                        "sd": round(st.stdev(accs[k]), 4)} for k in (1, 3, 5)}
        print(scheme, max_stats[scheme], flush=True)

    print("\n===== Max 配对 McNemar（top-1）=====", flush=True)
    per_seed_mcn = []
    pooled = {"a_only": 0, "p_only": 0}
    for seed in SEEDS:
        a = load_done(OUTDIR / f"Ax1_s{seed}.jsonl")
        p = load_done(OUTDIR / f"P_s{seed}.jsonl")
        b = cc = 0
        for cid, ra in a.items():
            ha = any(hits_fn(ra)[:1])
            hp = any(hits_fn(p[cid])[:1])
            b += ha and not hp
            cc += (not ha) and hp
        pv = mcnemar_exact(b, cc)
        per_seed_mcn.append({"seed": seed, "a_right_p_wrong": b,
                             "p_right_a_wrong": cc, "p": round(pv, 4)})
        pooled["a_only"] += b
        pooled["p_only"] += cc
        print(f"s{seed}: A独对 {b} | P独对 {cc} | p={pv:.4f}", flush=True)
    pv_pool = mcnemar_exact(pooled["a_only"], pooled["p_only"])
    print(f"合并({len(SEEDS)}×87): A独对 {pooled['a_only']} | P独对 "
          f"{pooled['p_only']} | p={pv_pool:.4f}", flush=True)

    flash = json.loads(FLASH_SUMMARY.read_text()) if FLASH_SUMMARY.exists() else {}
    print("\n===== Max vs Flash（seed 统计并排）=====", flush=True)
    for scheme in ("Ax1", "P"):
        fs = (flash.get("stats", {}).get(scheme, {}))
        print(f"{scheme}: Max {max_stats[scheme]}")
        print(f"{'':{len(scheme)+2}}Flash {fs}")

    (OUTDIR / "seed_summary.json").write_text(json.dumps(
        {"per_seed": {f"{s}_s{seed}": per_seed[(s, seed)]
                      for s in ("Ax1", "P") for seed in SEEDS},
         "stats": max_stats,
         "mcnemar_per_seed": per_seed_mcn,
         "mcnemar_pooled": pooled}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已写入 {OUTDIR / 'seed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
