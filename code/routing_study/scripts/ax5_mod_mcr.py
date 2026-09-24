#!/usr/bin/env python3
"""Ax5Mod 在 MCR（外部 406 例）上的析因复制：同提示 5 采样 + 主持人。

Paper2 的关键外部复制：CPC 上"独立生成 + 主持人综合"充分、专业角色冗余；
本脚本把同一配方搬到 MedCaseReasoning，检验外部语料复现。
阶段（PHASE=sample|mod|judge|analyze|all）：
  sample  → results/topn_ax5_mod_mcr/McrAx5_s{1..5}.jsonl（T=0.7，5 份采样
            + Borda 聚合 top5；406×5×5 = 10,150 次调用）
  mod     → results/topn_ax5_mod_mcr/McrAx5Mod_s{1..5}.jsonl（2,030 次）
  judge   → 冻结 GLM v3 缓存只补缺失对
  analyze → results/ax5_mod_mcr.{json,md}：与 A×1/P/MDT 的病例级配对检验
环境变量：AX5MOD_MCR_OUTDIR、AX5MOD_MCR_LIMIT、AX5MOD_MCR_SEEDS。
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
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import append_row, aggregate_top5, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
from topn_mcr import load_cases  # noqa: E402
from ax5_mod_cpc import moderator_prompt, opinions_from_samples, call_moderator  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("AX5MOD_MCR_OUTDIR", RESULTS / "topn_ax5_mod_mcr"))
SEEDS = [int(s) for s in os.environ.get("AX5MOD_MCR_SEEDS", "1,2,3,4,5").split(",") if s.strip()]
LIMIT = int(os.environ.get("AX5MOD_MCR_LIMIT", "0"))
TEMP_SAMPLE = 0.7
N_SAMPLES = 5
OUT_JSON = RESULTS / "ax5_mod_mcr.json"
OUT_MD = RESULTS / "ax5_mod_mcr.md"
MCR = RESULTS / "topn_mcr"
MCR_SEEDS = RESULTS / "topn_mcr_seeds"
PAIRS = [("McrAx5Mod", "Ax1"), ("McrAx5Mod", "MDT"), ("McrAx5Mod", "P"),
         ("McrAx5Mod", "McrAx5")]


def sample_path(seed):
    return OUTDIR / f"McrAx5_s{seed}.jsonl"


def mod_path(seed):
    return OUTDIR / f"McrAx5Mod_s{seed}.jsonl"


def run_sample(cases):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from topn_mcr import call_top5
    for seed in SEEDS:
        done = cs.load(sample_path(seed))
        todo = [c for c in cases if c["case_id"] not in done]
        print(f"[McrAx5 s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            samples, tokens_total = [], 0
            while len(samples) < N_SAMPLES:
                last_err = None
                top5 = None
                tokens = 0
                for attempt in range(4):
                    try:
                        top5, tokens = call_top5(
                            A_TOPN_PROMPT.format(case_text=c["text"]), TEMP_SAMPLE)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(3 * (attempt + 1))
                if last_err is not None:
                    raise last_err
                tokens_total += tokens
                if top5:
                    samples.append(top5)
            return {"case_id": c["case_id"], "gold": c["gold"],
                    "top5": aggregate_top5(samples), "samples": samples,
                    "total_tokens": tokens_total}

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    append_row(sample_path(seed), fut.result())
                except Exception as e:
                    print(f"[失败] McrAx5 s{seed} {c['case_id'][:30]}: {e}", flush=True)
                    continue
                if i % 50 == 0 or i == len(todo):
                    print(f"[McrAx5 s{seed}] {i}/{len(todo)} ({time.time()-t0:.0f}s)",
                          flush=True)
        rows = cs.load(sample_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"McrAx5 s{seed} 缺行 {len(missing)}"
        print(f"[McrAx5 s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_mod(cases):
    for seed in SEEDS:
        src = cs.load(sample_path(seed))
        done = cs.load(mod_path(seed))
        todo = [c for c in cases
                if c["case_id"] in src and c["case_id"] not in done]
        print(f"[McrAx5Mod s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            samples = src[c["case_id"]]["samples"]
            assert len(samples) == N_SAMPLES
            prompt = moderator_prompt(c["text"], opinions_from_samples(samples))
            for attempt in range(3):
                top5, tokens, raw = call_moderator(prompt)
                if len(top5) == 5:
                    return {"case_id": c["case_id"], "gold": c["gold"],
                            "top5": top5, "total_tokens": tokens}
            raise RuntimeError(f"{c['case_id']} 连续不足 5 项")

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    append_row(mod_path(seed), fut.result())
                except Exception as e:
                    print(f"[失败] McrAx5Mod s{seed} {c['case_id'][:30]}: {e}", flush=True)
                    continue
                if i % 50 == 0 or i == len(todo):
                    print(f"[McrAx5Mod s{seed}] {i}/{len(todo)} ({time.time()-t0:.0f}s)",
                          flush=True)
        rows = cs.load(mod_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"McrAx5Mod s{seed} 缺行 {len(missing)}"
        print(f"[McrAx5Mod s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_judge():
    rows = []
    for seed in SEEDS:
        rows.extend(cs.load(sample_path(seed)).values())
        rows.extend(cs.load(mod_path(seed)).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text()))
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)


def run_analyze(cases):
    ids = [c["case_id"] for c in cases]
    arms = {"McrAx5": {s: cs.load(sample_path(s)) for s in SEEDS},
            "McrAx5Mod": {s: cs.load(mod_path(s)) for s in SEEDS},
            "Ax1": {s: cs.load(MCR / "ax1.jsonl" if s == 1
                               else MCR_SEEDS / f"Ax1_s{s}.jsonl") for s in SEEDS},
            "P": {s: cs.load(MCR / "p.jsonl" if s == 1
                             else MCR_SEEDS / f"P_s{s}.jsonl") for s in SEEDS},
            "MDT": {s: cs.load(MCR / "mdt_synth.jsonl" if s == 1
                               else MCR_SEEDS / f"s{s}/mdt_synth.jsonl") for s in SEEDS}}
    for arm, by_seed in arms.items():
        for s in SEEDS:
            assert all(i in by_seed[s] for i in ids), f"{arm} s{s} 缺例"
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    stats = cs.split_stats(arms, {"full": ids}, PAIRS)
    meta = {"n_cases": len(ids), "n_seeds": len(SEEDS),
            "sample_calls": len(ids) * len(SEEDS) * N_SAMPLES,
            "mod_calls": len(ids) * len(SEEDS),
            "judge_cache": str(cs.GLM_CACHE)}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    res = stats["full"]
    L = ["# Ax5Mod 在 MCR 上的析因复制（同提示 5 采样 + 主持人）\n"]
    L.append("| 方案 | top-1 | top-3 | top-5 |")
    L.append("|---|---|---|---|")
    label = {"McrAx5Mod": "A×5+Mod", "McrAx5": "A×5 Borda", "Ax1": "A×1",
             "P": "P", "MDT": "MDT"}
    for arm in ("McrAx5Mod", "McrAx5", "Ax1", "P", "MDT"):
        a = res["arms"][arm]
        L.append(f"| {label[arm]} | {cs.acc_cell(a['mean_sd'],1)} | "
                 f"{cs.acc_cell(a['mean_sd'],3)} | {cs.acc_cell(a['mean_sd'],5)} |")
    L.append("\n| 对比 | top-k | 均值差 [95% CI] | Wilcoxon p |")
    L.append("|---|---|---|---|")
    for pair, by_k in res["comparisons"].items():
        a, b = pair.split("_vs_")
        for k in (1, 3, 5):
            c = by_k[f"top{k}"]
            if c["mean_rate_a"] is None:
                continue
            lo, hi = c["boot95_ci"]
            L.append(f"| {label[a]} vs {label[b]} | top-{k} | "
                     f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                     f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def main():
    phase = os.environ.get("PHASE", "all").lower()
    cases = load_cases()
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"MCR 数据集: {len(cases)} 例 | 输出: {OUTDIR}", flush=True)
    t0 = time.time()
    if phase in ("all", "sample"):
        run_sample(cases)
    if phase in ("all", "mod"):
        run_mod(cases)
    if phase in ("all", "judge"):
        run_judge()
    if phase in ("all", "analyze"):
        run_analyze(cases)
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
