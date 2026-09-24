import os as _os
_os.environ.setdefault("OPENROUTER_ER", "1")  # ER 合规路由：OpenRouter + zdr + data_collection deny
#!/usr/bin/env python3
"""第二家族（deepseek-flash）的 ER Ax5Mod 复制：A×1_ds / 采样 / 主持人。

回应模型策略修正后的第一优先实验：ER 反转与"采样+主持人"配方是否在
第二模型族上复现。开放权重满足 ER 数据的本地/开源约束。

臂：
  Ax1_ds     单次直调（deepseek-flash，T=0，max_tokens 2048）——族内基线
  ErAx5_ds   同提示 5 采样（T=0.7）+ Borda 聚合（机械聚合格）
  ErAx5Mod_ds 同样 5 采样 + LLM 主持人（deepseek-flash，T=0.3，max_tokens 4096）
推理均在 deepseek-flash 端点（与 qwen 主流水线不同源，可并行，不抢其限流配额）。
判定走 GLM v3 冻结共享缓存——注意与主管线判分**串行**执行（本脚本默认
只做推理；judge 阶段需显式指定且避开主管线判分窗口，或用分片缓存）。

阶段（PHASE=sample|ax1|mod|judge|analyze|all，默认 all）：
  sample → results/topn_ax5_mod_er_ds/ErAx5_ds_s{1..5}.jsonl（9,100 次）
  ax1    → results/topn_ax5_mod_er_ds/Ax1_ds_s{1..5}.jsonl（1,820 次）
  mod    → results/topn_ax5_mod_er_ds/ErAx5Mod_ds_s{1..5}.jsonl（1,820 次）
  judge  → 共享缓存补缺失对（默认跳过，DS_ER_JUDGE=1 才执行）
  analyze→ results/ax5_mod_er_ds.{json,md}（族内配对：ErAx5Mod_ds vs Ax1_ds 等）
环境变量：DSER_OUTDIR、DSER_LIMIT、DSER_SEEDS。
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

import caselevel_stats as cs  # noqa: E402
from run_inference import call_llm  # noqa: E402
from topn_cpc import load_done, append_row, parse_top5, aggregate_top5  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
from topn_erreason import load_cases  # noqa: E402
from ax5_mod_cpc import moderator_prompt, opinions_from_samples  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("DSER_OUTDIR", RESULTS / "topn_ax5_mod_er_ds"))
SEEDS = [int(s) for s in os.environ.get("DSER_SEEDS", "1,2,3,4,5").split(",") if s.strip()]
LIMIT = int(os.environ.get("DSER_LIMIT", "0"))
PROVIDER = "deepseek-flash"
TIMEOUT = 300
MAX_ATTEMPTS = 3
OUT_JSON = RESULTS / "ax5_mod_er_ds.json"
OUT_MD = RESULTS / "ax5_mod_er_ds.md"
ER = RESULTS / "topn_erreason"


def path_of(arm, seed):
    f = {"Ax1_ds": "Ax1_ds", "ErAx5_ds": "ErAx5_ds",
         "ErAx5Mod_ds": "ErAx5Mod_ds"}[arm]
    return OUTDIR / f"{f}_s{seed}.jsonl"


def call_ds_top5(prompt, temperature, max_tokens=2048):
    for attempt in range(MAX_ATTEMPTS):
        raw, usage = call_llm(prompt, temperature=temperature,
                              max_tokens=max_tokens, timeout=TIMEOUT,
                              provider=PROVIDER, disable_thinking=True)
        top5 = parse_top5(raw)
        seen, out = set(), []
        for dx in top5[:5]:
            k = dx.lower()
            if k not in seen:
                seen.add(k)
                out.append(dx)
        if len(out) == 5:
            return out, (usage or {}).get("total_tokens", 0)
        print(f"  [解析 {len(out)} 项] 重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
    raise RuntimeError("连续不足 5 项")


def run_arm(arm, cases, work, only_seed=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for seed in ([only_seed] if only_seed else SEEDS):
        path = path_of(arm, seed)
        done = load_done(path)
        todo = [c for c in cases if c["case_id"] not in done]
        print(f"[{arm} s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def wrap(c):
            row = work(c)
            row["case_id"], row["gold"] = c["case_id"], c["gold"]
            return row

        t0 = time.time()
        with ThreadPoolExecutor(6) as ex:
            futs = {ex.submit(wrap, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    append_row(path, fut.result())
                except Exception as e:
                    print(f"[失败] {arm} s{seed} {c['case_id'][:30]}: {e}", flush=True)
                    continue
                if i % 25 == 0 or i == len(todo):
                    print(f"[{arm} s{seed}] {i}/{len(todo)} ({time.time()-t0:.0f}s)",
                          flush=True)
        rows = load_done(path)
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"{arm} s{seed} 缺行 {len(missing)}: {missing[:3]}"
        print(f"[{arm} s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def main():
    phase = os.environ.get("PHASE", "infer").lower()
    cases = load_cases()
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"ER {len(cases)} 例 | seeds {SEEDS} | provider {PROVIDER} | 输出 {OUTDIR}",
          flush=True)
    t0 = time.time()

    if phase in ("all", "ax1"):
        def work_ax1(c):
            top5, tokens = call_ds_top5(A_TOPN_PROMPT.format(case_text=c["text"]), 0.0)
            return {"top5": top5, "total_tokens": tokens}
        run_arm("Ax1_ds", cases, work_ax1)

    if phase in ("all", "sample"):
        def work_sample(c):
            samples, tokens_total = [], 0
            while len(samples) < 5:
                top5, tokens = call_ds_top5(
                    A_TOPN_PROMPT.format(case_text=c["text"]), 0.7)
                samples.append(top5)
                tokens_total += tokens
            return {"top5": aggregate_top5(samples), "samples": samples,
                    "total_tokens": tokens_total}
        run_arm("ErAx5_ds", cases, work_sample)

    if phase in ("all", "mod"):
        for seed in SEEDS:
            src = load_done(path_of("ErAx5_ds", seed))

            def work_mod(c, src=src):
                samples = src[c["case_id"]]["samples"]
                prompt = moderator_prompt(c["text"], opinions_from_samples(samples))
                for attempt in range(MAX_ATTEMPTS):
                    raw, usage = call_llm(prompt, temperature=0.3,
                                          max_tokens=4096, timeout=TIMEOUT,
                                          provider=PROVIDER, disable_thinking=True)
                    top5 = parse_top5(raw)
                    seen, out = set(), []
                    for dx in top5[:5]:
                        k = dx.lower()
                        if k not in seen:
                            seen.add(k)
                            out.append(dx)
                    if len(out) == 5:
                        return {"top5": out,
                                "total_tokens": (usage or {}).get("total_tokens", 0)}
                    print(f"  [解析 {len(out)} 项] 重试 {attempt + 1}/{MAX_ATTEMPTS}",
                          flush=True)
                raise RuntimeError(f"{c['case_id']} 主持人连续不足 5 项")

            run_arm("ErAx5Mod_ds", cases, work_mod, only_seed=seed)

    if phase == "judge":
        if os.environ.get("DS_ER_JUDGE") != "1":
            print("[judge] 跳过（DS_ER_JUDGE=1 未设置；避免与主管线判分并发写共享缓存）",
                  flush=True)
            return
        rows = []
        for seed in SEEDS:
            for arm in ("Ax1_ds", "ErAx5_ds", "ErAx5Mod_ds"):
                rows.extend(load_done(path_of(arm, seed)).values())
        n_before = len(json.loads(cs.GLM_CACHE.read_text()))
        cache = cs.judge_missing(rows)
        print(f"[判定] 缓存 {n_before} → {len(cache)}", flush=True)

    if phase == "analyze":
        cache = json.loads(cs.GLM_CACHE.read_text())
        cs.set_cache(cache)
        arms = {arm: {s: cs.load(path_of(arm, s)) for s in SEEDS}
                for arm in ("Ax1_ds", "ErAx5_ds", "ErAx5Mod_ds")}
        # 参照：qwen 主臂（跨族对照单独标注）
        for arm, f in (("Ax1", "ax1"), ("P", "p"), ("MDT", "mdt_synth")):
            arms[arm] = {s: cs.load(ER / (f"{f}.jsonl" if s == 1
                                          else f"s{s}/{f}.jsonl")) for s in SEEDS}
        ids = [c["case_id"] for c in cases]
        PAIRS = [("ErAx5Mod_ds", "Ax1_ds"), ("ErAx5Mod_ds", "ErAx5_ds"),
                 ("ErAx5_ds", "Ax1_ds"), ("ErAx5Mod_ds", "Ax1"),
                 ("Ax1_ds", "Ax1")]
        stats = cs.split_stats(arms, {"full": ids}, PAIRS)
        meta = {"n_cases": len(ids), "n_seeds": len(SEEDS), "provider": PROVIDER,
                "model": os.environ["DEEPSEEK_MODEL"],
                "judge_cache": str(cs.GLM_CACHE)}
        OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        res = stats["full"]
        L = ["# 第二家族（deepseek-flash）ER 复制：A×1 / A×5 / Ax5Mod\n"]
        L.append("| 臂 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        lab = {"Ax1_ds": "A×1 (deepseek-flash)", "ErAx5_ds": "A×5 Borda (deepseek-flash)",
               "ErAx5Mod_ds": "A×5+Mod (deepseek-flash)", "Ax1": "A×1 (qwen, 跨族参照)",
               "P": "P (qwen)", "MDT": "MDT (qwen)"}
        for arm in ("Ax1_ds", "ErAx5_ds", "ErAx5Mod_ds", "Ax1", "P", "MDT"):
            a = res["arms"][arm]
            L.append(f"| {lab[arm]} | {cs.acc_cell(a['mean_sd'],1)} | "
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
                L.append(f"| {lab[a]} vs {lab[b]} | top-{k} | "
                         f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                         f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} |")
        OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L), flush=True)
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
