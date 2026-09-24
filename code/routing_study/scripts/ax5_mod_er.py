import os as _os
_os.environ.setdefault("OPENROUTER_ER", "1")  # ER 合规路由：OpenRouter + zdr + data_collection deny
#!/usr/bin/env python3
"""Ax5Mod 在 ER-Reason 上的补测：同提示 5 采样 + LLM 主持人（第四格在急诊语料）。

背景：CPC 上的 2×2 析因（S10d/S10e）表明"独立生成 + 主持人综合"缺一不可、
专业角色冗余；但 ER 反转只在角色臂（P/MDT）上建立过。本脚本把 Ax5Mod 臂
搬到 ER，检验主持人 over 普通采样在急诊语料是否同样反转。

阶段（PHASE=sample|mod|judge|analyze|all，默认 all）：
  sample  → results/topn_ax5_mod_er/ErAx5_s{1..5}.jsonl
            364 例 × 5 seeds × 5 份采样（A_TOPN_PROMPT，T=0.7，qwen，
            与 CPC 的 A×5 设计一致），行内含 Borda 聚合 top5（aggregate_top5）
            供机械聚合臂计分；9,100 次调用。
  mod     → results/topn_ax5_mod_er/ErAx5Mod_s{1..5}.jsonl
            对每例每 seed 的 5 份采样发起 1 次主持人调用（提示词与
            ax5_mod_cpc.py 逐字一致：mdt_cpc moderator 仅首句适配）；
            qwen，T=0.3，max_tokens=4096，disable_thinking；1,820 次。
  judge   → 只补 GLM v3 共享缓存缺失对（ErAx5 与 ErAx5Mod 两臂的 top5）。
  analyze → results/ax5_mod_er.{json,md}：与 A×1/P/MDT 的病例级配对检验
            （Wilcoxon + cluster bootstrap + 多数决 McNemar；主判官冻结缓存），
            含症状级/疾病级标签分层的 5-seed 均值。

环境变量：AX5MOD_ER_OUTDIR、AX5MOD_ER_LIMIT（冒烟）、AX5MOD_ER_SEEDS、GLM_CACHE。
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import append_row, aggregate_top5, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
from topn_erreason import load_cases  # noqa: E402
from ax5_mod_cpc import moderator_prompt, opinions_from_samples, call_moderator  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("AX5MOD_ER_OUTDIR", RESULTS / "topn_ax5_mod_er"))
SEEDS = [int(s) for s in os.environ.get("AX5MOD_ER_SEEDS", "1,2,3,4,5").split(",") if s.strip()]
LIMIT = int(os.environ.get("AX5MOD_ER_LIMIT", "0"))
TEMP_SAMPLE = 0.7
N_SAMPLES = 5
MAX_ATTEMPTS = 3
OUT_JSON = RESULTS / "ax5_mod_er.json"
OUT_MD = RESULTS / "ax5_mod_er.md"
ER = RESULTS / "topn_erreason"
PAIRS = [("ErAx5Mod", "Ax1"), ("ErAx5Mod", "MDT"), ("ErAx5Mod", "P"),
         ("ErAx5Mod", "ErAx5"), ("MDT", "Ax1")]
ARM_LABEL = {"ErAx5Mod": "A×5+Mod（采样 + 主持人）",
             "ErAx5": "A×5 Borda（采样 + 机械聚合）",
             "Ax1": "A×1 (T=0)", "P": "P (T=0.3)", "MDT": "MDT（5 角色 + 主持人）"}


def sample_path(seed):
    return OUTDIR / f"ErAx5_s{seed}.jsonl"


def mod_path(seed):
    return OUTDIR / f"ErAx5Mod_s{seed}.jsonl"


def run_sample(cases):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from topn_mcr import call_top5
    for seed in SEEDS:
        done = cs.load(sample_path(seed))
        todo = [c for c in cases if c["case_id"] not in done]
        print(f"[ErAx5 s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            samples, tokens_total = [], 0
            while len(samples) < N_SAMPLES:
                top5, tokens = call_top5(A_TOPN_PROMPT.format(case_text=c["text"]),
                                         TEMP_SAMPLE)
                if top5:
                    samples.append(top5)
                    tokens_total += tokens
                else:
                    tokens_total += tokens
            return {"case_id": c["case_id"], "gold": c["gold"],
                    "top5": aggregate_top5(samples), "samples": samples,
                    "total_tokens": tokens_total}

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"[失败] ErAx5 s{seed} {c['case_id']}: {e}", flush=True)
                    continue
                append_row(sample_path(seed), row)
                if i % 25 == 0 or i == len(todo):
                    print(f"[ErAx5 s{seed}] {i}/{len(todo)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        rows = cs.load(sample_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"ErAx5 s{seed} 缺行: {missing}"
        print(f"[ErAx5 s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_mod(cases):
    for seed in SEEDS:
        src = cs.load(sample_path(seed))
        done = cs.load(mod_path(seed))
        todo = [c for c in cases
                if c["case_id"] in src and c["case_id"] not in done]
        print(f"[ErAx5Mod s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            samples = src[c["case_id"]].get("samples") or []
            assert len(samples) == N_SAMPLES, \
                f"{c['case_id']} 样本数 {len(samples)} != {N_SAMPLES}"
            prompt = moderator_prompt(c["text"], opinions_from_samples(samples))
            last = ""
            for attempt in range(MAX_ATTEMPTS):
                top5, tokens, raw = call_moderator(prompt)
                last = raw or ""
                if len(top5) == 5:
                    return {"case_id": c["case_id"], "gold": c["gold"],
                            "top5": top5, "total_tokens": tokens}
                print(f"[ErAx5Mod s{seed}] {c['case_id'][:30]} 解析 {len(top5)} 项，"
                      f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
            raise RuntimeError(f"{c['case_id']} 连续不足 5 项 | {last.strip()[:200]!r}")

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"[失败] ErAx5Mod s{seed} {c['case_id']}: {e}", flush=True)
                    continue
                append_row(mod_path(seed), row)
                if i % 25 == 0 or i == len(todo):
                    print(f"[ErAx5Mod s{seed}] {i}/{len(todo)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        rows = cs.load(mod_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"ErAx5Mod s{seed} 缺行: {missing}"
        print(f"[ErAx5Mod s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_judge():
    rows = []
    for seed in SEEDS:
        rows.extend(cs.load(sample_path(seed)).values())
        rows.extend(cs.load(mod_path(seed)).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text()))
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)


def build_arms(cases):
    ids = [c["case_id"] for c in cases]
    arms = {"ErAx5": {s: cs.load(sample_path(s)) for s in SEEDS},
            "ErAx5Mod": {s: cs.load(mod_path(s)) for s in SEEDS},
            "Ax1": {s: cs.load(ER / ("ax1.jsonl" if s == 1
                                     else f"s{s}/ax1.jsonl")) for s in SEEDS},
            "P": {s: cs.load(ER / ("p.jsonl" if s == 1
                                   else f"s{s}/p.jsonl")) for s in SEEDS},
            "MDT": {s: cs.load(ER / ("mdt_synth.jsonl" if s == 1
                                     else f"s{s}/mdt_synth.jsonl")) for s in SEEDS}}
    for arm, by_seed in arms.items():
        for s in SEEDS:
            missing = [i for i in ids if i not in by_seed[s]]
            assert not missing, f"{arm} s{s} 缺 {len(missing)} 例"
    return arms, ids


def run_analyze(cases):
    arms, ids = build_arms(cases)
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    stats = cs.split_stats(arms, {"full": ids}, PAIRS)
    try:
        from recalc_erreason_judge import SYMPTOM
        sym = [c for c in ids if c in SYMPTOM]
        dis = [c for c in ids if c not in SYMPTOM]
        strata = {"symptom": sym, "disease": dis}
    except Exception:
        strata = {}
    md_rows = []
    for sname, sids in strata.items():
        row = {"stratum": sname, "n": len(sids)}
        for arm in ("Ax1", "P", "MDT", "ErAx5", "ErAx5Mod"):
            vals = []
            for s in SEEDS:
                hits = [cs.topk(cs.hit_flags(arms[arm][s][c], cache), k)
                        for c in sids for k in (5,)]
                per_seed5 = 100 * sum(1 for c in sids
                                      if cs.topk(cs.hit_flags(arms[arm][s][c], cache), 5)) / len(sids)
                vals.append(per_seed5)
            row[arm] = round(sum(vals) / len(vals), 1)
        md_rows.append(row)
    meta = {"n_cases": len(ids), "n_seeds": len(SEEDS), "seeds": SEEDS,
            "sample_calls": len(ids) * len(SEEDS) * N_SAMPLES,
            "mod_calls": len(ids) * len(SEEDS),
            "sample_temp": TEMP_SAMPLE,
            "moderator": "qwen3.8-flash, T=0.3, max_tokens=4096, disable_thinking（提示词与 CPC Ax5Mod 逐字一致）",
            "outdir": str(OUTDIR), "judge_cache": str(cs.GLM_CACHE),
            "judge_model": cs.GLM_MODEL,
            "strata": {r["stratum"]: {"n": r["n"], **{a: r[a] for a in
                       ("Ax1", "P", "MDT", "ErAx5", "ErAx5Mod")}} for r in md_rows}}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def render_md(meta, stats):
    L = ["# Ax5Mod 在 ER-Reason 上的补测（同提示 5 采样 + 主持人）\n",
         f"- 采样：A_TOPN_PROMPT，T={meta['sample_temp']}，每例每 seed 5 份"
         f"（{meta['sample_calls']} 次调用）；主持人：{meta['moderator']}"
         f"（{meta['mod_calls']} 次）。判官/口径与主实验一致。",
         f"- 标签分层 top-5（5-seed 均值）：{meta['strata']}\n"]
    for name, res in stats.items():
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for arm in ("ErAx5Mod", "ErAx5", "Ax1", "P", "MDT"):
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
    cases = load_cases()
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"数据集: {len(cases)} 例 | 输出: {OUTDIR}", flush=True)
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
