#!/usr/bin/env python3
"""2×2 析因补全：A×5 采样 + LLM 主持人（Ax5Mod 臂）。

回应评审追问："如果主持人重要，A×5 加上主持人会不会也增益？"
现有四格：
  A×5(Borda/SC)  = 同提示采样 + 机械聚合  → ≈A×1（ax5_full87）
  MDT-Borda      = 隔离角色   + 机械聚合  → ≈A×1（mdt_nomoderator）
  MDT            = 隔离角色   + 主持人    → 召回增益（主实验）
  Ax5Mod（本臂）  = 同提示采样 + 主持人    → 本脚本补测

推理：对已存档的 A×5 五份采样（results/topn_seeds_ax5/Ax5_s{1..5}.jsonl 的
`samples` 字段，T=0.7 采样，逐字不变的存档），每例每 seed 发起一次主持人
调用。主持人提示词与 mdt_cpc.py 的 moderator 逐字一致，仅第一句适配输入
事实（五份独立评估，而非"five specialists, each through their own lens"），
差异在此披露并存档于 results/prompt_actually_sent_ax5mod.md。
调用设置与 MDT 主持人一致：qwen3.8-flash，T=0.3，max_tokens=4096，
disable_thinking，timeout=300。每例每 seed 1 次调用，共 87×5=435 次。

阶段（PHASE=infer|judge|analyze|all，默认 all）：
  infer   → results/topn_ax5_mod/Ax5Mod_s{1..5}.jsonl（断点续跑）
  judge   → 只补 GLM v3 共享缓存缺失的 (gold, cand) 对
  analyze → results/ax5_mod_cpc.{json,md}：病例级配对检验
            （Wilcoxon + cluster bootstrap 10000 + 多数决精确 McNemar），
            对照 A×1 / P / MDT / A×5(Borda)，含 held-out 46 / dev 41。

环境变量：AX5MOD_OUTDIR、AX5MOD_LIMIT（冒烟）、GLM_CACHE。
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
from run_inference import call_llm  # noqa: E402
from topn_cpc import load_done, append_row, parse_top5, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
AX5DIR = RESULTS / "topn_seeds_ax5"
OUTDIR = Path(os.environ.get("AX5MOD_OUTDIR", RESULTS / "topn_ax5_mod"))
SEEDS = [1, 2, 3, 4, 5]
LIMIT = int(os.environ.get("AX5MOD_LIMIT", "0"))
PROVIDER = "qwen"
TEMP = 0.3
MAX_TOKENS = 4096
TIMEOUT = 300
MAX_ATTEMPTS = 3
OUT_JSON = RESULTS / "ax5_mod_cpc.json"
OUT_MD = RESULTS / "ax5_mod_cpc.md"
QWEN_SEEDS = RESULTS / "topn_seeds"
QWEN_MDT = RESULTS / "topn_mdt"
PAIRS = [("Ax5Mod", "Ax1"), ("Ax5Mod", "Ax5"), ("Ax5Mod", "P"),
         ("Ax5Mod", "MDT"), ("MDT", "Ax1")]
ARM_LABEL = {"Ax5Mod": "A×5+Mod（同提示采样 + 主持人）",
             "Ax5": "A×5 Borda（同提示采样 + 机械聚合）",
             "Ax1": "A×1 (T=0)", "P": "P (T=0.3)", "MDT": "MDT（5 角色 + 主持人）"}


def row_path(seed):
    return OUTDIR / f"Ax5Mod_s{seed}.jsonl"


def moderator_prompt(case_text, opinions_text):
    # 与 mdt_cpc.py 的 moderator 逐字一致，仅第一句适配输入事实（见 docstring）。
    return f"""You are the moderator of an MDT panel. Five independent assessments of the MGH CPC case below were produced without seeing one another. Their ranked candidate lists are given.

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


def opinions_from_samples(samples):
    lines = []
    for i, lst in enumerate(samples, 1):
        lines.append(f"Assessment {i}:")
        for j, dx in enumerate(lst[:5], 1):
            lines.append(f"  {j}. {dx}")
    return "\n".join(lines)


def call_moderator(prompt):
    raw, usage = call_llm(prompt, temperature=TEMP, max_tokens=MAX_TOKENS,
                          timeout=TIMEOUT, provider=PROVIDER,
                          disable_thinking=True)
    top5 = parse_top5(raw)
    seen, out = set(), []
    for dx in top5[:5]:
        k = dx.lower()
        if k not in seen:
            seen.add(k)
            out.append(dx)
    return out, (usage or {}).get("total_tokens", 0), raw


def run_infer(cases):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gold = {c["case_id"]: c["gold"] for c in cases}
    text = {c["case_id"]: c["text"] for c in cases}
    for seed in SEEDS:
        src = load_done(AX5DIR / f"Ax5_s{seed}.jsonl")
        path = row_path(seed)
        done = load_done(path)
        todo = [c for c in cases
                if c["case_id"] in src and c["case_id"] not in done]
        print(f"[Ax5Mod s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            samples = src[c["case_id"]].get("samples") or []
            assert len(samples) == 5, f"{c['case_id']} 样本数 {len(samples)} != 5"
            prompt = moderator_prompt(c["text"],
                                      opinions_from_samples(samples))
            last = ""
            for attempt in range(MAX_ATTEMPTS):
                top5, tokens, raw = call_moderator(prompt)
                last = raw or ""
                if len(top5) == 5:
                    return {"case_id": c["case_id"], "gold": c["gold"],
                            "top5": top5, "total_tokens": tokens}
                print(f"[Ax5Mod s{seed}] {c['case_id'][:40]} 解析 {len(top5)} 项，"
                      f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
            raise RuntimeError(f"{c['case_id']} 连续 {MAX_ATTEMPTS} 次不足 5 项 | "
                               f"末次片段: {last.strip()[:300]!r}")

        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"[失败] Ax5Mod s{seed} {c['case_id'][:40]}: {e}",
                          flush=True)
                    continue
                append_row(path, row)
                print(f"[Ax5Mod s{seed}] {i}/{len(todo)} {row['case_id'][:40]}: "
                      f"{row['top5'][:1]}", flush=True)
        rows = load_done(path)
        missing = [c["case_id"] for c in cases
                   if c["case_id"] in gold and c["case_id"] not in rows]
        assert not missing, f"s{seed} 缺行: {missing}"
        print(f"[Ax5Mod s{seed}] 完整 {len(rows)}/87", flush=True)


def run_judge():
    rows = []
    for seed in SEEDS:
        rows.extend(load_done(row_path(seed)).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text()))
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)


def build_arms():
    ax5mod = {s: load_done(row_path(s)) for s in SEEDS}
    ax5 = {s: load_done(AX5DIR / f"Ax5_s{s}.jsonl") for s in SEEDS}
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
    return {"Ax5Mod": ax5mod, "Ax5": ax5, **refs}


def run_analyze():
    arms = build_arms()
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    stats = cs.split_stats(arms, cs.split_ids(), PAIRS)
    missing = [0]
    for seed in SEEDS:
        for row in arms["Ax5Mod"][seed].values():
            cs.hit_flags(row, cache, missing=missing)
    meta = {"n_cases": len(arms["Ax5Mod"][SEEDS[0]]), "n_seeds": len(SEEDS),
            "seeds": SEEDS, "n_calls": len(arms["Ax5Mod"][SEEDS[0]]) * len(SEEDS),
            "outdir": str(OUTDIR), "samples_source": str(AX5DIR),
            "judge_cache": str(cs.GLM_CACHE), "judge_model": cs.GLM_MODEL,
            "moderator_call": "qwen3.8-flash, T=0.3, max_tokens=4096, disable_thinking（与 MDT 主持人一致）",
            "prompt_diff": "与 mdt_cpc.py moderator 逐字一致，仅首句 'Five specialists ... each through their own lens' → 'Five independent assessments ... without seeing one another'",
            "unadjudicated_pairs_ax5mod": missing[0]}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def render_md(meta, stats):
    L = ["# 2×2 析因补全：A×5 采样 + LLM 主持人（Ax5Mod）\n",
         "- 对已存档的 A×5 五份采样（`topn_seeds_ax5/`，T=0.7 采样存档逐字不变）"
         "每例每 seed 发起一次主持人调用；主持人提示词与 mdt_cpc.py 的 moderator "
         "逐字一致、仅首句适配输入事实（披露见 Methods/补充材料）。判官/口径与"
         "主实验一致（GLM-5.3-flash × v3 冻结缓存，病例级配对检验）。",
         f"- 新增调用 {meta['n_calls']} 次；Ax5Mod 未判定对（按 miss 计）："
         f"{meta['unadjudicated_pairs_ax5mod']}\n"]
    for name, res in stats.items():
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for arm in ("Ax5Mod", "Ax5", "Ax1", "P", "MDT"):
            if arm not in res["arms"]:
                continue
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
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"数据集: {len(cases)} 例 | 输出: {OUTDIR} | GLM 缓存: {cs.GLM_CACHE}",
          flush=True)
    t0 = time.time()
    if phase in ("all", "infer"):
        run_infer(cases)
    if phase in ("all", "judge"):
        run_judge()
    if phase in ("all", "analyze"):
        run_analyze()
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
