#!/usr/bin/env python3
"""第五格：P-split = P 的五视角生成（去内置综合）+ 独立主持人。

回应作者追问："P 表现差，是不是因为 P 里综合者不独立？把 P 拆成
'视角生成（不综合）' + '独立主持人' 会不会就好？"

该臂把 P 的失败分解为两个变量：
  (a) 生成时相互条件化（五视角顺序发言、后者可见前者）
  (b) 裁决者不独立（同一上下文自己综合）
现 P 同时具备 (a)(b)；MDT/Ax5Mod 两者皆无；P-split 保留 (a)、消除 (b)：
  P-split ≈ MDT/Ax5Mod → 毒药是 (b)：独立的是裁决而非证人
  P-split ≈ P/A×1      → 毒药是 (a)：条件化使候选池先坍缩，主持人无米可炊

推理（两阶段，均 qwen3.8-flash，T=0.3，max_tokens=4096，disable_thinking）：
  gen → results/topn_p_split/Pgen_s{1..5}.jsonl
        PERSPECTIVE_PROMPT 截断至 "## Synthesis Task" 之前（即只要求五段
        视角分析），并追加一行"只输出五段分析、不得综合"的显式指令。
        每例每 seed 1 次，87×5=435 次；行内存 analyses 原文。
  mod → results/topn_p_split/Psplit_s{1..5}.jsonl
        主持人提示词与 ax5_mod_cpc.py 逐字一致，仅首句如实描述输入：
        五份分析系同一上下文顺序写成、后者可见前者（这正是本臂的操纵点，
        必须如实告知裁判）。panel opinions = 五段分析原文。435 次。
  judge → GLM v3 冻结缓存只补缺失对。
  analyze → results/p_split_cpc.{json,md}：vs P / A×1 / MDT / Ax5Mod 的
            病例级配对检验（口径同主实验）。

环境变量：PSPLIT_OUTDIR、PSPLIT_LIMIT（冒烟）、PSPLIT_SEEDS。
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
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("PSPLIT_OUTDIR", RESULTS / "topn_p_split"))
SEEDS = [int(s) for s in os.environ.get("PSPLIT_SEEDS", "1,2,3,4,5").split(",") if s.strip()]
LIMIT = int(os.environ.get("PSPLIT_LIMIT", "0"))
TEMP = 0.3
MAX_TOKENS = 4096
TIMEOUT = 300
MAX_ATTEMPTS = 3
OUT_JSON = RESULTS / "p_split_cpc.json"
OUT_MD = RESULTS / "p_split_cpc.md"
GEN_SUFFIX = ("\nOutput only the five perspective analyses above, each under its bold header, "
              "2-4 sentences each. Do not synthesize them, do not weigh them against one another, "
              "and do not state a final diagnosis.\n")
GEN_CUT = "## Synthesis Task"
PAIRS = [("Psplit", "P"), ("Psplit", "Ax1"), ("Psplit", "MDT"), ("Psplit", "Ax5Mod"),
         ("MDT", "Ax1")]
ARM_LABEL = {"Psplit": "P-split（条件化视角 + 独立主持人）",
             "P": "P（条件化视角 + 自我综合）", "Ax1": "A×1 (T=0)",
             "MDT": "MDT（隔离角色 + 主持人）", "Ax5Mod": "A×5+Mod（采样 + 主持人）"}


def gen_path(seed):
    return OUTDIR / f"Pgen_s{seed}.jsonl"


def split_path(seed):
    return OUTDIR / f"Psplit_s{seed}.jsonl"


def gen_prompt(case_text):
    base = PERSPECTIVE_PROMPT.split(GEN_CUT)[0].rstrip()
    return base.format(structured_case=case_text) + "\n\n" + GEN_SUFFIX


def split_moderator_prompt(case_text, analyses_text):
    # 与 ax5_mod_cpc.moderator_prompt 逐字一致，仅首句如实描述输入性质
    # （顺序写成、相互可见——这正是本臂保留的操纵点）。
    return f"""You are the moderator of an MDT panel. Five analyses of the MGH CPC case below are given; they were written sequentially in a single context, each written with sight of the previous ones.

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
{analyses_text}
"""


def parse5(raw):
    from topn_cpc import parse_top5
    seen, out = set(), []
    for dx in parse_top5(raw)[:5]:
        k = dx.lower()
        if k not in seen:
            seen.add(k)
            out.append(dx)
    return out


def run_gen(cases):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        done = load_done(gen_path(seed))
        todo = [c for c in cases if c["case_id"] not in done]
        print(f"[Pgen s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            prompt = gen_prompt(c["text"])
            last = ""
            for attempt in range(MAX_ATTEMPTS):
                raw, usage = call_llm(prompt, temperature=TEMP,
                                      max_tokens=MAX_TOKENS, timeout=TIMEOUT,
                                      provider="qwen", disable_thinking=True)
                last = raw or ""
                analyses = raw.strip()
                if len(analyses) > 200 and "Perspective" in analyses:
                    return {"case_id": c["case_id"], "gold": c["gold"],
                            "analyses": analyses,
                            "total_tokens": (usage or {}).get("total_tokens", 0)}
                print(f"[Pgen s{seed}] {c['case_id'][:40]} 输出异常，"
                      f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
            raise RuntimeError(f"{c['case_id']} 视角生成连续失败 | {last[:200]!r}")

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    append_row(gen_path(seed), fut.result())
                except Exception as e:
                    print(f"[失败] Pgen s{seed} {c['case_id'][:40]}: {e}", flush=True)
                    continue
                if i % 25 == 0 or i == len(todo):
                    print(f"[Pgen s{seed}] {i}/{len(todo)} ({time.time() - t0:.0f}s)",
                          flush=True)
        rows = load_done(gen_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"Pgen s{seed} 缺行: {missing}"
        print(f"[Pgen s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_mod(cases):
    for seed in SEEDS:
        src = load_done(gen_path(seed))
        done = load_done(split_path(seed))
        todo = [c for c in cases
                if c["case_id"] in src and c["case_id"] not in done]
        print(f"[Psplit s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

        def work(c):
            analyses = src[c["case_id"]]["analyses"]
            prompt = split_moderator_prompt(c["text"], analyses)
            last = ""
            for attempt in range(MAX_ATTEMPTS):
                raw, usage = call_llm(prompt, temperature=TEMP,
                                      max_tokens=MAX_TOKENS, timeout=TIMEOUT,
                                      provider="qwen", disable_thinking=True)
                last = raw or ""
                top5 = parse5(raw)
                tokens = (usage or {}).get("total_tokens", 0)
                if len(top5) == 5:
                    return {"case_id": c["case_id"], "gold": c["gold"],
                            "top5": top5, "total_tokens": tokens}
                print(f"[Psplit s{seed}] {c['case_id'][:40]} 解析 {len(top5)} 项，"
                      f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
            raise RuntimeError(f"{c['case_id']} 连续不足 5 项 | {last.strip()[:200]!r}")

        t0 = time.time()
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    append_row(split_path(seed), fut.result())
                except Exception as e:
                    print(f"[失败] Psplit s{seed} {c['case_id'][:40]}: {e}", flush=True)
                    continue
                if i % 25 == 0 or i == len(todo):
                    print(f"[Psplit s{seed}] {i}/{len(todo)} ({time.time() - t0:.0f}s)",
                          flush=True)
        rows = load_done(split_path(seed))
        missing = [c["case_id"] for c in cases if c["case_id"] not in rows]
        assert not missing, f"Psplit s{seed} 缺行: {missing}"
        print(f"[Psplit s{seed}] 完整 {len(rows)}/{len(cases)}", flush=True)


def run_judge():
    rows = []
    for seed in SEEDS:
        rows.extend(load_done(split_path(seed)).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text()))
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)


def run_analyze():
    arms = {"Psplit": {s: load_done(split_path(s)) for s in SEEDS},
            "P": {s: cs.load(RESULTS / "topn_seeds" / f"P_s{s}.jsonl") for s in SEEDS},
            "Ax1": {s: cs.load(RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl") for s in SEEDS},
            "MDT": {s: cs.load(RESULTS / "topn_mdt"
                               / ("synthesis.jsonl" if s == 1
                                  else f"s{s}/synthesis.jsonl")) for s in SEEDS},
            "Ax5Mod": {s: cs.load(RESULTS / "topn_ax5_mod" / f"Ax5Mod_s{s}.jsonl")
                       for s in SEEDS}}
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    stats = cs.split_stats(arms, cs.split_ids(), PAIRS)
    missing = [0]
    for seed in SEEDS:
        for row in arms["Psplit"][seed].values():
            cs.hit_flags(row, cache, missing=missing)
    meta = {"n_cases": len(arms["Psplit"][SEEDS[0]]), "n_seeds": len(SEEDS),
            "seeds": SEEDS, "outdir": str(OUTDIR),
            "judge_cache": str(cs.GLM_CACHE), "judge_model": cs.GLM_MODEL,
            "gen_prompt": "PERSPECTIVE_PROMPT 截断至 '## Synthesis Task' 前 + 只输出五段分析的显式指令",
            "mod_prompt_diff": "与 ax5_mod_cpc.moderator_prompt 逐字一致，仅首句如实描述'顺序写成、相互可见'",
            "unadjudicated_pairs_psplit": missing[0]}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def render_md(meta, stats):
    L = ["# 第五格：P-split（条件化五视角 + 独立主持人）\n",
         "- 分解 P 的失败：保留 (a) 生成时相互条件化、消除 (b) 裁决者不独立。"
         "若 Psplit ≈ MDT/Ax5Mod → 毒药是自我综合；若 Psplit ≈ P/A×1 → "
         "条件化使候选池先坍缩。判官/口径与主实验一致。",
         f"- Psplit 未判定对（按 miss 计）：{meta['unadjudicated_pairs_psplit']}\n"]
    for name, res in stats.items():
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 |")
        L.append("|---|---|---|---|")
        for arm in ("Psplit", "P", "Ax1", "MDT", "Ax5Mod"):
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
    print(f"数据集: {len(cases)} 例 | 输出: {OUTDIR}", flush=True)
    t0 = time.time()
    if phase in ("all", "gen"):
        run_gen(cases)
    if phase in ("all", "mod"):
        run_mod(cases)
    if phase in ("all", "judge"):
        run_judge()
    if phase in ("all", "analyze"):
        run_analyze()
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
