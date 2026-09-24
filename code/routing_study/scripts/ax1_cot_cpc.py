#!/usr/bin/env python3
"""A×1+CoT 对照臂：单次调用 + 推理引出（回应"策略 vs 推理引出"混杂的评审意见）。

背景：策略 A×1 的提示词要求"只输出 JSON、不产生推理文本"，而 P/MDT 都产出
推理文本。于是 A×1 vs P/MDT 的差异里，"策略/结构"与"是否被要求推理"两件事
混杂。本臂补上 A×1+CoT：
- 与 A×1(T=0) 完全同构：单次调用、同一模型（qwen3.8-flash，disable_thinking）、
  同一温度 T=0.0、同一病例文本（load_merged，逐字节一致）；
- 唯一差别是提示词要求先给出简明逐步推理，再以恰好一个 JSON 代码块给出 top-5。
因此 A×1+CoT vs A×1 的差 = 纯 elicitation 效应（无角色分工、无多次调用、
无聚合），A×1+CoT vs P/MDT 的残余差 = 结构/多智能体效应。

提示词：以 promptv2 的 A_TOPN_PROMPT 为骨架（Guidelines 逐字保留、JSON schema
逐字保留），在"只输出 JSON"处替换为下述推理要求；不含任何角色分工（角色分工是
P/MDT 的属性）。最终提示词全文见常量 A_TOPN_COT_PROMPT —— 下面是其中新增的
那一段（逐字，即可直接放进补充材料）：

    First reason through the case in a concise step-by-step analysis (5-10 sentences):
    1. State the discriminating features: the key positives and the key negatives.
    2. Generate candidate diagnoses and weigh them against each other on how well each explains the whole picture.
    3. Explain why the leading candidate is favoured over the strongest alternatives.
    Then finish with exactly one JSON code block (the only JSON in your answer), preceded by no other JSON and followed by nothing:
    ```json
    {"top5": [{"rank": 1, "diagnosis": "..."}, {"rank": 2, "diagnosis": "..."}, {"rank": 3, "diagnosis": "..."}, {"rank": 4, "diagnosis": "..."}, {"rank": 5, "diagnosis": "..."}]}
    ```
    Exactly 5 items, ranked most to least likely.

推理参数：temperature=0.0、provider="qwen"、disable_thinking=True、
max_tokens=4096（比 A×1 的 2048 大，因为输出里要容纳推理文本；A×1/P 的
2048 会让"推理+JSON"被截断，从而把 elicitation 混成"截断"）。

输出：routing_study/results/topn_seeds_cot/Ax1cot_s{1..5}.jsonl，
每行 {case_id, gold, top5, total_tokens, reasoning_chars}，其中
reasoning_chars = JSON 代码块之前的文本长度（证明推理确实产生了）。
断点续跑：load_done 跳过已完成；空 top5 重试 3 次后放弃并记入 failures.jsonl。

判定：复用 ax1_t03_sensitivity.py 的 GLM-5.3-flash × judge_v3.V3_PROMPT 范式
（6 并发、150s 超时、最多 3 轮），共享缓存 judge_cache_glm_v3.json
（键 gold[:150]+"||"+cand[:150]），只补缺失对。

分析（口径与 stats_caselevel.py 完全一致，逐字复刻该文件的实现：
病例级 5-seed 命中率 → 配对 Wilcoxon 双侧符号秩；病例级 cluster bootstrap
10,000 次 95% 百分位 CI；多数决 >=3/5 精确 McNemar），对比 A×1(T=0) / P / MDT，
并在 CPC87 全量上报 held-out 46 例（data/mgh_qa_dataset_new_cases.json）与
dev 41 例子集。产出 results/ax1_cot_cpc.json + .md。

PHASE 环境变量：infer / judge / analyze / all（可逗号组合）。
未指定 PHASE 时脚本不做任何事 —— 推理阶段必须显式 PHASE=infer。
SMOKE=1 只跑前 3 例到 /tmp/cot_smoke（不判定、不分析）。
"""
import json
import os
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from run_inference import call_llm  # noqa: E402
from topn_cpc import (MAX_WORKERS, TIMEOUT, append_row,  # noqa: E402
                      load_done, parse_top5)
from topn_cpc_promptv2 import GUIDELINES  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from ax1_t03_sensitivity import judge_missing, key_of  # noqa: E402
from seeds_87 import mcnemar_exact  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
COT_OUTDIR = RESULTS / "topn_seeds_cot"
SEEDS = (1, 2, 3, 4, 5)
MAX_TOKENS = 4096
OUT_JSON = RESULTS / "ax1_cot_cpc.json"
OUT_MD = RESULTS / "ax1_cot_cpc.md"
GLM_CACHE = RESULTS / "judge_cache_glm_v3.json"
STATS_JSON = RESULTS / "stats_caselevel.json"
HELD_OUT_JSON = ROOT / "data" / "mgh_qa_dataset_new_cases.json"
SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/cot_smoke"))

A_TOPN_COT_PROMPT = """You are an expert internist reviewing an MGH CPC case.

Produce a ranked differential diagnosis: the 5 most likely diagnoses, most likely first. Be specific (disease name plus the key qualifier that matters for this case).

""" + GUIDELINES + """

First reason through the case in a concise step-by-step analysis (5-10 sentences):
1. State the discriminating features: the key positives and the key negatives.
2. Generate candidate diagnoses and weigh them against each other on how well each explains the whole picture.
3. Explain why the leading candidate is favoured over the strongest alternatives.
Then finish with exactly one JSON code block (the only JSON in your answer), preceded by no other JSON and followed by nothing:
```json
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}
```
Exactly 5 items, ranked most to least likely.

Case:
{case_text}
"""

# 与 webapp.prompts.extract_json 同形的围栏正则；取最后一个（判分/解析取的就是它）
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def reasoning_length(raw):
    """JSON 代码块之前的文本长度；无围栏时退化为最后一个 '{' 之前。"""
    raw = raw or ""
    blocks = list(_FENCE.finditer(raw))
    if blocks:
        return blocks[-1].start()
    i = raw.rfind("{")
    return i if i > 0 else len(raw)


def infer_one(case_text):
    raw, usage = call_llm(A_TOPN_COT_PROMPT.format(case_text=case_text),
                          temperature=0.0, max_tokens=MAX_TOKENS,
                          timeout=TIMEOUT, provider="qwen",
                          disable_thinking=True)
    raw = raw or ""
    return raw, parse_top5(raw), (usage or {}).get("total_tokens", 0)


def run_seed(seed, cases, outdir):
    path = outdir / f"Ax1cot_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[Ax1cot s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        rc = 0
        for attempt in range(3):
            raw, top5, tokens = infer_one(c["text"])
            rc = reasoning_length(raw)
            if top5:
                return {"case_id": c["case_id"], "gold": c["gold"],
                        "top5": top5, "total_tokens": tokens,
                        "reasoning_chars": rc}
            print(f"[Ax1cot s{seed}] {c['case_id'][:40]} 空 top5"
                  f"（reasoning_chars={rc}），重试 {attempt + 1}/3", flush=True)
        append_row(outdir / "failures.jsonl",
                   {"case_id": c["case_id"], "seed": seed, "gold": c["gold"],
                    "reasoning_chars": rc, "note": "连续 3 次空 top5"})
        raise RuntimeError(f"{c['case_id']} 连续 3 次空 top5")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                append_row(path, fut.result())
            except Exception as e:
                print(f"[Ax1cot s{seed}] {c['case_id'][:40]} 失败: {e}",
                      flush=True)
                continue
            if i % 20 == 0 or i == len(todo):
                print(f"[Ax1cot s{seed}] {i}/{len(todo)}", flush=True)


# ---------- 判分后分析：口径逐字复刻 stats_caselevel.py ----------

N_BOOT = 10000
# 与 stats_caselevel.py 同种子的模块级 RNG；基础三对的调用顺序也与之对齐，
# 因此 CPC87 上复算的 bootstrap CI 可与 stats_caselevel.json 逐位对上（见
# sanity_check_vs_stats_caselevel）。
RNG = np.random.default_rng(20260917)

METHODS = ["Ax1cot", "Ax1", "P", "MDT"]
NAME = {"Ax1cot": "A×1+CoT (T=0)", "Ax1": "A×1 (T=0)", "P": "P (T=0.3)",
        "MDT": "MDT (T=0.3)"}
# 基础三对放在最前：与 stats_caselevel.py 的 PAIRS 顺序一致（用于口径核对）
PAIRS_BASE = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]
PAIRS_COT = [("Ax1cot", "Ax1"), ("Ax1cot", "P"), ("Ax1cot", "MDT")]


def path_of(method, seed):
    if method == "Ax1cot":
        return COT_OUTDIR / f"Ax1cot_s{seed}.jsonl"
    if method == "Ax1":
        return RESULTS / "topn_seeds" / f"Ax1_s{seed}.jsonl"
    if method == "P":
        return RESULTS / "topn_seeds" / f"P_s{seed}.jsonl"
    return (RESULTS / "topn_mdt" / "synthesis.jsonl" if seed == 1
            else RESULTS / "topn_mdt" / f"s{seed}" / "synthesis.jsonl")


def load_jsonl(path):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(path) if l.strip()}


def topk(flags, k):
    f = flags[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def boot_ci(diffs):
    d = np.asarray(diffs)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def per_seed_metrics(rows_by_case):
    n = len(rows_by_case)
    t = {1: 0, 3: 0, 5: 0}
    for rec in rows_by_case.values():
        verdicts = [bool(JUDGE.get(key_of(rec["gold"], c)))
                    for c in rec["top5"][:5]]
        hit = next((r for r, v in enumerate(verdicts, 1) if v), None)
        if hit:
            t[1] += hit == 1
            t[3] += hit <= 3
            t[5] += 1
    return {"n": n, **{f"top{k}": t[k] for k in (1, 3, 5)},
            **{f"top{k}_acc": round(t[k] / n, 4) if n else None
               for k in (1, 3, 5)}}


def case_rates(method, ids, k):
    rates = {}
    for cid in ids:
        vals = []
        for s in SEEDS:
            flags = HITS.get((method, s, cid))
            if flags is None:
                vals = None
                break
            vals.append(topk(flags, k))
        if vals is None or any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / len(SEEDS)
    return rates


def case_majority(method, ids, k):
    out = {}
    for cid in ids:
        vals = []
        for s in SEEDS:
            flags = HITS.get((method, s, cid))
            if flags is None:
                vals = None
                break
            vals.append(topk(flags, k))
        if vals is None or any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) >= 3)
    return out


def compare(a, b, k):
    ra, rb = RATES[k][a], RATES[k][b]
    common = sorted(set(ra) & set(rb))
    va = np.array([ra[c] for c in common])
    vb = np.array([rb[c] for c in common])
    diff = va - vb
    if np.all(diff == 0):
        wp = 1.0
    else:
        wp = float(wilcoxon(va, vb, zero_method="wilcox").pvalue)
    md, lo, hi = boot_ci(diff)

    ma, mb = MAJ[k][a], MAJ[k][b]
    cm = sorted(set(ma) & set(mb))
    ao = sum(1 for c in cm if ma[c] and not mb[c])
    bo = sum(1 for c in cm if mb[c] and not ma[c])
    mp = mcnemar_exact(ao, bo)

    pao = pbo = 0
    for s in SEEDS:
        for cid in SPLIT_IDS:
            ha = topk(HITS.get((a, s, cid)) or [], k)
            hb = topk(HITS.get((b, s, cid)) or [], k)
            if ha and not hb:
                pao += 1
            elif hb and not ha:
                pbo += 1
    return {
        "n_cases": len(common),
        "mean_rate_a": float(va.mean()) if len(va) else None,
        "mean_rate_b": float(vb.mean()) if len(vb) else None,
        "mean_diff": md,
        "boot95_ci": [lo, hi],
        "wilcoxon_p": wp,
        "majority": {"a_only": ao, "b_only": bo, "mcnemar_p": mp},
        "pooled_mcnemar": {"a_only": pao, "b_only": pbo,
                           "p": mcnemar_exact(pao, pbo)},
    }


def fmt_p(p):
    if p is None:
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def run_analyze():
    global JUDGE, HITS, RATES, MAJ, SPLIT_IDS
    JUDGE = json.loads(GLM_CACHE.read_text())
    RATES, MAJ, SPLIT_IDS = {}, {}, []

    runs = {}
    for m in METHODS:
        for s in SEEDS:
            p = path_of(m, s)
            runs[(m, s)] = load_jsonl(p) if p.exists() else {}

    empty = sorted(f"{m}_s{s}" for (m, s), rows in runs.items() if not rows)
    if empty:
        print(f"[分析] 注意：以下运行缺失或为空 {empty}", flush=True)
    for s in SEEDS:
        n = len(runs[("Ax1cot", s)])
        if n and n < 87:
            missing = sorted(set(runs[("Ax1", 1)]) - set(runs[("Ax1cot", s)]))
            print(f"[分析] Ax1cot s{s} 只有 {n} 例，缺 {missing}", flush=True)

    n_missing = 0
    HITS = {}
    for (m, s), cases in runs.items():
        for cid, rec in cases.items():
            flags = []
            for c in rec["top5"][:5]:
                k = key_of(rec["gold"], c)
                if k in JUDGE:
                    flags.append(bool(JUDGE[k]))
                else:
                    n_missing += 1
                    flags.append(None)
            HITS[(m, s, cid)] = flags
    print(f"[分析] 判定缓存 {len(JUDGE)} 对 | 本分析缺失 {n_missing} 对（按 miss 计）",
          flush=True)

    universe = sorted(runs[("Ax1", 1)])
    held_out = {c["case_id"] for c in json.loads(HELD_OUT_JSON.read_text())}
    splits = [("CPC87", universe),
              ("dev41", [c for c in universe if c not in held_out]),
              ("heldout46", [c for c in universe if c in held_out])]

    out = {"arm": "Ax1cot", "config": {
        "model": os.environ.get("QWEN_MODEL", "qwen3.8-flash"),
        "temperature": 0.0, "provider": "qwen", "disable_thinking": True,
        "max_tokens": MAX_TOKENS, "seeds": list(SEEDS),
        "judge": "GLM-5.3-flash × judge_v3.V3_PROMPT",
        "judge_cache": str(GLM_CACHE.relative_to(ROOT)),
        "missing_judge_pairs": n_missing,
    }, "per_seed": {}, "mean_sd": {}, "caselevel": {}}

    # 1) 逐 seed 准确率（与 seeds_87 / ax1_t03_sensitivity 的表同口径）
    per_seed = {}
    for m in METHODS:
        accs = {k: [] for k in (1, 3, 5)}
        for s in SEEDS:
            if not runs[(m, s)]:
                continue
            met = per_seed_metrics(runs[(m, s)])
            per_seed[f"{m}_s{s}"] = met
            for k in (1, 3, 5):
                accs[k].append(met[f"top{k}_acc"])
        out["mean_sd"][m] = {
            f"top{k}": {"mean": st.mean(accs[k]) if accs[k] else None,
                        "sd": st.stdev(accs[k]) if len(accs[k]) > 1 else 0.0,
                        "n_seeds": len(accs[k])}
            for k in (1, 3, 5)}
    out["per_seed"] = per_seed
    cot_rows = [r for s in SEEDS for r in runs[("Ax1cot", s)].values()]
    if cot_rows:
        out["reasoning"] = {
            "mean_reasoning_chars": st.mean(r["reasoning_chars"] for r in cot_rows),
            "min_reasoning_chars": min(r["reasoning_chars"] for r in cot_rows),
            "zero_reasoning_cases": sum(1 for r in cot_rows
                                        if r["reasoning_chars"] == 0),
            "n": len(cot_rows),
            "mean_total_tokens": st.mean(r["total_tokens"] for r in cot_rows),
        }

    # 2) 病例级主口径（RNG 调用顺序：先基础三对，再本臂三对）
    for name, ids in splits:
        SPLIT_IDS = ids
        res = {"n_cases": len(ids), "topk": {}}
        for k in (1, 3, 5):
            RATES[k] = {m: case_rates(m, ids, k) for m in METHODS}
            MAJ[k] = {m: case_majority(m, ids, k) for m in METHODS}
            res["topk"][k] = {
                "case_rate_mean": {
                    m: st.mean(RATES[k][m].values()) if RATES[k][m] else None
                    for m in METHODS},
                "comparisons": {},
            }
        for k in (1, 3, 5):
            for a, b in PAIRS_BASE:
                res["topk"][k]["comparisons"][f"{a}_vs_{b}"] = compare(a, b, k)
        for k in (1, 3, 5):
            for a, b in PAIRS_COT:
                res["topk"][k]["comparisons"][f"{a}_vs_{b}"] = compare(a, b, k)
        out["caselevel"][name] = res

    # 3) 口径核对：CPC87 上基础三对必须与 stats_caselevel.json 逐字段一致
    check = {"reference": str(STATS_JSON.relative_to(ROOT)), "fields": {}}
    if STATS_JSON.exists():
        ref = json.loads(STATS_JSON.read_text())["CPC87"]["topk"]
        ok_all = True
        for k in (1, 3, 5):
            for a, b in PAIRS_BASE:
                key = f"{a}_vs_{b}"
                mine = out["caselevel"]["CPC87"]["topk"][k]["comparisons"][key]
                theirs = ref[str(k)]["comparisons"][key]
                for f in ("mean_diff", "wilcoxon_p", "pooled_mcnemar",
                          "majority", "mean_rate_a", "mean_rate_b", "n_cases"):
                    same = mine[f] == theirs[f]
                    ok_all &= same
                    check["fields"][f"top{k}/{key}/{f}"] = same
                same = ([round(x, 10) for x in mine["boot95_ci"]]
                        == [round(x, 10) for x in theirs["boot95_ci"]])
                ok_all &= same
                check["fields"][f"top{k}/{key}/boot95_ci"] = same
        check["identical"] = ok_all
        print(f"[口径核对] 与 stats_caselevel.py 一致: {ok_all}", flush=True)
        for k, v in check["fields"].items():
            if not v:
                print(f"  [口径核对] 不一致: {k}", flush=True)
    else:
        check["identical"] = None
        print("[口径核对] 未找到 stats_caselevel.json，跳过", flush=True)
    out["sanity_check_vs_stats_caselevel"] = check

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- md ----
    lines = []
    lines.append("# A×1+CoT 对照臂（单次调用 + 推理引出，GLM-5.3-flash × v3 判官）\n")
    lines.append("本臂与 A×1(T=0) 唯一差别：提示词要求先给 5–10 句简明逐步推理，"
                 "再以恰好一个 JSON 代码块给出 top-5。模型/温度/病例文本/调用次数"
                 "完全相同（qwen3.8-flash、thinking 关闭、T=0、单次调用）。"
                 "故 A×1+CoT − A×1 = 推理引出的纯效应；A×1+CoT vs P/MDT = "
                 "结构（多视角/多智能体）的残余效应。\n")
    lines.append("统计口径与 `stats_caselevel.py` 一致：病例级 5-seed 命中率 → "
                 "配对 Wilcoxon 双侧符号秩；病例级 cluster bootstrap "
                 f"{N_BOOT} 次 95% CI；多数决（>=3/5）精确 McNemar。"
                 f"缺失判定对 {n_missing}（按 miss 计）。\n")

    lines.append("\n## 1. 逐 seed 准确率（CPC 87 例 × 5 seeds）\n")
    lines.append("| 方案 | top-1 | top-3 | top-5 |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        ms = out["mean_sd"][m]
        if ms["top1"]["mean"] is None:
            lines.append(f"| {NAME[m]} | NA | NA | NA |")
            continue
        lines.append(f"| {NAME[m]} | "
                     f"{ms['top1']['mean']*100:.1f}% ± {ms['top1']['sd']*100:.1f} | "
                     f"{ms['top3']['mean']*100:.1f}% ± {ms['top3']['sd']*100:.1f} | "
                     f"{ms['top5']['mean']*100:.1f}% ± {ms['top5']['sd']*100:.1f} |")
    r = out.get("reasoning")
    if r:
        lines.append(f"\n推理产出核对（A×1+CoT，{r['n']} 次调用）："
                     f"reasoning_chars 均值 {r['mean_reasoning_chars']:.0f}"
                     f"（最小 {r['min_reasoning_chars']}，"
                     f"零推理 {r['zero_reasoning_cases']} 例）；"
                     f"total_tokens 均值 {r['mean_total_tokens']:.0f}。"
                     f"对照 A×1 的 max_tokens=2048、本臂 4096。\n")

    lines.append("\n## 2. 病例级配对比较（主口径）\n")
    for name, ids in splits:
        res = out["caselevel"][name]
        lines.append(f"\n### {name}（n={res['n_cases']}）\n")
        lines.append("| top-k | 对比 | 命中率 A vs B | 均值差 [95% CI] | "
                     "Wilcoxon p | 多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p |")
        lines.append("|---|---|---|---|---|---|---|")
        for k in (1, 3, 5):
            for pair, cmp in res["topk"][k]["comparisons"].items():
                a, b = pair.split("_vs_")
                lo, hi = cmp["boot95_ci"]
                maj = cmp["majority"]
                old = cmp["pooled_mcnemar"]
                wp = cmp["wilcoxon_p"]
                lines.append(
                    f"| top-{k} | {NAME[a]} vs {NAME[b]} | "
                    f"{cmp['mean_rate_a']*100:.1f}% vs {cmp['mean_rate_b']*100:.1f}% | "
                    f"{cmp['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{fmt_p(wp)}{sig(wp)} | "
                    f"{maj['a_only']}:{maj['b_only']} p={fmt_p(maj['mcnemar_p'])}"
                    f"{sig(maj['mcnemar_p'])} | "
                    f"{old['a_only']}:{old['b_only']} p={fmt_p(old['p'])}"
                    f"{sig(old['p'])} |")
        cm = res["topk"][5]["case_rate_mean"]
        lines.append("")
        lines.append("病例级 5-seed 平均命中率（top-5）："
                     + " / ".join(f"{NAME[m]} {cm[m]*100:.1f}%"
                                  if cm[m] is not None else f"{NAME[m]} NA"
                                  for m in METHODS))

    lines.append("\n## 3. 与主口径脚本的口径核对\n")
    if check.get("identical") is None:
        lines.append("未找到 stats_caselevel.json，未核对。")
    else:
        lines.append("CPC87 上基础三对（MDT vs Ax1、MDT vs P、P vs Ax1）的 "
                     "mean_diff / Wilcoxon p / bootstrap 95% CI / 多数决 McNemar / "
                     "合并 McNemar 与 `stats_caselevel.py` 输出逐字段比对："
                     + ("**全部一致**（脚本复刻未偏离既有口径）。"
                        if check["identical"] else
                        "**存在不一致**，见控制台输出与 json 的 "
                        "sanity_check_vs_stats_caselevel 字段。"))
    lines.append("\n## 4. 结论口径提示\n")
    lines.append("- A×1+CoT vs A×1：推理引出的净效应（同温度 T=0、同模型、同病例、"
                 "同调用次数）。")
    lines.append("- A×1+CoT vs P / MDT：扣除 elicitation 后剩下的结构效应。"
                 "若 P/MDT 的优势在此比较中消失，则原差异主要由"
                 "「是否被要求推理」解释；若仍显著，则结构/聚合有独立贡献。")
    lines.append("- 逐 seed 表为标准差口径（run-to-run 方差）；第 2 节为主口径"
                 "（病例级 5-seed 均值 + 配对检验），两者不可混用显著性。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def smoke():
    cases = load_merged()[:3]
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    path = SMOKE_DIR / "Ax1cot_s1.jsonl"
    raw_path = SMOKE_DIR / "raw_s1.jsonl"
    for p in (path, raw_path):
        if p.exists():
            p.unlink()
    print(f"[冒烟] {len(cases)} 例 → {SMOKE_DIR}", flush=True)
    ok = True
    for c in cases:
        raw, top5, tokens = infer_one(c["text"])
        rc = reasoning_length(raw)
        row = {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
               "total_tokens": tokens, "reasoning_chars": rc}
        append_row(path, row)
        append_row(raw_path, {"case_id": c["case_id"], "raw": raw})
        parsed = parse_top5(raw)
        good = bool(parsed) and rc > 0 and parsed == top5
        ok &= good
        print(f"\n[冒烟] {c['case_id']}")
        print(f"  gold            : {c['gold'][:90]}")
        print(f"  top5[0]         : {top5[0] if top5 else None}")
        print(f"  top5 解析条数    : {len(top5)} | parse_top5 复算一致: {parsed == top5}")
        print(f"  reasoning_chars : {rc} (>0: {rc > 0})")
        print(f"  total_tokens    : {tokens}")
        print(f"  推理开头        : {raw[:160].replace(chr(10), ' ')}")
        print(flush=True)
    print(f"[冒烟] 全部通过: {ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    if os.environ.get("SMOKE") == "1":
        smoke()
        return
    raw_phase = os.environ.get("PHASE", "").strip().lower()
    if not raw_phase:
        print("未指定 PHASE；本脚本默认不做任何事（推理阶段必须显式指定）。\n"
              "用法：PHASE=infer|judge|analyze|all（可逗号组合）", flush=True)
        return
    phases = ({"infer", "judge", "analyze"} if raw_phase == "all"
              else {p.strip() for p in raw_phase.split(",") if p.strip()})
    bad = phases - {"infer", "judge", "analyze"}
    if bad:
        raise SystemExit(f"未知 PHASE: {sorted(bad)}")

    cases = load_merged()
    print(f"数据集: {len(cases)} 例 | PHASE={sorted(phases)}", flush=True)

    if "infer" in phases:
        COT_OUTDIR.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            run_seed(seed, cases, COT_OUTDIR)
        for seed in SEEDS:
            rows = list(load_done(COT_OUTDIR / f"Ax1cot_s{seed}.jsonl").values())
            if len(rows) != len(cases):
                gap = sorted({c["case_id"] for c in cases}
                             - {r["case_id"] for r in rows})
                print(f"[检查] s{seed} 缺 {len(gap)} 例：{gap[:10]}", flush=True)
            else:
                print(f"[检查] s{seed} 完整 {len(rows)} 例", flush=True)

    if "judge" in phases:
        rows = [r for s in SEEDS
                for r in load_done(COT_OUTDIR / f"Ax1cot_s{s}.jsonl").values()]
        judge_missing(rows)

    if "analyze" in phases:
        run_analyze()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
