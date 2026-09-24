#!/usr/bin/env python3
"""Prompt-v2（三处小改）后的 A×1 / P 重跑，与基线对比。

三处改动（英文写入提示词）：
1. 待解诊断不一定是器质性病变：已声明的诊断（含精神科）、入院主因本身就是答案。
2. 显著的器质性发现可能是伴随发现；存在两条诊断线时并列评估、按解释力排序。
3. 允许组合式诊断；感染性候选按病原体分种单列（毛霉 vs 曲霉）。

输出独立目录 topn_ablation_promptv2/；判定复用基线 judge_cache（内容寻址）。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from topn_cpc import (load_dataset, load_done, append_row, call_top5,  # noqa: E402
                      run_judge_phase, MAX_WORKERS)
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_ablation_promptv2"
BASE_SUMMARY = ROOT / "routing_study" / "results" / "topn_ablation" / "summary.json"

GUIDELINES = """Guidelines:
- The diagnosis under question is not necessarily a structural/organic lesion: a diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer to determine.
- A striking organic finding (mass, lesion) may be an incidental companion finding; if two separate diagnostic lines exist, evaluate both and rank each by how well it explains the whole case.
- Combination diagnoses are allowed when they reflect the real process (e.g., "post-influenza bacterial superinfection pneumonia"). For infectious candidates, name specific pathogens as separate items when clinically distinct (e.g., mucormycosis vs aspergillosis)."""

A_TOPN_PROMPT = """You are an expert internist reviewing an MGH CPC case.

Produce a ranked differential diagnosis: the 5 most likely diagnoses, most likely first. Be specific (disease name plus the key qualifier that matters for this case).

""" + GUIDELINES + """

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}

Case:
{case_text}
"""

P_TOPN_SUFFIX = """

""" + GUIDELINES + """

## Output Requirement (final block)
After your five-perspective analysis, conclude with one JSON code block (the only JSON in your answer):
```json
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}
```
Exactly 5 items, ranked most to least likely. Nothing may follow the JSON block."""


def run_scheme(name, path, work_fn, cases):
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{name}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work_fn, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            c, row = fut.result()
            append_row(path, row)
            print(f"[{name}] {i + 1}/{len(todo)} {row['case_id'][:40]}: "
                  f"{row['top5'][:1]}", flush=True)


def summarize(scheme_rows, hits_fn):
    summary = {}
    for scheme, rows in scheme_rows.items():
        n = len(rows)
        stat = {"n": n, "top1": 0, "top3": 0, "top5": 0,
                "gold_at_rank": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                                 "miss": 0}}
        details = []
        for row in rows:
            verdicts = hits_fn(row)
            hit = next((r for r, v in enumerate(verdicts, start=1) if v), None)
            if hit:
                if hit == 1:
                    stat["top1"] += 1
                if hit <= 3:
                    stat["top3"] += 1
                stat["top5"] += 1
                stat["gold_at_rank"][str(hit)] += 1
            else:
                stat["gold_at_rank"]["miss"] += 1
            details.append({"case_id": row["case_id"], "top5": row["top5"],
                            "verdicts": verdicts, "hit": hit})
        for k in (1, 3, 5):
            stat[f"top{k}_acc"] = round(stat[f"top{k}"] / n, 3) if n else None
        summary[scheme] = {**stat, "details": details}
        print(f"{scheme}: n={n} top1 {stat['top1']} ({stat['top1_acc']}) "
              f"top3 {stat['top3']} ({stat['top3_acc']}) "
              f"top5 {stat['top5']} ({stat['top5_acc']}) | {stat['gold_at_rank']}",
              flush=True)
    return summary


def compare_with_baseline(summary):
    base = json.loads(BASE_SUMMARY.read_text())
    for scheme in ("Ax1", "P"):
        bd = {d["case_id"]: d for d in base[scheme]["details"]}
        vd = {d["case_id"]: d for d in summary[scheme]["details"]}
        gain1 = [c for c in vd if vd[c]["hit"] == 1 and bd[c]["hit"] != 1]
        lose1 = [c for c in vd if bd[c]["hit"] == 1 and vd[c]["hit"] != 1]
        gain5 = [c for c in vd if vd[c]["hit"] and not bd[c]["hit"]]
        lose5 = [c for c in vd if bd[c]["hit"] and not vd[c]["hit"]]
        print(f"\n== {scheme} vs 基线 ==")
        print(f"  top-1 变对: {len(gain1)} 例 {gain1}")
        print(f"  top-1 变错: {len(lose1)} 例 {lose1}")
        print(f"  top-5 新命中: {len(gain5)} 例 {gain5}")
        print(f"  top-5 丢失: {len(lose5)} 例 {lose5}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_dataset()

    run_scheme("A×1", OUTDIR / "ax1.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5(A_TOPN_PROMPT.format(
                                             case_text=c["text"]),
                                             temperature=0.0)))}),
               cases)
    run_scheme("P", OUTDIR / "p.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5(
                                             PERSPECTIVE_PROMPT.format(
                                                 structured_case=c["text"])
                                             + P_TOPN_SUFFIX,
                                             temperature=0.3)))}),
               cases)

    scheme_rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
    }
    hits_fn = run_judge_phase(cases, scheme_rows)
    summary = summarize(scheme_rows, hits_fn)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_with_baseline(summary)


if __name__ == "__main__":
    main()
