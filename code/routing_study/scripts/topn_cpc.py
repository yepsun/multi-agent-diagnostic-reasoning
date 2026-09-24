#!/usr/bin/env python3
"""41 例 MGH CPC：A×1 / A×5 / P 的 top-5 诊断运行 + top-1/3/5 语义判分。

推理：DashScope qwen3.8-flash（disable_thinking）。A×1 temp=0；A×5 每例 5 份
temp=0.7 的 top-5，按 Borda 聚合成总排序；P 单次多视角。
判分：v2 语义 judge（DeepSeek），逐位（rank1-5）判定 gold 是否命中，
(gold, candidate) 去重并持久化缓存。支持断点续跑。

输出：routing_study/results/topn_ablation/{ax1,p}.jsonl、ax5_samples.jsonl、
summary.json。
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

from run_inference import call_llm  # noqa: E402
from run_static_routing import _gold_text, judge  # noqa: E402
from case_extraction import format_structured_case, preprocess_case_text  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402
from webapp.clustering import same_disease  # noqa: E402

DATASET = ROOT / "data" / "mgh_qa_dataset.json"
OUTDIR = ROOT / "routing_study" / "results" / "topn_ablation"
JUDGE_CACHE = OUTDIR / "judge_cache.json"
N_CASES = 41
A5_SAMPLES = 5
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
TIMEOUT = 300

A_TOPN_PROMPT = """You are an expert internist reviewing an MGH CPC case.

Produce a ranked differential diagnosis: the 5 most likely diagnoses, most likely first. Be specific (disease name plus the key qualifier that matters for this case).

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}

Case:
{case_text}
"""

P_TOPN_SUFFIX = """

## Output Requirement (final block)
After your five-perspective analysis, conclude with one JSON code block (the only JSON in your answer):
```json
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}
```
Exactly 5 items, ranked most to least likely. Nothing may follow the JSON block."""


def load_dataset():
    cases = json.loads(DATASET.read_text())[:N_CASES]
    out = []
    for c in cases:
        text = format_structured_case(preprocess_case_text(c.get("Q", "")))
        out.append({"case_id": c.get("case_id"), "text": text,
                    "gold": _gold_text(c)})
    return out


def load_done(path):
    done = {}
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["case_id"]] = row
    return done


def append_row(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_top5(raw):
    data = extract_json(raw) or {}
    items = data.get("top5") or []
    top5 = []
    for it in items[:5]:
        dx = (it.get("diagnosis") if isinstance(it, dict) else str(it)).strip()
        if dx and dx.lower() not in [x.lower() for x in top5]:
            top5.append(dx)
    return top5


def call_top5(prompt, temperature):
    raw, usage = call_llm(prompt, temperature=temperature, max_tokens=2048,
                          timeout=TIMEOUT, provider="qwen", disable_thinking=True)
    return parse_top5(raw), (usage or {}).get("total_tokens", 0)


def aggregate_top5(sample_lists):
    """Borda 聚合：rank1 权重 1.0 递减到 rank5 0.2，同病簇合并。"""
    clusters = []
    for lst in sample_lists:
        for rank, dx in enumerate(lst[:5], start=1):
            w = (6 - rank) / 5.0
            placed = False
            for c in clusters:
                if same_disease(c["rep"], dx):
                    c["score"] += w
                    c["names"].append(dx)
                    c["primaries"] += 1 if rank == 1 else 0
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": dx, "names": [dx], "score": w,
                                 "primaries": 1 if rank == 1 else 0})
    out = []
    for c in sorted(clusters, key=lambda c: (c["score"], c["primaries"]),
                    reverse=True)[:5]:
        counts = {}
        for n in c["names"]:
            counts[n] = counts.get(n, 0) + 1
        out.append(max(counts.items(), key=lambda kv: kv[1])[0])
    return out


def run_inference_phase(cases):
    # ---- A×1 ----
    p_ax1 = OUTDIR / "ax1.jsonl"
    done = load_done(p_ax1)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[A×1] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work_ax1(c):
        top5, tokens = call_top5(
            A_TOPN_PROMPT.format(case_text=c["text"]), temperature=0.0)
        return c, top5, tokens

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for i, fut in enumerate([ex.submit(work_ax1, c) for c in todo]):
            c, top5, tokens = fut.result()
            append_row(p_ax1, {"case_id": c["case_id"], "gold": c["gold"],
                               "top5": top5, "total_tokens": tokens})
            print(f"[A×1] {i + 1}/{len(todo)} {c['case_id']}: {top5[:1]}",
                  flush=True)

    # ---- P ----
    from scheme_perspective import PERSPECTIVE_PROMPT
    p_p = OUTDIR / "p.jsonl"
    done = load_done(p_p)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[P] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work_p(c):
        prompt = PERSPECTIVE_PROMPT.format(structured_case=c["text"]) + P_TOPN_SUFFIX
        top5, tokens = call_top5(prompt, temperature=0.3)
        return c, top5, tokens

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for i, fut in enumerate([ex.submit(work_p, c) for c in todo]):
            c, top5, tokens = fut.result()
            append_row(p_p, {"case_id": c["case_id"], "gold": c["gold"],
                             "top5": top5, "total_tokens": tokens})
            print(f"[P] {i + 1}/{len(todo)} {c['case_id']}: {top5[:1]}",
                  flush=True)

    # ---- A×5（每例 5 份，例内串行、例间并行）----
    p_ax5s = OUTDIR / "ax5_samples.jsonl"
    done = load_done(p_ax5s)
    todo = [c for c in cases if len(done.get(c["case_id"], {}).get("samples", [])) < A5_SAMPLES]
    print(f"[A×5] 已完成 {len([c for c in cases if len(done.get(c['case_id'], {}).get('samples', [])) >= A5_SAMPLES])}，待跑 {len(todo)}", flush=True)

    def work_ax5(c):
        samples = list(done.get(c["case_id"], {}).get("samples", []))
        tokens_total = 0
        while len(samples) < A5_SAMPLES:
            top5, tokens = call_top5(
                A_TOPN_PROMPT.format(case_text=c["text"]),
                temperature=0.7)
            samples.append(top5)
            tokens_total += tokens
        return c, samples, tokens_total

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work_ax5, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            c = futs[fut]
            try:
                c_, samples, tokens = fut.result()
            except Exception as e:
                print(f"[A×5] {c['case_id']} 失败: {e}", flush=True)
                continue
            append_row(p_ax5s, {"case_id": c_["case_id"], "gold": c_["gold"],
                                "samples": samples, "total_tokens": tokens})
            print(f"[A×5] {i + 1}/{len(todo)} {c_['case_id']}: "
                  f"{len(samples)} 份完成", flush=True)


def build_ax5_rows(cases):
    done = load_done(OUTDIR / "ax5_samples.jsonl")
    rows = []
    for c in cases:
        d = done.get(c["case_id"])
        if not d or len(d.get("samples", [])) < A5_SAMPLES:
            continue
        rows.append({"case_id": c["case_id"], "gold": c["gold"],
                     "top5": aggregate_top5(d["samples"]),
                     "samples": d["samples"],
                     "total_tokens": d.get("total_tokens", 0)})
    return rows


def run_judge_phase(cases, scheme_rows):
    cache = {}
    if JUDGE_CACHE.exists():
        cache = json.loads(JUDGE_CACHE.read_text())
    jobs = {}
    for scheme, rows in scheme_rows.items():
        for row in rows:
            gold = row["gold"]
            for cand in row["top5"][:5]:
                key = f"{gold[:150]}||{cand[:150]}"
                if key not in cache:
                    jobs[key] = (gold, cand)
    print(f"[判定] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)

    def work(item):
        key, (gold, cand) = item
        try:
            return key, bool(judge(gold, cand))
        except Exception:
            return key, False

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for i, (key, verdict) in enumerate(ex.map(work, jobs.items()), 1):
            cache[key] = verdict
            if i % 40 == 0:
                print(f"[判定] {i}/{len(jobs)}", flush=True)
                JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))

    def hits(row):
        gold = row["gold"]
        out = []
        for cand in row["top5"][:5]:
            out.append(bool(cache.get(f"{gold[:150]}||{cand[:150]}")))
        return out

    return hits


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_dataset()
    print(f"数据集: {len(cases)} 例", flush=True)

    run_inference_phase(cases)

    scheme_rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "Ax5": build_ax5_rows(cases),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
    }

    hits_fn = run_judge_phase(cases, scheme_rows)

    summary = {}
    for scheme, rows in scheme_rows.items():
        n = len(rows)
        stat = {"n": n, "top1": 0, "top3": 0, "top5": 0,
                "gold_at_rank": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                                 "miss": 0},
                "total_tokens": 0}
        details = []
        for row in rows:
            stat["total_tokens"] += row.get("total_tokens", 0) or 0
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
        summary[scheme] = {**{k: v for k, v in stat.items() if k != "details"},
                           "details": details}
        print(f"{scheme}: n={n} top1 {stat['top1']} ({stat['top1_acc']}) "
              f"top3 {stat['top3']} ({stat['top3_acc']}) "
              f"top5 {stat['top5']} ({stat['top5_acc']}) | 位置 {stat['gold_at_rank']}",
              flush=True)

    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
