#!/usr/bin/env python3
"""87 例 CPC 全流程，推理与判官均用 deepseek-flash（API 的 flash 档，
即用户所称 deepseek-flash；552B）。v2 提示词，A×1 + P。

判分缓存独立（judge_cache_dsflash.json），不与 deepseek-flash 判定混用。
严格判分：空/无法解析的返回不写缓存（留待重试），避免假 False。
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
os.environ["DEEPSEEK_MODEL"] = "deepseek-flash"  # must precede imports

from run_inference import call_llm, call_llm_judge  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_dsflash"
JUDGE_CACHE = ROOT / "routing_study" / "results" / "topn_dsflash" / "judge_cache_dsflash.json"

JUDGE_PROMPT = """You are a medical evaluation judge. A model produced a diagnosis for a clinical case.
Reference (correct) diagnosis: {gold}
Model's diagnosis: {pred}

Judging rules (apply in order):
1. CORRECT (YES) if the model's diagnosis names the same DISEASE as the reference —
   synonyms, abbreviations, and translations are acceptable.
2. The model adding extra findings, complications, etiologies, or secondary diagnoses
   does NOT make it wrong, as long as the reference disease is named as (part of) the
   main diagnosis.
3. Missing qualifiers in the reference (disease stage, severity, "in remission",
   anatomic subtype, etiologic form) do NOT make the model wrong: e.g. "celiac disease"
   is correct for "celiac disease in histologic remission"; "tularemia" is correct for
   "ulceroglandular tularemia"; "acute pulmonary embolism" is correct for "acute massive
   pulmonary embolism with clot in transit".
4. WRONG (NO) only if the model names a DIFFERENT disease as its main diagnosis, or a
   generic category that never specifically names the reference disease.
Answer with exactly one word: YES or NO."""


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def call_top5_ds(prompt, temperature):
    raw, usage = call_llm(prompt, temperature=temperature, max_tokens=2048,
                          timeout=300, provider="deepseek-flash", disable_thinking=True)
    data = extract_json(raw) or {}
    items = data.get("top5") or []
    top5 = []
    for it in items[:5]:
        dx = (it.get("diagnosis") if isinstance(it, dict) else str(it)).strip()
        if dx and dx.lower() not in [x.lower() for x in top5]:
            top5.append(dx)
    return top5, (usage or {}).get("total_tokens", 0)


def run_scheme(name, path, work_fn, cases):
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{name}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work_fn, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c, row = fut.result()
            append_row(path, row)
            if i % 10 == 0 or i == len(todo):
                print(f"[{name}] {i}/{len(todo)}", flush=True)


def judge_strict(gold, cand):
    raw, _ = call_llm_judge(JUDGE_PROMPT.format(gold=gold, pred=cand),
                            timeout=60, max_retries=2)
    v = (raw or "").strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    raise RuntimeError(f"unparseable verdict: {(raw or '')[:40]!r}")


def run_judge(scheme_rows):
    cache = {}
    if JUDGE_CACHE.exists():
        cache = json.loads(JUDGE_CACHE.read_text())
    jobs = {}
    for rows in scheme_rows.values():
        for row in rows:
            for cand in row["top5"][:5]:
                key = key_of(row["gold"], cand)
                if key not in cache:
                    jobs[key] = (row["gold"], cand)
    print(f"[判定] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)

    unresolved = set(jobs.keys())
    for round_no in (1, 2, 3):
        if not unresolved:
            break
        print(f"[判定] 第 {round_no} 轮：{len(unresolved)} 条", flush=True)
        items = jobs.items() if round_no == 1 else [(k, jobs[k]) for k in unresolved]
        nxt = []

        def work(item):
            key, (gold, cand) = item
            try:
                return key, judge_strict(gold, cand)
            except Exception:
                return key, None

        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for key, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(key)
                else:
                    cache[key] = verdict
        unresolved = set(nxt)
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定] 完成，缓存 {len(cache)} 条，未解析 {len(unresolved)}", flush=True)

    def hits(row):
        return [bool(cache.get(key_of(row["gold"], c)))
                for c in row["top5"][:5]]

    return hits


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


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    print(f"数据集: {len(cases)} 例 | 推理+判官: {os.environ['DEEPSEEK_MODEL']}",
          flush=True)

    run_scheme("A×1", OUTDIR / "ax1.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5_ds(A_TOPN_PROMPT.format(
                                             case_text=c["text"]),
                                             temperature=0.0)))}),
               cases)
    run_scheme("P", OUTDIR / "p.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5_ds(
                                             PERSPECTIVE_PROMPT.format(
                                                 structured_case=c["text"])
                                             + P_TOPN_SUFFIX,
                                             temperature=0.3)))}),
               cases)

    scheme_rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
    }
    hits_fn = run_judge(scheme_rows)
    summary = summarize(scheme_rows, hits_fn)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
