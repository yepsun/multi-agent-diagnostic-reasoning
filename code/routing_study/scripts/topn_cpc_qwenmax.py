#!/usr/bin/env python3
"""87 例 CPC，Qwen3.8-Max（2400B）跑 A×1 / P（v2 提示词，no-think 协议）。

判官固定 deepseek-flash：独立缓存（种子=dsflash 运行的判定，保证判官一致）。
同时把 qwen3.8-flash 的 87 例运行（topn_ablation_promptv2_87）用同一判官
重判一遍，使三模型对比的判官完全一致。最后输出三模型并排表。
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
os.environ["QWEN_MODEL"] = "qwen3.8-max"  # must precede imports

from run_inference import call_llm  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_qwenmax"
QFLASH_DIR = ROOT / "routing_study" / "results" / "topn_ablation_promptv2_87"
DSFLASH_CACHE = ROOT / "routing_study" / "results" / "topn_dsflash" / "judge_cache_dsflash.json"
JUDGE_CACHE_QM = OUTDIR / "judge_cache_qwenmax_dsflash.json"
JUDGE_CACHE_QF = OUTDIR / "judge_cache_qwenflash_dsflash.json"

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


def call_top5_qm(prompt, temperature):
    raw, usage = call_llm(prompt, temperature=temperature, max_tokens=4096,
                          timeout=600, provider="qwen", disable_thinking=True)
    data = extract_json(raw) or {}
    items = data.get("top5") or []
    top5 = []
    for it in items[:5]:
        dx = (str(it.get("diagnosis") or "").strip()
              if isinstance(it, dict) else str(it or "").strip())
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


def judge_rows(rows, cache_path):
    """deepseek-flash 判官；缓存种子 = dsflash 运行的判定（判官一致）。"""
    cache = {}
    if DSFLASH_CACHE.exists():
        cache.update(json.loads(DSFLASH_CACHE.read_text()))
    if cache_path.exists():
        cache.update(json.loads(cache_path.read_text()))
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            key = key_of(row["gold"], cand)
            if key not in cache:
                jobs[key] = (row["gold"], cand)
    print(f"[判定] 种子缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
    unresolved = set(jobs.keys())
    for round_no in (1, 2, 3):
        if not unresolved:
            break
        items = jobs.items() if round_no == 1 else [(k, jobs[k]) for k in unresolved]
        nxt = []

        def work(item):
            key, (gold, cand) = item
            try:
                raw, _ = call_llm(JUDGE_PROMPT.format(gold=gold, pred=cand),
                                  temperature=0.0, max_tokens=10, timeout=60,
                                  provider="deepseek-flash", disable_thinking=True)
                v = (raw or "").strip().upper()
                if v.startswith("YES"):
                    return key, True
                if v.startswith("NO"):
                    return key, False
                return key, None
            except Exception:
                return key, None

        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for key, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(key)
                else:
                    cache[key] = verdict
        unresolved = set(nxt)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定] 缓存写入 {cache_path.name}（{len(cache)} 条，未解析 {len(unresolved)}）",
          flush=True)

    def hits(row):
        return [bool(cache.get(key_of(row["gold"], c))) for c in row["top5"][:5]]

    return hits


def summarize(tag, rows, hits_fn):
    n = len(rows)
    stat = {"n": n, "top1": 0, "top3": 0, "top5": 0,
            "gold_at_rank": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "miss": 0}}
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
    print(f"{tag}: n={n} top1 {stat['top1']} ({stat['top1_acc']}) "
          f"top3 {stat['top3']} ({stat['top3_acc']}) "
          f"top5 {stat['top5']} ({stat['top5_acc']}) | {stat['gold_at_rank']}",
          flush=True)
    return {**stat, "details": details}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    print(f"数据集: {len(cases)} 例 | 推理: {os.environ['QWEN_MODEL']} | "
          f"判官: deepseek-flash", flush=True)

    run_scheme("A×1", OUTDIR / "ax1.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5_qm(A_TOPN_PROMPT.format(
                                             case_text=c["text"]),
                                             temperature=0.0)))}),
               cases)
    run_scheme("P", OUTDIR / "p.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5_qm(
                                             PERSPECTIVE_PROMPT.format(
                                                 structured_case=c["text"])
                                             + P_TOPN_SUFFIX,
                                             temperature=0.3)))}),
               cases)

    qm_rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
    }
    qf_rows = {
        "Ax1": list(load_done(QFLASH_DIR / "ax1.jsonl").values()),
        "P": list(load_done(QFLASH_DIR / "p.jsonl").values()),
    }

    print("\n===== Qwen3.8-Max（2400B）=====", flush=True)
    hits_qm = judge_rows(qm_rows["Ax1"] + qm_rows["P"], JUDGE_CACHE_QM)
    sm_qm = {s: summarize(f"Max {s}", qm_rows[s], hits_qm) for s in qm_rows}

    print("\n===== Qwen3.8-Flash（176B，改用 deepseek-flash 判官重判）=====", flush=True)
    hits_qf = judge_rows(qf_rows["Ax1"] + qf_rows["P"], JUDGE_CACHE_QF)
    sm_qf = {s: summarize(f"Flash {s}", qf_rows[s], hits_qf) for s in qf_rows}

    (OUTDIR / "summary.json").write_text(json.dumps(
        {"qwen3.8-max": sm_qm, "qwen3.8-flash_dsflash_judge": sm_qf},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 三模型对比（判官统一 deepseek-flash；ds-flash 判定来自其自身运行）=====",
          flush=True)
    ds = json.loads((ROOT / "routing_study" / "results" / "topn_dsflash"
                     / "summary.json").read_text())
    for scheme in ("Ax1", "P"):
        print(f"-- {scheme} --")
        for tag, sm in (("qwen3.8-max(2400B)", sm_qm),
                        ("qwen3.8-flash(176B)", sm_qf),
                        ("deepseek-flash(552B)", ds[scheme])):
            print(f"  {tag:24s} top1 {sm['top1_acc']:.1%}  top3 "
                  f"{sm['top3_acc']:.1%}  top5 {sm['top5_acc']:.1%}")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
