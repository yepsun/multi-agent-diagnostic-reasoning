#!/usr/bin/env python3
"""deepseek-flash 矩阵补全：MCR 406 × A×1/P + MDT（CPC 87 + MCR 406）。

判官 deepseek-flash（v2 规则），缓存种子 = qwenmax 对比所用缓存
（同判官同规则，跨模型可比）。断点续跑。
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
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

from run_inference import call_llm, call_llm_judge  # noqa: E402
from topn_cpc import load_done, append_row  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from mdt_cpc import ROLES, call_role  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from topn_mcr import load_cases as load_mcr_cases  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_dsflash_ext"
SEED_JUDGE = ROOT / "routing_study" / "results" / "topn_qwenmax" / "judge_cache_qwenmax_dsflash.json"
JUDGE_CACHE = OUTDIR / "judge_cache_dsflash.json"

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
   anatomic subtype, etiologic form) do NOT make the model wrong.
4. WRONG (NO) only if the model names a genuinely DIFFERENT disease entity as its
   main diagnosis — a distinct entity, not merely a broader category, a component
   of a compound diagnosis, or a neighboring subtype.
Answer with exactly one word: YES or NO."""


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def call_top5_ds(prompt, temperature):
    for _ in range(3):
        raw, usage = call_llm(prompt, temperature=temperature, max_tokens=4096,
                              timeout=300, provider="deepseek-flash",
                              disable_thinking=True)
        data = extract_json(raw) or {}
        items = data.get("top5") or []
        top5 = []
        for it in items[:5]:
            dx = (str(it.get("diagnosis") or "").strip()
                  if isinstance(it, dict) else str(it or "").strip())
            if dx and dx.lower() not in [x.lower() for x in top5]:
                top5.append(dx)
        if top5:
            return top5, (usage or {}).get("total_tokens", 0)
    return [], 0


def run_simple(name, path, prompt_fn, temperature, cases):
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{name}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        top5, tokens = call_top5_ds(prompt_fn(c), temperature)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 25 == 0 or i == len(todo):
                print(f"[{name}] {i}/{len(todo)}", flush=True)


def run_mdt(name, cases):
    roles_path = OUTDIR / f"mdt_roles_{name}.jsonl"
    synth_path = OUTDIR / f"mdt_synth_{name}.jsonl"
    all_roles = []
    if roles_path.exists():
        for line in roles_path.read_text().splitlines():
            line = line.strip()
            if line:
                all_roles.append(json.loads(line))
    done_roles = {(r["case_id"], r["role"]) for r in all_roles}
    roles_by = {}
    for r in all_roles:
        roles_by.setdefault(r["case_id"], {})[r["role"]] = r["top5"]
    todo = [(c, t, b) for c in cases for t, b in ROLES
            if (c["case_id"], t) not in done_roles]
    print(f"[{name} MDT 角色] 已完成 {len(done_roles)}，待跑 {len(todo)}",
          flush=True)

    def work_role(c, t, b):
        # 家族隔离：deepseek-flash 臂的角色意见必须由 deepseek-flash 生成
        return c, t, call_role(t, b, c["text"], 0.3,
                               provider="deepseek-flash", disable_thinking=True)

    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(work_role, c, t, b): (c, t) for c, t, b in todo}
        done_n = 0
        for fut in as_completed(futs):
            c, t = futs[fut]
            try:
                c, t, (top5, tokens) = fut.result()
            except Exception as e:
                print(f"[{name} MDT 角色] {c['case_id'][:30]} {t} 失败: {e}",
                      flush=True)
                continue
            append_row(roles_path, {"case_id": c["case_id"], "role": t,
                                    "top5": top5, "total_tokens": tokens})
            roles_by.setdefault(c["case_id"], {})[t] = top5
            done_n += 1
            if done_n % 50 == 0 or done_n == len(todo):
                print(f"[{name} MDT 角色] {done_n}/{len(todo)}", flush=True)

    done = load_done(synth_path)
    todo = [c for c in cases
            if c["case_id"] not in done
            and len(roles_by.get(c["case_id"], {})) == len(ROLES)]
    print(f"[{name} MDT 汇总] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work_synth(c):
        lines = []
        for title, _ in ROLES:
            lines.append(f"{title}:")
            for i, item in enumerate(roles_by[c["case_id"]][title][:5], 1):
                line = f"  {i}. {item['diagnosis']}"
                if item.get("rationale"):
                    line += f" — {item['rationale']}"
                lines.append(line)
        prompt = f"""You are the moderator of an MDT panel. Five specialists independently reviewed the case below, each through their own lens, without seeing each other's opinions. Their ranked candidate lists are given.

Integrate them into the final ranked top-5 for the case:
- Candidates supported by multiple specialists generally rise.
- A unique candidate with specific, case-grounded support must NOT be dropped merely because only one specialist listed it.
- Resolve conflicts by re-checking against the case text.
- Combination diagnoses are allowed.

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, ... exactly 5 items]}}

Case:
{c['text']}

Panel opinions:
{chr(10).join(lines)}
"""
        top5, tokens = call_top5_ds(prompt, 0.3)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(work_synth, c): c for c in todo}
        done_n = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                append_row(synth_path, fut.result())
            except Exception as e:
                print(f"[{name} MDT 汇总] {c['case_id'][:30]} 失败: {e}",
                      flush=True)
            done_n += 1
            if done_n % 20 == 0 or done_n == len(todo):
                print(f"[{name} MDT 汇总] {done_n}/{len(todo)}", flush=True)


def judge_phase(rows):
    cache = json.loads(SEED_JUDGE.read_text()) if SEED_JUDGE.exists() else {}
    if JUDGE_CACHE.exists():
        cache.update(json.loads(JUDGE_CACHE.read_text()))
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
                raw, _ = call_llm_judge(JUDGE_PROMPT.format(gold=gold, pred=cand),
                                        timeout=60, max_retries=2)
                v = (raw or "").strip().upper()
                if v.startswith("YES"):
                    return key, True
                if v.startswith("NO"):
                    return key, False
                return key, None
            except Exception:
                return key, None

        with ThreadPoolExecutor(4) as ex:
            for key, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(key)
                else:
                    cache[key] = verdict
        unresolved = set(nxt)
        JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定] 完成，缓存 {len(cache)}，未解析 {len(unresolved)}", flush=True)

    def hits(row):
        return [bool(cache.get(key_of(row["gold"], c))) for c in row["top5"][:5]]

    return hits


def summarize(tag, rows, hits_fn):
    n = len(rows)
    stat = {"n": n, "top1": 0, "top3": 0, "top5": 0,
            "gold_at_rank": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "miss": 0}}
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
    for k in (1, 3, 5):
        stat[f"top{k}_acc"] = round(stat[f"top{k}"] / n, 3) if n else None
    print(f"{tag}: n={n} top1 {stat['top1']} ({stat['top1_acc']}) "
          f"top3 {stat['top3']} ({stat['top3_acc']}) "
          f"top5 {stat['top5']} ({stat['top5_acc']}) | {stat['gold_at_rank']}",
          flush=True)
    return stat


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cpc = load_merged()
    mcr = load_mcr_cases()
    print(f"数据集: CPC {len(cpc)} + MCR {len(mcr)} | 推理 deepseek-flash | "
          f"判官 deepseek-flash", flush=True)

    run_simple("A×1 mcr", OUTDIR / "ax1_mcr.jsonl",
               lambda c: A_TOPN_PROMPT.format(case_text=c["text"]), 0.0, mcr)
    run_simple("P mcr", OUTDIR / "p_mcr.jsonl",
               lambda c: PERSPECTIVE_PROMPT.format(
                   structured_case=c["text"]) + P_TOPN_SUFFIX, 0.3, mcr)
    run_mdt("mcr", mcr)
    run_mdt("cpc", cpc)

    groups = {
        "cpc": {
            "Ax1": list(load_done(ROOT / "routing_study" / "results"
                                  / "topn_dsflash" / "ax1.jsonl").values()),
            "P": list(load_done(ROOT / "routing_study" / "results"
                                / "topn_dsflash" / "p.jsonl").values()),
            "MDT": list(load_done(OUTDIR / "mdt_synth_cpc.jsonl").values()),
        },
        "mcr": {
            "Ax1": list(load_done(OUTDIR / "ax1_mcr.jsonl").values()),
            "P": list(load_done(OUTDIR / "p_mcr.jsonl").values()),
            "MDT": list(load_done(OUTDIR / "mdt_synth_mcr.jsonl").values()),
        },
    }
    all_rows = [r for g in groups.values() for rows in g.values() for r in rows]
    hits_fn = judge_phase(all_rows)

    summary = {}
    for tag, schemes in groups.items():
        for scheme, rows in schemes.items():
            summary[f"DS_{scheme}_{tag}"] = summarize(
                f"DS {scheme} {tag}", rows, hits_fn)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
