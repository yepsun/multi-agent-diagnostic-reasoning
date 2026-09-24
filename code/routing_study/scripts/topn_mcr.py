#!/usr/bin/env python3
"""MedCaseReasoning 外部验证：406 例分层子集 × A×1 / P / MDT（qwen3.8-flash）。

- 病例用数据集自带 case_prompt（呈现段，不含诊断讨论的 text 全文——防泄漏）。
- 提示词 v2 原样冻结使用（纯测试集，零迭代）。
- 判官 deepseek-flash，读写统一缓存 judge_cache_dsflash_unified.json；
  严格判分（空/无法解析不写缓存）。
- 各阶段断点续跑；P/A×1 输出为空时最多重试 2 次（deepseek-flash 格式漂移教训）。
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
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from mdt_cpc import ROLES, GUIDELINES, call_role, parse_top5, key_of  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402

DATA = ROOT / "data" / "medcasereasoning_subset.json"
OUTDIR = ROOT / "routing_study" / "results" / "topn_mcr"
UNIFIED_CACHE = ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"

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


def load_cases():
    rows = json.loads(DATA.read_text())
    return [{"case_id": r["case_id"], "text": r["Q"],
             "gold": r["A"]["final_diagnosis"]} for r in rows]


def call_top5(prompt, temperature):
    """带 2 次空重试的 top-5 调用（返回 None 表示全部尝试为空）。"""
    last = []
    for _ in range(3):
        raw, usage = call_llm(prompt, temperature=temperature, max_tokens=4096,
                              timeout=300, provider="qwen",
                              disable_thinking=True)
        data = extract_json(raw)
        if not isinstance(data, dict):
            data = {}
        items = data.get("top5") or []
        top5 = []
        for it in items[:5]:
            dx = (str(it.get("diagnosis") or "").strip()
                  if isinstance(it, dict) else str(it or "").strip())
            if dx and dx.lower() not in [x.lower() for x in top5]:
                top5.append(dx)
        if top5:
            return top5, (usage or {}).get("total_tokens", 0)
        last = []
    return last, 0


def run_simple(name, path, prompt_fn, temperature, cases):
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{name}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        top5, tokens = call_top5(prompt_fn(c), temperature)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 20 == 0 or i == len(todo):
                print(f"[{name}] {i}/{len(todo)}", flush=True)


def run_mdt(cases):
    roles_path = OUTDIR / "mdt_roles.jsonl"
    synth_path = OUTDIR / "mdt_synth.jsonl"
    all_roles = []
    if roles_path.exists():
        for line in roles_path.read_text().splitlines():
            line = line.strip()
            if line:
                all_roles.append(json.loads(line))
    done_roles = {(r["case_id"], r["role"]) for r in all_roles}
    todo = [(c, title, brief) for c in cases for title, brief in ROLES
            if (c["case_id"], title) not in done_roles]
    print(f"[MDT 角色] 已完成 {len(done_roles)}，待跑 {len(todo)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(call_role, t, b, c["text"]): (c, t)
                for c, t, b in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c, role = futs[fut]
            try:
                top5, tokens = fut.result()
            except Exception as e:
                print(f"[MDT 角色] {c['case_id'][:30]} {role} 失败: {e}",
                      flush=True)
                continue
            append_row(roles_path, {"case_id": c["case_id"], "role": role,
                                    "top5": top5, "total_tokens": tokens})
            if i % 50 == 0 or i == len(todo):
                print(f"[MDT 角色] {i}/{len(todo)}", flush=True)

    done = load_done(synth_path)
    # re-read after the role phase: the file was appended to above
    all_roles = []
    if roles_path.exists():
        for line in roles_path.read_text().splitlines():
            line = line.strip()
            if line:
                all_roles.append(json.loads(line))
    roles_by = {}
    for r in all_roles:
        roles_by.setdefault(r["case_id"], {})[r["role"]] = r["top5"]
    todo = []
    for c in cases:
        if c["case_id"] in done:
            continue
        opinions = roles_by.get(c["case_id"], {})
        if len(opinions) < len(ROLES):
            continue
        lines = []
        for title, _ in ROLES:
            lines.append(f"{title}:")
            for i, item in enumerate(opinions[title][:5], 1):
                line = f"  {i}. {item['diagnosis']}"
                if item.get("rationale"):
                    line += f" — {item['rationale']}"
                lines.append(line)
        todo.append((c, "\n".join(lines)))
    print(f"[MDT 汇总] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c, opinions_text):
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
{opinions_text}
"""
        top5, tokens = call_top5(prompt, 0.3)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(work, c, ops): c for c, ops in todo}
        done_n = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                append_row(synth_path, fut.result())
            except Exception as e:
                print(f"[MDT 汇总] {c['case_id'][:30]} 失败: {e}", flush=True)
            done_n += 1
            if done_n % 20 == 0 or done_n == len(todo):
                print(f"[MDT 汇总] {done_n}/{len(todo)}", flush=True)


def judge_phase(rows):
    cache = json.loads(UNIFIED_CACHE.read_text()) if UNIFIED_CACHE.exists() else {}
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            key = key_of(row["gold"], cand)
            if key not in cache:
                jobs[key] = (row["gold"], cand)
    print(f"[判定] 统一缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
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

        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for key, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(key)
                else:
                    cache[key] = verdict
        unresolved = set(nxt)
        UNIFIED_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定] 完成，未解析 {len(unresolved)}", flush=True)

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
    cases = load_cases()
    print(f"数据集: {len(cases)} 例 MedCaseReasoning 子集 | 模型 qwen3.8-flash | "
          f"判官 deepseek-flash（统一缓存）", flush=True)

    run_simple("A×1", OUTDIR / "ax1.jsonl",
               lambda c: A_TOPN_PROMPT.format(case_text=c["text"]), 0.0, cases)
    run_simple("P", OUTDIR / "p.jsonl",
               lambda c: PERSPECTIVE_PROMPT.format(
                   structured_case=c["text"]) + P_TOPN_SUFFIX, 0.3, cases)
    run_mdt(cases)

    rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
        "MDT": list(load_done(OUTDIR / "mdt_synth.jsonl").values()),
    }
    hits_fn = judge_phase([r for v in rows.values() for r in v])
    summary = {s: summarize(s, v, hits_fn) for s, v in rows.items()}
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
