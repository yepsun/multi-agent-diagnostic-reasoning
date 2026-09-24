#!/usr/bin/env python3
"""ER-Reason 外部验证试点：353 例分层子集 × A×1 / P / MDT（qwen3.8-flash，1 seed）。

- 病例输入：结构化头（年龄/性别/主诉）+ H&P 全文（出院小结/病程/会诊/
  ED 医师笔记不进输入，防金标签泄漏），见 build_erreason_subset.py。
- 提示词 v2 原样冻结使用（纯测试集，零迭代）。
- 判官 deepseek-flash，双口径同时判：v2 规则写统一缓存
  judge_cache_dsflash_unified.json，v3 规则写 judge_cache_dsflash_v3.json；
  严格判分（空/无法解析不写缓存）。
- 各阶段断点续跑；输出 topn_erreason/{ax1,p,mdt_roles,mdt_synth}.jsonl
  与 summary.json（v2/v3 两套指标并排）。
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

import os as _os_er
_os_er.environ.setdefault("OPENROUTER_ER", "1")  # ER 合规路由：本进程的生成调用（call_top5）经 OpenRouter
from run_inference import call_llm, call_llm_judge  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from mdt_cpc import ROLES, call_role, key_of  # noqa: E402
from topn_mcr import JUDGE_PROMPT, call_top5, summarize  # noqa: E402
from judge_v3 import V3_PROMPT  # noqa: E402

DATA = Path(os.environ.get("ER_DATA", ROOT / "data" / "er_reason_subset.json"))
OUTDIR = ROOT / "routing_study" / "results" / os.environ.get(
    "ER_OUTDIR", "topn_erreason")
V2_CACHE = ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"
V3_CACHE = ROOT / "routing_study" / "results" / "judge_cache_dsflash_v3.json"
GLM_CACHE = ROOT / "routing_study" / "results" / "judge_cache_glm_v3.json"


def load_cases():
    return json.loads(DATA.read_text())


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
            if line.strip():
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
    all_roles = []
    if roles_path.exists():
        for line in roles_path.read_text().splitlines():
            if line.strip():
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


def judge_phase(rows, cache_path, prompt, use_v3):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            k = key_of(row["gold"], cand)
            if k not in cache:
                jobs[k] = (row["gold"], cand)
    tag = "v3" if use_v3 else "v2"
    print(f"[判定-{tag}] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
    unresolved = set(jobs.keys())
    for round_no in (1, 2, 3):
        if not unresolved:
            break
        items = jobs.items() if round_no == 1 else [(k, jobs[k]) for k in unresolved]
        nxt = []

        def work(item):
            k, (gold, cand) = item
            try:
                if use_v3 == "glm":
                    from judge_glm_v3_v4check import glm_judge
                    verdict, _, _ = glm_judge(gold, cand)
                    return k, verdict
                if use_v3:
                    raw, _ = call_llm(prompt.format(gold=gold, pred=cand),
                                      temperature=0.0, max_tokens=10,
                                      timeout=60, provider="deepseek-flash",
                                      disable_thinking=True,
                                      er_privacy=False)  # 判分只传诊断串，恒直连
                else:
                    raw, _ = call_llm_judge(prompt.format(gold=gold, pred=cand),
                                            timeout=60, max_retries=2)
                v = (raw or "").strip().upper()
                if v.startswith("YES"):
                    return k, True
                if v.startswith("NO"):
                    return k, False
                return k, None
            except Exception:
                return k, None

        with ThreadPoolExecutor(12 if use_v3 == "glm" else MAX_WORKERS) as ex:
            for k, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(k)
                else:
                    cache[k] = verdict
        unresolved = set(nxt)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定-{tag}] 完成，未解析 {len(unresolved)}", flush=True)

    def hits(row):
        return [bool(cache.get(key_of(row["gold"], c))) for c in row["top5"][:5]]

    return hits


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    print(f"数据集: {len(cases)} 例 ER-Reason 子集 | 模型 qwen3.8-flash | "
          f"判官 deepseek-flash（v2+v3 双口径）", flush=True)

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
    if os.environ.get("SKIP_JUDGE"):
        print("SKIP_JUDGE=1：仅推理，跳过判定（待 GLM 判官就绪后统一判）",
              flush=True)
        return
    all_rows = [r for v in rows.values() for r in v]
    judge_mode = os.environ.get("JUDGE", "dual").lower()
    hits_v2 = judge_phase(all_rows, V2_CACHE, JUDGE_PROMPT, use_v3=False)
    if judge_mode == "glm":
        hits_v3 = judge_phase(all_rows, GLM_CACHE, V3_PROMPT, use_v3="glm")
        tag = "glm-v3"
    else:
        hits_v3 = judge_phase(all_rows, V3_CACHE, V3_PROMPT, use_v3=True)
        tag = "v3"
    summary = {
        "v2": {s: summarize(f"{s} (v2)", v, hits_v2) for s, v in rows.items()},
        tag: {s: summarize(f"{s} ({tag})", v, hits_v3) for s, v in rows.items()},
    }
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTDIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
