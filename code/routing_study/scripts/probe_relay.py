#!/usr/bin/env python3
"""封闭旗舰探针（中转站）：gpt-5.5 + gpt-4.1 × CPC 87 例 × 1 seed × 3 臂。

目的：成本/吞吐实测 + A×1 vs P vs MDT 方向预览。非论文数据。
协议与主实验逐字对齐（提示词、温度、输出 schema、解析含 P 的双花括号兜底）；
判定仍走冻结 GLM×v3 共享缓存（flock 保护）。
环境变量：RELAY_KEY（必填）、RELAY_BASE（默认 https://api.openai.com/v1 /* scrubbed: set OPENAI_API_BASE */"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402
from topn_cpc import parse_top5, load_dataset  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from seeds_87_dsflash import parse_top5_repair  # noqa: E402

BASE = os.environ.get("RELAY_BASE", "https://api.openai.com/v1 /* scrubbed: set OPENAI_API_BASE */")
KEY = os.environ.get("RELAY_KEY", "")
WORKERS = int(os.environ.get("PROBE_WORKERS", "3"))
MODELS = os.environ.get("PROBE_MODELS", "gpt-5.5,gpt-4.1").split(",")
OUT = ROOT / "routing_study" / "results" / "probe_relay"
LOG = ROOT / "routing_study" / "results" / "probe_relay.log"

ROLES = [
    ("Attending Internist",
     "overall clinical synthesis: which unifying diagnosis best explains the whole case, and which features are most discriminating"),
    ("Pathophysiologist",
     "underlying mechanism: what process could produce this constellation of findings, and hallmark histologic, laboratory, or molecular clues"),
    ("Imaging & Laboratory Specialist",
     "objective data: how to interpret the imaging, laboratory, and vital-sign patterns, and which patterns or paradoxes stand out"),
    ("Epidemiologist",
     "demographics, geography, travel, exposures, medications, occupation, and comorbidities — hidden risk factors in the narrative"),
    ("Skeptic / Challenger",
     "the strongest argument AGAINST the leading hypothesis: alternatives that explain more findings with fewer contradictions, and findings that remain unexplained"),
]
GUIDELINES = """Guidelines:
- A diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer.
- A striking organic finding may be an incidental companion finding.
- Combination diagnoses are allowed. For infections, name specific pathogens as separate items when clinically distinct."""


def log(msg):
    line = f"[probe] {time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def relay_call(model, prompt, temperature, max_tokens=4096, timeout=240):
    """中转调用：429/5xx/超时退避重试，返回 (top5, total_tokens)。"""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            raw = d["choices"][0]["message"].get("content") or ""
            tokens = (d.get("usage") or {}).get("total_tokens", 0)
            return raw, tokens
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(8 * (attempt + 1), 40))
    raise RuntimeError(f"relay 重试耗尽: {type(last).__name__} {str(last)[:120]}")


def parse_arm(arm, raw):
    """A×1/MDT 用 parse_top5；P 用双花括号兜底解析。返回 top5 list[str]。"""
    if arm == "P":
        top5, _ = parse_top5_repair(raw)
        return top5
    data = parse_top5(raw)
    return data


def gen_arm(model, arm, cases):
    """生成单臂单模型。返回 dict rows + 累计 tokens。"""
    outdir = OUT / model
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{arm}_s1.jsonl"
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["case_id"]] = r
    todo = [c for c in cases if c["case_id"] not in done]
    log(f"{model}/{arm}: 已完成 {len(done)}，待跑 {len(todo)}")
    tokens_acc = sum(r.get("total_tokens", 0) for r in done.values())

    def work(c):
        text = c["text"]
        if arm == "Ax1":
            prompt = A_TOPN_PROMPT.format(case_text=text)
            temp = 0.0
        elif arm == "P":
            prompt = PERSPECTIVE_PROMPT.format(structured_case=text) + P_TOPN_SUFFIX
            temp = 0.3
        else:  # Mdt：先 5 角色，再主持人
            roles_path = outdir / "MdtRoles_s1.jsonl"
            with open(roles_path, "a", encoding="utf-8") as f:
                opinions = []
                for title, brief in ROLES:
                    rp = (f"You are the {title} on an MDT panel reviewing an MGH CPC case.\n\n"
                          f"Your assigned lens: {brief}\n\n"
                          "Through this lens, list the 5 candidate diagnoses that best fit the case, "
                          "ranked most to least likely FROM YOUR PERSPECTIVE. Include candidates another "
                          "specialist might overlook if the findings support them.\n\n"
                          f"{GUIDELINES}\n\n"
                          'Respond with ONLY a JSON object, no other text:\n'
                          '{"top5": [{"rank": 1, "diagnosis": "...", "rationale": "one line through your lens"}, ... exactly 5 items]}\n\n'
                          f"Case:\n{text}\n")
                    raw, tk = relay_call(model, rp, 0.3)
                    top5 = parse_top5(raw)
                    tokens_acc_local = tk
                    opinion = "\n".join(f"{i+1}. {x.get('diagnosis') if isinstance(x, dict) else x}"
                                        for i, x in enumerate(top5[:5]))
                    opinions.append(f"{title}: {opinion}")
                    f.write(json.dumps({"case_id": c["case_id"], "role": title,
                                        "top5": top5}, ensure_ascii=False) + "\n")
            with open(roles_path, encoding="utf-8") as f:
                by_case = {}
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        by_case.setdefault(r["case_id"], []).append(r["role"])
            opinions = []
            for title, _ in ROLES:
                pass
        return None

    # Mdt 分支较复杂，单独实现（见下）
    if arm == "Mdt":
        return gen_mdt(model, cases, outdir, tokens_acc)

    results = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(relay_call, model, prompt_builder(arm, c), temp_of(arm)): c
                for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                raw, tk = fut.result()
                top5 = parse_arm(arm, raw)
                tokens_acc += tk
                if len(top5) == 5:
                    row = {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                           "total_tokens": tk}
                    results[c["case_id"]] = row
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                log(f"  [失败] {model}/{arm} {c['case_id'][:36]}: {e}")
            if i % 25 == 0 or i == len(todo):
                log(f"  {model}/{arm} {i}/{len(todo)}")
    return {**done, **results}, tokens_acc


def prompt_builder(arm, c):
    if arm == "Ax1":
        return A_TOPN_PROMPT.format(case_text=c["text"])
    return PERSPECTIVE_PROMPT.format(structured_case=c["text"]) + P_TOPN_SUFFIX


def temp_of(arm):
    return 0.0 if arm == "Ax1" else 0.3


def gen_mdt(model, cases, outdir, tokens_acc):
    roles_path = outdir / "MdtRoles_s1.jsonl"
    have_roles = set()
    if roles_path.exists():
        for line in open(roles_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                have_roles.add((r["case_id"], r["role"]))
    todo_roles = [(c, t, b) for c in cases for (t, b) in ROLES
                  if (c["case_id"], t) not in have_roles]
    log(f"  MdtRoles: 已有 {len(have_roles)}，待跑 {len(todo_roles)}")
    role_tk = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(relay_call, model, role_prompt(t, b, c["text"]), 0.3): (c, t)
                for c, t, b in todo_roles}
        for i, fut in enumerate(as_completed(futs), 1):
            c, t = futs[fut]
            try:
                raw, _tk = fut.result()
                role_tk.append(_tk)
                top5 = parse_top5(raw)
                with open(roles_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"case_id": c["case_id"], "role": t,
                                        "top5": top5}, ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                log(f"  [失败] role {c['case_id'][:30]}/{t}: {e}")
            if i % 50 == 0 or i == len(todo_roles):
                log(f"  MdtRoles {i}/{len(todo_roles)}")

    roles_by_case = {}
    for line in open(roles_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            roles_by_case.setdefault(r["case_id"], {})[r["role"]] = r["top5"]

    path = outdir / "Mdt_s1.jsonl"
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["case_id"]] = r
    todo = [c for c in cases
            if c["case_id"] in roles_by_case
            and len(roles_by_case[c["case_id"]]) == 5
            and c["case_id"] not in done]
    tokens_acc += sum(role_tk)
    log(f"  Mdt 主持人: 已完成 {len(done)}，待跑 {len(todo)}")
    results = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(relay_call, model, mod_prompt(c["text"], roles_by_case[c["case_id"]]), 0.3): c
                for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                raw, tk = fut.result()
                tokens_acc += tk
                top5 = parse_top5(raw)
                if len(top5) == 5:
                    row = {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                           "total_tokens": tk}
                    results[c["case_id"]] = row
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                log(f"  [失败] Mdt {c['case_id'][:36]}: {e}")
            if i % 25 == 0 or i == len(todo):
                log(f"  Mdt {i}/{len(todo)}")
    return {**done, **results}, tokens_acc


def role_prompt(title, brief, text):
    return (f"You are the {title} on an MDT panel reviewing an MGH CPC case.\n\n"
            f"Your assigned lens: {brief}\n\n"
            "Through this lens, list the 5 candidate diagnoses that best fit the case, "
            "ranked most to least likely FROM YOUR PERSPECTIVE. Include candidates another "
            "specialist might overlook if the findings support them.\n\n"
            f"{GUIDELINES}\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"top5": [{"rank": 1, "diagnosis": "...", "rationale": "one line through your lens"}, ... exactly 5 items]}\n\n'
            f"Case:\n{text}\n")


def mod_prompt(text, roles_map):
    opinions = "\n".join(
        f"{title}: " + "\n".join(f"  {i+1}. {(x.get('diagnosis') if isinstance(x, dict) else x)}"
                                 for i, x in enumerate(roles_map.get(title, [])[:5]))
        for title, _ in ROLES)
    return (f"You are the moderator of an MDT panel. Five specialists independently reviewed "
            f"the MGH CPC case below, each through their own lens, without seeing each other's "
            f"opinions. Their ranked candidate lists are given.\n\n"
            "Integrate them into the final ranked top-5 for the case:\n"
            "- Candidates supported by multiple specialists generally rise.\n"
            "- A unique candidate with specific, case-grounded support must NOT be dropped "
            "merely because only one specialist listed it.\n"
            "- Resolve conflicts by re-checking against the case text.\n"
            "- Combination diagnoses are allowed.\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"top5": [{"rank": 1, "diagnosis": "..."}}, ... exactly 5 items]}\n\n'
            f"Case:\n{text}\n\n"
            f"Panel opinions:\n{opinions}\n")


PROBE_VERDICTS = OUT / "probe_verdicts.json"


def _relay_verdict(gold, cand):
    from judge_v3 import V3_PROMPT
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    req = urllib.request.Request(f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            v = (d["choices"][0]["message"].get("content") or "").strip().upper()
            if v.startswith("YES"):
                return True
            if v.startswith("NO"):
                return False
            return None
        except Exception:  # noqa: BLE001
            time.sleep(4)
    return None


def relay_judge_rows(rows, workers=6):
    """中转判定，结果存独立文件（不污染冻结缓存）。"""
    store = {}
    if PROBE_VERDICTS.exists():
        store = json.loads(PROBE_VERDICTS.read_text())
    todo = {}
    for r in rows:
        for c in r["top5"][:5]:
            k = cs.key_of(r["gold"], c)
            if k not in store:
                todo[k] = (r["gold"], c)
    log(f"  探针判定：待中转判定 {len(todo)} 对（缓存 {len(store)}）")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_relay_verdict, g, c): k for k, (g, c) in todo.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                store[k] = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"  [判定失败] {k[:50]}: {e}")
            if i % 200 == 0 or i == len(todo):
                PROBE_VERDICTS.write_text(json.dumps(store, ensure_ascii=False))
                log(f"  探针判定 {i}/{len(todo)}")
    PROBE_VERDICTS.write_text(json.dumps(store, ensure_ascii=False))
    return store


def main():
    cases = load_merged()
    log(f"探针启动：{MODELS} × 3 臂 × {len(cases)} 例 × seed 1")
    summary = {}
    for model in MODELS:
        summary[model] = {}
        tokens_model = 0
        for arm in ("Ax1", "P", "Mdt"):
            rows, tk = {}, 0
            for attempt in range(5):
                rows, tk = gen_arm(model, arm, cases)
                tokens_model += tk
                if len(rows) >= len(cases):
                    break
                log(f"  {model}/{arm} 第 {attempt+1} 遍后 {len(rows)}/{len(cases)}，休 90s 补采")
                time.sleep(90)
            tokens_model += tk
            # 判定：走中转 GLM（探针内部决策用，不写入冻结缓存；
            # 与直连判定一致率 94.1%，见 relay_judge_validation.py）
            rlist = list(rows.values())
            cache = relay_judge_rows(rlist)
            hits = {1: 0, 3: 0, 5: 0}
            n_ok = 0
            for r in rlist:
                fl = cs.hit_flags(r, cache)
                if fl is None:
                    continue
                n_ok += 1
                for k in (1, 3, 5):
                    if cs.topk(fl, k):
                        hits[k] += 1
            rates = {k: (100 * hits[k] / n_ok if n_ok else 0) for k in hits}
            summary[model][arm] = {"n": n_ok, "rates": rates,
                                   "tokens": tokens_model}
            log(f"{model}/{arm}: n={n_ok} top1/3/5 = "
                f"{rates[1]:.1f}/{rates[3]:.1f}/{rates[5]:.1f}% (累计 tokens {tokens_model})")
    lines = ["# 封闭旗舰探针结果（CPC 87 × 1 seed，非论文数据）\n",
             "| 模型 | 臂 | n | top-1 | top-3 | top-5 | 累计 tokens |",
             "|---|---|---|---|---|---|---|"]
    for model, arms in summary.items():
        for arm, s in arms.items():
            r = s["rates"]
            lines.append(f"| {model} | {arm} | {s['n']} | {r[1]:.1f} | {r[3]:.1f} | "
                         f"{r[5]:.1f} | {s['tokens']:,} |")
    (ROOT / "routing_study" / "results" / "probe_relay.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    log("探针完成，写入 probe_relay.md")


if __name__ == "__main__":
    main()
