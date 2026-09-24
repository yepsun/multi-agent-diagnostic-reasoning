#!/usr/bin/env python3
"""GLM-5.3-flash 作为独立判官的可行性探测。

在 100 条双人盲评样本上，用冻结的 v2 判定规则（英文原文术语）让 GLM 判定，
与人类 A/B 及 deepseek-flash 判官对比 κ。思考模式不可关，用 effort low +
max_tokens 8192；严格解析（无 YES/NO 视为失败并计数）。
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import requests

from judge_study import load_sample, cohens_kappa, STUDY_DIR  # noqa: E402

KEY = [l.split("=", 1)[1].strip() for l in open(ROOT / "scripts" / ".env")
       if l.startswith("ZHIPU_API_KEY=")][0]
OUT = STUDY_DIR / "glm_judge_v2_100.json"

PROMPT = """You are a medical evaluation judge. A model produced a diagnosis for a clinical case.
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


def glm_judge(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user",
                          "content": PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json=body, timeout=420)
    d = r.json()
    content = (d["choices"][0]["message"].get("content") or "")
    tokens = (d.get("usage") or {}).get("total_tokens", 0)
    v = content.strip().upper()
    if v.startswith("YES"):
        return True, tokens, None
    if v.startswith("NO"):
        return False, tokens, None
    return None, tokens, content.strip()[:60]


def main():
    sample = load_sample()
    out_path = OUT
    verdicts = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [t for t in sample if str(t["item_id"]) not in verdicts]
    print(f"GLM 独立判官试批：已完成 {len(verdicts)}，待判 {len(todo)}", flush=True)

    llm = {t["item_id"]: t["llm_verdict"] for t in sample}
    results = {}
    parse_fail = 0
    tokens_total = 0
    latencies = []
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(glm_judge, t["gold"], t["candidate"]): t
                for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                verdict, tokens, raw_tail = fut.result()
            except Exception as e:
                verdict, tokens, raw_tail = None, 0, f"EXC {e}"
            tokens_total += tokens
            if verdict is None:
                parse_fail += 1
                print(f"  [{i}] 解析失败: {raw_tail}", flush=True)
            else:
                verdicts[str(t["item_id"])] = verdict
                results[t["item_id"]] = verdict
            out_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1))
    print(f"试批完成：成功 {len(verdicts)} | 解析失败 {parse_fail} | "
          f"总 tokens {tokens_total}", flush=True)

    # κ 对比（只算 GLM 成功且两人类都有的条目）
    a = json.loads((STUDY_DIR / "annotations_a.json").read_text())
    b = json.loads((STUDY_DIR / "annotations_b.json").read_text())
    ids = [int(k) for k in verdicts]
    va = [a[str(i)] for i in ids]
    vb = [b[str(i)] for i in ids]
    vg = [verdicts[str(i)] for i in ids]
    vl = [llm[i] for i in ids]
    print("\n===== GLM-5.3-flash 独立判官 vs 基准 =====")
    print(f"GLM vs 评者A:      {cohens_kappa(vg, va)}")
    print(f"GLM vs 评者B:      {cohens_kappa(vg, vb)}")
    print(f"GLM vs deepseek判官: {cohens_kappa(vg, vl)}")
    print(f"（参照）deepseek判官 vs 评者A: {cohens_kappa(vl, va)}")
    print(f"（参照）deepseek判官 vs 评者B: {cohens_kappa(vl, vb)}")


if __name__ == "__main__":
    main()
