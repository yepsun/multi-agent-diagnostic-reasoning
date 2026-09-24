#!/usr/bin/env python3
"""方案一：静态路由验证 runner。

对每个 case：
  A  = qwen3.8-flash, k=5 自洽采样 (temp=0.7, JSON 模式, thinking off)
  P  = deepseek-flash, 多视角单次 (disable_thinking)
  P' = qwen3.8-flash 同 prompt（同模型消融）
全部答案由 call_llm_judge 对金标准判对错，写 JSONL（可断点续跑）。
"""
import sys, os, json, argparse, re, time
from typing import Dict, List, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

# 必须在 import run_inference 之前：模块常量在 import 时读 env
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from run_inference import call_llm, parse_llm_output, call_llm_judge, get_current_model
from case_extraction import preprocess_case_text, format_structured_case
from scheme_a import _build_scheme_a_prompt
from scheme_perspective import PERSPECTIVE_PROMPT

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
RAW_PATH = os.path.join(RESULTS, "static_routing_raw.jsonl")
DEFAULT_DATASET = os.path.join(HERE, "..", "..", "data", "mgh_qa_dataset.json")

K_SAMPLES = 5
A_TEMPERATURE = 0.7
MAX_WORKERS = 4


def normalize_dx(s: str) -> str:
    s = (s or "").strip().rstrip(".").rstrip(",").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _gold_text(case: Dict) -> str:
    a = case.get("A")
    if isinstance(a, dict):
        for k in ("final_diagnosis", "gold_standard_diagnosis",
                  "most_likely_diagnosis", "answer", "diagnosis", "A", "a"):
            if a.get(k):
                return str(a[k])
        return json.dumps(a, ensure_ascii=False)
    return str(a)


def judge(gold: str, pred: str) -> bool:
    """语义判分：pred 点名的疾病与 gold 是否相同。判分失败按错误计。"""
    if not pred or pred.strip().lower().startswith("unable to determine"):
        return False
    prompt = f"""You are a medical evaluation judge. A model produced a diagnosis for a clinical case.
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
    raw, _ = call_llm_judge(prompt)
    return raw.strip().upper().startswith("YES")


def run_a_samples(structured_case: str, k: int = K_SAMPLES) -> List[Dict]:
    """A 的 k 次自洽采样（qwen3.8-flash）。"""
    prompt = _build_scheme_a_prompt(structured_case)
    out = []
    for _ in range(k):
        raw, usage = call_llm(prompt, temperature=A_TEMPERATURE, max_tokens=2048,
                              use_json_mode=True, provider="qwen", timeout=180)
        parsed = parse_llm_output(raw)
        out.append({
            "diagnosis": parsed.get("most_likely_diagnosis", "").strip(),
            "confidence": parsed.get("confidence_score"),
            "raw_len": len(raw),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        })
    return out


def run_p_single(structured_case: str, provider: str) -> Tuple[str, int]:
    """P 的多视角单次调用（provider 决定走 deepseek-flash 还是 qwen）。"""
    prompt = PERSPECTIVE_PROMPT.format(structured_case=structured_case)
    raw, usage = call_llm(prompt, temperature=0.1, max_tokens=2048,
                          disable_thinking=True, provider=provider, timeout=300)
    return parse_llm_output(raw).get("most_likely_diagnosis", "").strip(), \
        int(usage.get("total_tokens", 0) or 0)


def majority_answer(diags: List[str]) -> Tuple[str, float]:
    """归一化后的多数诊断与一致率（k=0 返回 ("", 0.0)；并列取先出现者）。"""
    norm = [normalize_dx(d) for d in diags if d]
    if not norm:
        return "", 0.0
    counts = Counter(norm)
    best = max(counts.values())
    ans = next(d for d in counts if counts[d] == best)
    return ans, best / len(diags)


def run_case(case: Dict) -> Dict:
    case_id = str(case.get("case_id", "unknown"))
    structured = format_structured_case(preprocess_case_text(case.get("Q", "")))
    gold = _gold_text(case)

    a_samples = run_a_samples(structured)
    a_diags = [s["diagnosis"] for s in a_samples]
    a_majority, a_cons = majority_answer(a_diags)
    a_tokens = sum(s["total_tokens"] for s in a_samples)

    p_answer, p_tokens = run_p_single(structured, "deepseek-flash")
    p_qwen_answer, p_qwen_tokens = run_p_single(structured, "qwen")

    row = {
        "case_id": case_id,
        "gold": gold,
        "a_samples": a_samples,
        "a_majority": a_majority,
        "a_consistency": a_cons,
        "a_correct": judge(gold, a_majority),
        "p_answer": p_answer, "p_correct": judge(gold, p_answer),
        "p_qwen_answer": p_qwen_answer, "p_qwen_correct": judge(gold, p_qwen_answer),
        "a_tokens": a_tokens, "p_tokens": p_tokens, "p_qwen_tokens": p_qwen_tokens,
    }
    print(f"[{case_id}] A={'T' if row['a_correct'] else 'F'}({a_cons:.0%}) "
          f"P={'T' if row['p_correct'] else 'F'} P'={'T' if row['p_qwen_correct'] else 'F'}", flush=True)
    return row


def load_done() -> set:
    if not os.path.exists(RAW_PATH):
        return set()
    with open(RAW_PATH) as f:
        return {json.loads(l)["case_id"] for l in f if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    cases = json.load(open(args.dataset))
    if args.limit:
        cases = cases[:args.limit]
    done = load_done()
    todo = [c for c in cases if str(c["case_id"]) not in done]
    print(f"dataset={len(cases)}, done={len(done)}, todo={len(todo)}, "
          f"A model=qwen3.8-flash, P model={os.environ.get('DEEPSEEK_MODEL')}")

    with open(RAW_PATH, "a") as f, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(run_case, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:
                print(f"[ERROR] case {futs[fut].get('case_id')}: {e}", flush=True)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
