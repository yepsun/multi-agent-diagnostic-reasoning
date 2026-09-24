#!/usr/bin/env python3
"""修复 judge_cache 中因网络故障被误判为 False 的条目。

只针对 topn_seeds 实验用到的 (gold, candidate) 对：凡缓存为 False 的，
用 v2 判定提示词重新判定（call_llm_judge，严格模式：空/无法解析的返回
不写入缓存，留待重试）。覆写后重算 seed 汇总。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from run_inference import call_llm_judge  # noqa: E402

SEEDS_DIR = ROOT / "routing_study" / "results" / "topn_seeds"
JUDGE_CACHE = ROOT / "routing_study" / "results" / "topn_ablation" / "judge_cache.json"
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
   anatomic subtype, etiologic form) do NOT make the model wrong: e.g. "celiac disease"
   is correct for "celiac disease in histologic remission"; "tularemia" is correct for
   "ulceroglandular tularemia"; "acute pulmonary embolism" is correct for "acute massive
   pulmonary embolism with clot in transit".
4. WRONG (NO) only if the model names a DIFFERENT disease as its main diagnosis, or a
   generic category that never specifically names the reference disease.
Answer with exactly one word: YES or NO."""


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def judge_strict(gold, cand):
    """返回 True/False；网络或解析失败抛异常（不产生假 False）。"""
    raw, _ = call_llm_judge(PROMPT.format(gold=gold, pred=cand),
                            timeout=60, max_retries=2)
    v = (raw or "").strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    raise RuntimeError(f"unparseable verdict: {raw[:40]!r}")


def main():
    cache = json.loads(JUDGE_CACHE.read_text())
    pairs = set()
    for f in SEEDS_DIR.glob("*.jsonl"):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for cand in row["top5"]:
                pairs.add((row["gold"], cand))

    suspects = [(g, c) for g, c in pairs
                if cache.get(key_of(g, c)) is False]
    print(f"seed 实验候选对 {len(pairs)}，其中缓存为 False 待复核 {len(suspects)}",
          flush=True)

    repaired = confirmed_false = unresolved = 0
    changed_keys = []
    for round_no in (1, 2):  # 第二轮只重试未解析的
        if not suspects:
            break
        print(f"-- 第 {round_no} 轮：{len(suspects)} 条", flush=True)
        nxt = []
        with ThreadPoolExecutor(4) as ex:
            for (g, c), verdict in zip(suspects,
                                       ex.map(lambda pc: _safe(pc[0], pc[1]),
                                              suspects)):
                key = key_of(g, c)
                if verdict is None:
                    nxt.append((g, c))
                    continue
                cache[key] = verdict
                changed_keys.append(key)
                if verdict:
                    repaired += 1
                else:
                    confirmed_false += 1
        suspects = nxt
    if suspects:
        print(f"仍未解析（保持缓存不变，稍后重试）: {len(suspects)}", flush=True)
        for g, c in suspects:
            print("  UNRESOLVED:", key_of(g, c)[:90], flush=True)

    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"复核完成：False→True 翻转 {repaired}，确认 False {confirmed_false}，"
          f"未解析 {unresolved + len(suspects)}", flush=True)


def _safe(gold, cand):
    try:
        return judge_strict(gold, cand)
    except Exception:
        return None


if __name__ == "__main__":
    main()
