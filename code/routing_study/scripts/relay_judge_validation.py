#!/usr/bin/env python3
"""中转 GLM 判官一致性验证：抽样已判定对重判，衡量与直连判定的一致率。

通过门槛：一致率 ≥95%（判定模型相同、参数相同，理论上应等于模型的
自一致性上限）。通过后中转通道方可用于判定加速，并在补充材料披露。
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
from judge_v3 import V3_PROMPT  # noqa: E402

KEY = "/* scrubbed */"
URL = "https://api.openai.com/v1 /* scrubbed: set OPENAI_API_BASE */"
N_PER_CLASS = 60


def relay_verdict(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
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
            time.sleep(5)
    return "timeout"


def main():
    cache = json.loads((ROOT / "routing_study" / "results" /
                        "judge_cache_glm_v3.json").read_text())
    yes_keys = [k for k, v in cache.items() if v is True]
    no_keys = [k for k, v in cache.items() if v is False]
    random.seed(20260922)
    sample = ([(k, True) for k in random.sample(yes_keys, N_PER_CLASS)] +
              [(k, False) for k in random.sample(no_keys, N_PER_CLASS)])
    print(f"抽样 {len(sample)} 对（YES/NO 各 {N_PER_CLASS}）", flush=True)

    def work(item):
        k, primary = item
        gold, cand = k.split("||", 1)
        return k, primary, relay_verdict(gold, cand)

    agree = disagree = none_ct = 0
    disagreements = []
    with ThreadPoolExecutor(6) as ex:
        for i, (k, primary, v) in enumerate(ex.map(work, sample), 1):
            if v in (True, False):
                if v == primary:
                    agree += 1
                else:
                    disagree += 1
                    if len(disagreements) < 5:
                        disagreements.append((k[:80], primary, v))
            else:
                none_ct += 1
            if i % 30 == 0:
                print(f"  {i}/{len(sample)}", flush=True)
    total = agree + disagree
    rate = 100 * agree / total if total else 0
    print(f"\n一致 {agree} / 不一致 {disagree} / 未决 {none_ct}")
    print(f"一致率: {rate:.1f}%（门槛 95%）→ {'通过 ✓' if rate >= 95 else '不通过 ✗'}")
    for k, p, v in disagreements:
        print(f"  不一致样例: primary={p} relay={v} | {k}")
    (ROOT / "routing_study" / "results" / "relay_judge_validation.json").write_text(
        json.dumps({"agree": agree, "disagree": disagree, "unresolved": none_ct,
                    "rate": rate}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
