#!/usr/bin/env python3
"""抽 100 条全新验证对（排除原盲评样本），供 v3 的无偏人类验证。

写入 judge_validity/sample_v4check.json；评者用 ?set=v2check 访问。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

from judge_study import CACHE, STUDY_DIR  # noqa: E402
from webapp.clustering import same_disease  # noqa: E402

OUT = STUDY_DIR / "sample_v4check.json"
N_PER_CLASS = 50


def main():
    cache = json.loads(CACHE.read_text())
    used = {t["key"] for t in json.loads(
        (STUDY_DIR / "sample.json").read_text())}
    triples = []
    for key, verdict in cache.items():
        if key in used:
            continue
        gold, sep, cand = key.partition("||")
        if not sep or len(gold) < 4 or len(cand) < 4:
            continue
        triples.append({"key": key, "gold": gold, "candidate": cand,
                        "llm_verdict": bool(verdict),
                        "llm_v3": None,
                        "heuristic_same": same_disease(gold, cand)})
    yes = [t for t in triples if t["llm_verdict"]]
    no = [t for t in triples if not t["llm_verdict"]]

    def hard(t):
        return t["llm_verdict"] != t["heuristic_same"]
    rng = random.Random(77)
    for pool in (yes, no):
        rng.shuffle(pool)
    sample = []
    for pool, n in ((yes, N_PER_CLASS), (no, N_PER_CLASS)):
        h = [t for t in pool if hard(t)]
        picked = h[:n // 2]
        seen = {id(t) for t in picked}
        for t in pool:
            if len(picked) >= n:
                break
            if id(t) in seen:
                continue
            picked.append(t)
            seen.add(id(t))
        sample.extend(picked)
    rng.shuffle(sample)
    for i, t in enumerate(sample):
        t["item_id"] = i
    OUT.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    print(f"新验证样本 {len(sample)} 条（YES {sum(1 for t in sample if t['llm_verdict'])}"
          f"/NO {sum(1 for t in sample if not t['llm_verdict'])}，边界对 "
          f"{sum(1 for t in sample if hard(t))}）已写 {OUT}")


if __name__ == "__main__":
    main()
