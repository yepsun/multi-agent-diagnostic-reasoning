#!/usr/bin/env python3
"""统计 gpt-5.1 全臂（Ax1/P 各 5 seeds + MDT 5 seeds synthesis）相对共享缓存仍缺的判分对数。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
from caselevel_stats import RESULTS, GLM_CACHE, key_of  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
GPT51 = RESULTS / "topn_seeds_gpt51"
MDT51 = RESULTS / "topn_mdt_gpt51"


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["case_id"]] = r
    return rows


def main():
    cache = json.loads(GLM_CACHE.read_text())
    rows = []
    for s in SEEDS:
        for scheme in ("Ax1", "P"):
            rows.extend(load(GPT51 / f"{scheme}_s{s}.jsonl").values())
        base = MDT51 / ("synthesis.jsonl" if s == 1 else f"s{s}/synthesis.jsonl")
        rows.extend(load(base).values())
    missing = sum(1 for r in rows for c in r["top5"][:5]
                  if key_of(r["gold"], c) not in cache)
    print(missing)


if __name__ == "__main__":
    main()
