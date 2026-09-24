#!/usr/bin/env python3
"""v3 判官重判统一缓存全量（约 2.25 万对）→ judge_cache_dsflash_v3.json。

输出翻转统计（预期单向 NO→YES）。主结果敏感性分析的数据基础。
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_v3 import v3_verdict  # noqa: E402

UNIFIED = ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"
V3_CACHE = ROOT / "routing_study" / "results" / "judge_cache_dsflash_v3.json"
WORKERS = 6


def main():
    old = json.loads(UNIFIED.read_text())
    new = json.loads(V3_CACHE.read_text()) if V3_CACHE.exists() else {}
    todo = [k for k in old if k not in new]
    print(f"全量 {len(old)} 对；v3 已有 {len(new)}（含 100 条盲评样本），待判 {len(todo)}",
          flush=True)

    def work(key):
        gold, _, cand = key.partition("||")
        try:
            return key, v3_verdict(gold, cand)
        except Exception:
            return key, None

    errs = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, (key, verdict) in enumerate(ex.map(work, todo), 1):
            if verdict is None:
                errs += 1
                continue
            new[key] = verdict
            if i % 1000 == 0 or i == len(todo):
                V3_CACHE.write_text(json.dumps(new, ensure_ascii=False))
                flips = sum(1 for k in new if k in old and new[k] != old[k])
                print(f"  {i}/{len(todo)} | 累计翻转 {flips} | 失败 {errs}",
                      flush=True)
    V3_CACHE.write_text(json.dumps(new, ensure_ascii=False))
    flips = [(k, old[k], new[k]) for k in old if k in new and old[k] != new[k]]
    n2y = sum(1 for _, o, n in flips if not o and n)
    print(f"\n完成：v3 判定 {len(new)} 对 | 翻转 {len(flips)}（NO→YES {n2y}，"
          f"YES→NO {len(flips)-n2y}）| 失败 {errs}", flush=True)
    print(f"已写 {V3_CACHE}", flush=True)


if __name__ == "__main__":
    main()
