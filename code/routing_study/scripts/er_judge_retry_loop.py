import os as _os
#!/usr/bin/env python3
"""ER Ax5Mod 判定补齐循环：反复调用 judge_missing 直到 0 缺失（限流自愈）。"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'routing_study'/'scripts')); sys.path.insert(0, str(ROOT/'scripts'))
import caselevel_stats as cs
from topn_erreason import load_cases
from ax5_mod_er import sample_path, mod_path

rows = []
cases = load_cases()
ids = {c['case_id'] for c in cases}
for s in range(1, 6):
    for p in (sample_path(s), mod_path(s)):
        for cid, r in cs.load(p).items():
            if cid in ids:
                rows.append(r)
cache = json.loads(cs.GLM_CACHE.read_text())
print(f"初始缓存 {len(cache)}，待判定对（首次统计）...", flush=True)
missing0 = cs.missing_pairs(rows, cache)
print(f"缺 {len(missing0)} 对", flush=True)
for rnd in range(1, 21):
    if not missing0:
        break
    print(f"[轮 {rnd}] 尝试 {len(missing0)} 对", flush=True)
    cache = cs.judge_missing(rows)
    json.dump(cache, open(cs.GLM_CACHE, 'w'), ensure_ascii=False)
    missing0 = cs.missing_pairs(rows, cache)
    print(f"[轮 {rnd}] 剩余缺失 {len(missing0)}", flush=True)
    if missing0:
        time.sleep(600)  # 限流自愈等待 10 分钟
print("DONE，剩余缺失:", len(missing0))
