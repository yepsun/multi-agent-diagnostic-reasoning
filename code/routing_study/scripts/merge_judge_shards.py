#!/usr/bin/env python3
"""Merge per-arm shard caches back into the shared judge cache.

Keys are content-addressed (`gold[:150] + "||" + cand[:150]`), so a shard only
ever contributes keys the main cache lacks; a key present on both sides is kept
from the main cache and any verdict disagreement is reported. The main cache is
backed up before being replaced, and the write is atomic.

Usage:
  ./.venv/bin/python routing_study/scripts/merge_judge_shards.py [shard ...]

With no arguments, every `judge_shard_*.json` next to the main cache is merged.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "routing_study" / "results"
MAIN = RESULTS / "judge_cache_glm_v3.json"


def main():
    shards = ([Path(p) for p in sys.argv[1:]] or
              sorted(RESULTS.glob("judge_shard_*.json")))
    main = json.loads(MAIN.read_text())
    before = len(main)
    shutil.copy2(MAIN, MAIN.with_suffix(".json.bak_pre_merge"))
    added = conflicts = 0
    report = []
    for shard in shards:
        data = json.loads(shard.read_text())
        new = conflict = 0
        for k, v in data.items():
            if k not in main:
                main[k] = v
                new += 1
            elif main[k] != v:
                conflict += 1
        added += new
        conflicts += conflict
        report.append(f"  {shard.name}: {len(data)} 条, 新增 {new}, 冲突 {conflict}")
    tmp = MAIN.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(main, ensure_ascii=False))
    os.replace(tmp, MAIN)
    print("\n".join(report))
    print(f"[merge] 主缓存 {before} -> {len(main)}（新增 {added}，"
          f"与既有判定冲突 {conflicts}）")
    print(f"[merge] 备份: {MAIN.with_suffix('.json.bak_pre_merge').name}")


if __name__ == "__main__":
    main()
