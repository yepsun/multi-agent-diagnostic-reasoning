#!/usr/bin/env python3
"""Per-arm shard judging: fill one shard cache from a set of run files.

The shared judge cache is a single JSON file that the judging helpers rewrite
whole, so concurrent writers would lose each other's verdicts. To judge several
new experiment arms in parallel we give each arm its own shard file, started
from a snapshot of the main cache (so only genuinely new pairs are sent), and
merge the shards back afterwards with `merge_judge_shards.py`.

Usage:
  SHARD_WORKERS=8 ./.venv/bin/python routing_study/scripts/judge_shards.py \
      <shard_path> <run_file_glob> [<run_file_glob> ...]

Rows are read as JSONL with `gold` and `top5`; a `top5_sc` field, when present,
is judged as a second candidate list (the self-consistency variant).
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402


def load_rows(patterns):
    rows = []
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    for path in files:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"gold": r["gold"], "top5": r.get("top5") or []})
            if r.get("top5_sc"):
                rows.append({"gold": r["gold"], "top5": r["top5_sc"]})
    print(f"[shard] {len(files)} 个文件，{len(rows)} 行候选列表", flush=True)
    return rows


def main():
    shard = Path(sys.argv[1])
    patterns = sys.argv[2:]
    cs.GLM_WORKERS = int(os.environ.get("SHARD_WORKERS", "8"))
    rows = load_rows(patterns)
    before = len(json.loads(shard.read_text())) if shard.exists() else 0
    cache = cs.judge_missing(rows, cache_path=shard)
    print(f"[shard] {shard}: {before} -> {len(cache)} 条判定 "
          f"(新增 {len(cache) - before})", flush=True)


if __name__ == "__main__":
    main()
