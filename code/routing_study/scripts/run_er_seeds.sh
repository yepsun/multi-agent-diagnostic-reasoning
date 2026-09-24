#!/bin/bash
# ER-Reason seeds 2-5 推理驱动：断点续跑直到完整性校验通过。
# 用法: bash run_er_seeds.sh <seed>
set -u
cd "$(dirname "$0")/../.."
s="$1"
for attempt in 1 2 3 4 5 6; do
  ER_OUTDIR="topn_erreason/s${s}" SKIP_JUDGE=1 \
    ./.venv/bin/python routing_study/scripts/topn_erreason.py
  if ./.venv/bin/python routing_study/scripts/er_seeds_clean.py \
      "routing_study/results/topn_erreason/s${s}"; then
    echo "[s${s}] 完整性校验通过"
    exit 0
  fi
  echo "[s${s}] 第 ${attempt} 轮不完整，续跑"
done
echo "[s${s}] 6 轮后仍不完整" >&2
exit 1
