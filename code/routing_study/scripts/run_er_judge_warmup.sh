#!/bin/bash
# ER 5-seed GLM 判定预热循环：推理期间每 10 分钟对现有行补判一轮，
# 直到 routing_study/results/topn_erreason/.infer_done 出现（由主流程创建）。
set -u
cd "$(dirname "$0")/../.."
for i in $(seq 1 24); do
  if [ -f routing_study/results/topn_erreason/.infer_done ]; then
    echo "推理已完成标记出现，退出预热循环"
    exit 0
  fi
  PHASE=judge ./.venv/bin/python routing_study/scripts/erreason_5seeds.py
  sleep 600
done
