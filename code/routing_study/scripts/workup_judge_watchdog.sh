#!/bin/bash
# workup 全量判定的看门狗：GLM 判定只在缓存缺失时执行，重启幂等。
# 停滞（缓存 15 分钟不增长）则杀掉重启。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT || exit 1

CACHE=routing_study/results/judge_cache_glm_v3.json
LOG=routing_study/results/workup_judge_watchdog.log
STALL_LIMIT=900

count_cache() { ./.venv/bin/python -c "import json;print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo 0; }

attempt=0
while true; do
  attempt=$((attempt+1))
  before=$(count_cache)
  echo "[$(date +%H:%M:%S)] 第 ${attempt} 轮启动，缓存 ${before}" >> "$LOG"

  SEEDS="1,2,3,4,5" PHASE=judge GLM_WORKERS=8 GLM_TIMEOUT=45 ROUND_BUDGET=900 \
    ./.venv/bin/python routing_study/scripts/eval_workup_vs_baseline.py >> "$LOG" 2>&1 &
  pid=$!
  last=$before; last_change=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    now=$(count_cache)
    if [ "$now" != "$last" ]; then last=$now; last_change=$(date +%s); fi
    if [ $(( $(date +%s) - last_change )) -gt $STALL_LIMIT ]; then
      echo "[$(date +%H:%M:%S)] 停滞 ${STALL_LIMIT}s（缓存 ${now}），重启" >> "$LOG"
      kill -9 $pid 2>/dev/null; sleep 2; break
    fi
  done
  wait $pid 2>/dev/null
  after=$(count_cache)
  echo "[$(date +%H:%M:%S)] 第 ${attempt} 轮结束，缓存 ${before} -> ${after}" >> "$LOG"
  if [ "$after" = "$before" ] && [ $attempt -gt 2 ]; then
    echo "[$(date +%H:%M:%S)] 连续无进展，退出" >> "$LOG"; break
  fi
  if [ "$after" -ge 53000 ]; then echo "[$(date +%H:%M:%S)] 缓存已达预期规模，退出" >> "$LOG"; break; fi
done
echo "[$(date +%H:%M:%S)] 看门狗结束，最终缓存 $(count_cache)" >> "$LOG"
