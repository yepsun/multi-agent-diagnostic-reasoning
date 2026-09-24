#!/bin/bash
# ER 5-seed GLM 判定看门狗：低并发跑判定，检测到停滞（缓存 5 分钟不增长）就杀掉重启。
# 判定脚本按缓存内容寻址，只判缺失对，因此反复重启是幂等的。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT || exit 1

CACHE=routing_study/results/judge_cache_glm_v3.json
LOG=routing_study/results/erreason_5seeds_judge_watchdog.log
STALL_LIMIT=900   # 秒：缓存多久不增长视为停滞

count_cache() { ./.venv/bin/python -c "import json;print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo 0; }

attempt=0
while true; do
  attempt=$((attempt+1))
  before=$(count_cache)
  echo "[$(date +%H:%M:%S)] 第 ${attempt} 轮启动，缓存 ${before}" >> "$LOG"

  GLM_WORKERS=8 GLM_TIMEOUT=45 ROUND_BUDGET=1800 PHASE=judge \
    ./.venv/bin/python routing_study/scripts/erreason_5seeds.py >> "$LOG" 2>&1 &
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
  if [ "$after" = "$before" ] && [ $attempt -gt 3 ]; then
    echo "[$(date +%H:%M:%S)] 连续无进展，退出" >> "$LOG"; break
  fi
done
echo "[$(date +%H:%M:%S)] 看门狗结束，最终缓存 $(count_cache)" >> "$LOG"
