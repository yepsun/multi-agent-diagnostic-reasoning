#!/bin/zsh
# 看门狗 v2：监督主链（ER判分→supervisor v2）与第二棒（chain2 deepseek-flash ER）。
# 每 5 分钟采样进度（各日志总行数）与进程存活：
# - 进度 60 分钟无增长且进程存活 → 杀链并重启当前阶段应跑的链（幂等断点续跑）
# - 进程数为 0 且完成标记未出现 → 重启（崩溃自愈），主链未完重启主链，主链已完重启 chain2
# - results/chain2_done.marker 出现且进程退出 → 全部完成，看门狗退出
# 事件写入 results/watchdog_events.log，状态快照写入 results/watchdog_status.log
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
RESUME_LOG=routing_study/results/resume_20260920.log
EVENTS=routing_study/results/watchdog_events.log
STATUS=routing_study/results/watchdog_status.log
DONE_MARK=results/chain2_done.marker
ROOT=/Users/Yepsun/Mywork/Vscodeprojects/programs/MDT

progress_snapshot() {
  local n=0
  for f in "$RESUME_LOG" /tmp/er_mod.log /tmp/er_judge.log /tmp/er_analyze.log \
           /tmp/ps_gen.log /tmp/ps_mod.log /tmp/ps_judge.log /tmp/ps_analyze.log \
           /tmp/anatomy.log /tmp/anatomy_a.log /tmp/mcr.log \
           /tmp/ds_er_sample.log /tmp/ds_er_mod.log /tmp/ds_er_judge.log \
           /tmp/ds_er_analyze.log routing_study/results/chain2.log \
           /tmp/mcr_judge_retry.log \
           routing_study/results/mcr_judge_retry.log; do
    [ -f "$f" ] && n=$((n + $(wc -l < "$f" | tr -d ' ')))
  done
  echo "$n"
}

chain_alive() {
  pgrep -f "er_judge_retry_lo[o]p|pipeline_supervisor_v[2]|ax5_mod_e[r].py|p_split_cp[c].py|moderator_anatom[y].py|ax5_mod_mc[r].py|chain2_ds_e[r]|ax5_mod_er_d[s].py|mcr_judge_retry_loo[p]" > /dev/null
}

kill_chain() {
  pkill -f "er_judge_retry_lo[o]p"
  pkill -f "pipeline_supervisor_v[2]"
  pkill -f "ax5_mod_e[r].py"
  pkill -f "p_split_cp[c].py"
  pkill -f "moderator_anatom[y].py"
  pkill -f "ax5_mod_mc[r].py"
  pkill -f "chain2_ds_e[r]"
  pkill -f "mcr_judge_retry_loo[p]"
  pkill -f "ax5_mod_er_d[s].py"
  sleep 5
}

start_main() {
  nohup zsh -c "
    set -o pipefail
    { echo '=== [重] supervisor v2 重启于 '\"\$(date)\"' ===';
      bash $ROOT/routing_study/scripts/pipeline_supervisor_v2.sh
      echo '=== [重] supervisor 退出码 '\$?' 结束于 '\"\$(date)\"' ==='; } >> $ROOT/routing_study/results/resume_20260920.log 2>&1
  " > /dev/null 2>&1 &
}

start_chain2() {
  nohup bash $ROOT/routing_study/scripts/chain2_ds_er.sh > /dev/null 2>&1 &
}

main_chain_done() {
  grep -q "supervisor 退出码" "$RESUME_LOG" 2>/dev/null && ! pgrep -f "pipeline_supervisor_v[2]|ax5_mod_mc[r].py" > /dev/null
}

echo "$(date '+%F %T') 看门狗v2启动" >> "$EVENTS"
STALL_LIMIT=12          # 12 × 5min = 60 分钟无进展
stall_count=0
restarts=0
last_prog=-1

while true; do
  prog=$(progress_snapshot)
  alive=$(chain_alive && echo 1 || echo 0)
  echo "$(date '+%F %T') prog=$prog alive=$alive stall=$stall_count restarts=$restarts" >> "$STATUS"

  if [ "$alive" = "0" ] && [ -f "$DONE_MARK" ]; then
    echo "$(date '+%F %T') 两棒全部完成（chain2 标记出现），看门狗退出" >> "$EVENTS"
    exit 0
  fi

  if [ "$alive" = "0" ]; then
    if [ "$restarts" -ge 5 ]; then
      echo "$(date '+%F %T') 重启次数达上限 5，放弃（需人工介入）" >> "$EVENTS"
      exit 1
    fi
    restarts=$((restarts + 1))
    if main_chain_done; then
      echo "$(date '+%F %T') 进程全灭且主链已完 → 第 $restarts 次重启 chain2" >> "$EVENTS"
      start_chain2
    else
      echo "$(date '+%F %T') 进程全灭且主链未完 → 第 $restarts 次重启主链" >> "$EVENTS"
      start_main
    fi
    sleep 60
    continue
  fi

  if [ "$prog" = "$last_prog" ]; then
    stall_count=$((stall_count + 1))
    if [ "$stall_count" -ge "$STALL_LIMIT" ]; then
      if [ "$restarts" -ge 5 ]; then
        echo "$(date '+%F %T') 停滞达限且重启次数达上限，放弃（需人工介入）" >> "$EVENTS"
        exit 1
      fi
      restarts=$((restarts + 1))
      echo "$(date '+%F %T') 进度 $prog 停滞 60 分钟 → 杀链并第 $restarts 次重启" >> "$EVENTS"
      kill_chain
      if main_chain_done; then start_chain2; else start_main; fi
      stall_count=0
      sleep 60
      continue
    fi
  else
    stall_count=0
  fi
  last_prog=$prog
  sleep 300
done
