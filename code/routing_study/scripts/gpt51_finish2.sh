#!/bin/zsh
# gpt-5.1 判分补收尾：循环 judge 直到缺失=0，才允许跑 analyze。
# 教训：chain3 依赖退出码 + 固定 3 次尝试，服务商劣化窗口内带着 3,494 缺失就跑了分析。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
export GLM_WORKERS=6 GLM_HARD_DEADLINE=180
LOG=routing_study/results/gpt51_retry2.log
echo "=== gpt51 补收尾启动 $(date) ===" >> $LOG
for i in {1..24}; do
  PHASE=judge .venv/bin/python routing_study/scripts/gpt51_family_cpc.py >> $LOG 2>&1
  M=$(.venv/bin/python routing_study/scripts/gpt51_missing.py)
  echo "[$(date '+%m-%d %H:%M')] 第 $i 轮 judge 退出，剩余缺失 $M" >> $LOG
  if [ "$M" -eq 0 ]; then
    echo "[$(date '+%m-%d %H:%M')] 缺失清零，运行 analyze" >> $LOG
    break
  fi
  sleep 300
done
PHASE=analyze .venv/bin/python routing_study/scripts/gpt51_family_cpc.py >> routing_study/results/gpt51_analyze.log 2>&1
echo "gpt51 补收尾完成 $(date)" >> routing_study/results/chain3.log
