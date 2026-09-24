#!/bin/zsh
# chain3：DS 收尾循环退出后，补判 gpt-5.1 剩余对（6 worker，GLM 官方端点）+ 重分析。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
export GLM_WORKERS=6 GLM_HARD_DEADLINE=180
for i in 1 2 3; do
  PHASE=judge .venv/bin/python routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_judge.log 2>&1 && break
  sleep 120
done
PHASE=analyze .venv/bin/python routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_analyze.log 2>&1
echo "chain3 完成 $(date)" >> routing_study/results/chain3.log
