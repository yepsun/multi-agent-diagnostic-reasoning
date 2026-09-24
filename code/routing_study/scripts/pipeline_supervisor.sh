#!/bin/zsh
# 流水线监督：等 ER 链结束 → P-split → 主持人解剖 → MCR 析因复制
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
PY=.venv/bin/python
echo "[supervisor] 等待 ER 链结束..."
while pgrep -f "ax5_mod_e[r].py" > /dev/null; do sleep 90; done
echo "[supervisor] ER 结束，P-split 重跑（断点续跑）"
$PY routing_study/scripts/p_split_cpc.py 2>&1 | tail -2
echo "[supervisor] 主持人解剖三联"
$PY routing_study/scripts/moderator_anatomy.py 2>&1 | tail -2
echo "[supervisor] MCR 析因复制全量"
$PY routing_study/scripts/ax5_mod_mcr.py 2>&1 | tail -3
echo "[supervisor] ALL DONE"
