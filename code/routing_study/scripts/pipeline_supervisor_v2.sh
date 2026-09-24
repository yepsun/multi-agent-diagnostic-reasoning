#!/bin/zsh
# 监督 v2：全串行 + 每阶段失败自动重试一轮（各阶段断点续跑）
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
PY=.venv/bin/python

run() {
  local name="$1"; local log="$2"; shift 2
  echo "[v2] $(date '+%H:%M:%S') 开始 $name"
  if ! "$@" > "$log" 2>&1; then
    echo "[v2] $name 失败，重试一轮"
    if ! "$@" > "$log" 2>&1; then
      echo "[v2] $name 重试仍失败，监督终止"; exit 1
    fi
  fi
  echo "[v2] $(date '+%H:%M:%S') 完成 $name"
}

echo "[v2] 监督启动 $(date '+%H:%M:%S')"
run "ER-mod"     /tmp/er_mod.log     env PHASE=mod     $PY routing_study/scripts/ax5_mod_er.py
run "ER-judge"   /tmp/er_judge.log   env PHASE=judge   $PY routing_study/scripts/ax5_mod_er.py
run "ER-analyze" /tmp/er_analyze.log env PHASE=analyze $PY routing_study/scripts/ax5_mod_er.py
run "PS-gen"     /tmp/ps_gen.log     env PHASE=gen     $PY routing_study/scripts/p_split_cpc.py
run "PS-mod"     /tmp/ps_mod.log     env PHASE=mod     $PY routing_study/scripts/p_split_cpc.py
run "PS-judge"   /tmp/ps_judge.log   env PHASE=judge   $PY routing_study/scripts/p_split_cpc.py
run "PS-analyze" /tmp/ps_analyze.log env PHASE=analyze $PY routing_study/scripts/p_split_cpc.py
run "ANATOMY"    /tmp/anatomy.log    env ANATOMY_PHASES=lists_only,k_sweep,order $PY routing_study/scripts/moderator_anatomy.py
run "ANATOMY-A"  /tmp/anatomy_a.log  env ANATOMY_PHASES=analyze $PY routing_study/scripts/moderator_anatomy.py
run "MCR"        /tmp/mcr.log        env PHASE=all $PY routing_study/scripts/ax5_mod_mcr.py
echo "[v2] ALL DONE $(date '+%H:%M:%S')"
