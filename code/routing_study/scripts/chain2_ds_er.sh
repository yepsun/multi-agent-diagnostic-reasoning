#!/bin/zsh
# 第二棒：主链（supervisor v2）结束后自动接续 deepseek-flash ER 复制收尾。
# 等待 → PHASE=sample（补 s5）→ PHASE=mod（补全 5×364）→ judge（循环至 0 缺失）→ analyze → 标记。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
PY=.venv/bin/python
LOG=results/chain2.log
MARK=results/chain2_done.marker
RESUME_LOG=routing_study/results/resume_20260920.log

log() { echo "[chain2] $(date '+%F %T') $*" >> "$LOG"; }

# 1) 等待主链结束：supervisor 进程消失且恢复日志出现退出码行
while true; do
  if ! pgrep -f "pipeline_supervisor_v[2]|ax5_mod_mc[r].py|er_judge_retry_lo[o]p" > /dev/null; then
    if grep -q "supervisor 退出码" "$RESUME_LOG" 2>/dev/null; then
      break
    fi
  fi
  sleep 120
done
log "主链已结束，开始 deepseek-flash ER 收尾"

# 2) 采样补 s5 → 主持人补全（各重试一轮）
for ph in sample mod; do
  log "开始 PHASE=$ph"
  if ! PHASE=$ph $PY routing_study/scripts/ax5_mod_er_ds.py >> /tmp/ds_er_$ph.log 2>&1; then
    log "PHASE=$ph 失败，重试一轮"
    PHASE=$ph $PY routing_study/scripts/ax5_mod_er_ds.py >> /tmp/ds_er_$ph.log 2>&1 || { log "PHASE=$ph 重试仍失败，终止"; exit 1; }
  fi
  log "完成 PHASE=$ph"
done

# 3) 判分：循环至 0 缺失（最多 20 轮，轮间 10 分钟退避）
for rnd in $(seq 1 20); do
  log "判分轮 $rnd"
  DS_ER_JUDGE=1 PHASE=judge $PY routing_study/scripts/ax5_mod_er_ds.py >> /tmp/ds_er_judge.log 2>&1 \
    || { log "判分轮 $rnd 调用失败"; sleep 600; continue; }
  miss=$($PY - <<'PYEOF'
import json, sys
from pathlib import Path
ROOT = Path("/Users/Yepsun/Mywork/Vscodeprojects/programs/MDT")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'routing_study'/'scripts')); sys.path.insert(0, str(ROOT/'scripts'))
import caselevel_stats as cs
import ax5_mod_er_ds as ds
rows = []
for seed in ds.SEEDS:
    for arm in ("Ax1_ds", "ErAx5_ds", "ErAx5Mod_ds"):
        rows.extend(ds.load_done(ds.path_of(arm, seed)).values())
if not rows:
    print("ERR"); sys.exit(0)
cache = json.loads(cs.GLM_CACHE.read_text())
print(len(cs.missing_pairs(rows, cache)))
PYEOF
)
  log "判分轮 $rnd 剩余缺失 $miss"
  [ "$miss" = "0" ] && break
  [ "$miss" = "ERR" ] && { log "缺失统计异常，终止待人工"; exit 1; }
  sleep 600
done

# 4) 分析
log "开始 analyze"
PHASE=analyze $PY routing_study/scripts/ax5_mod_er_ds.py >> /tmp/ds_er_analyze.log 2>&1 \
  && log "analyze 完成" || { log "analyze 失败"; exit 1; }
touch "$MARK"
log "第二棒完成"
