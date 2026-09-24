#!/bin/zsh
# gpt-5.1 × CPC 87 × 5 seeds × 3 臂（A×1/P/MDT）全量生成 → 判分 → 分析。
# 生成走中转（OPENAI_* 已在 scripts/.env）；判定走冻结 GLM 直连（flock 与
# deepseek-flash ER 判定互斥）。产出 results/gpt51_family_cpc.{json,md}。
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
PY=.venv/bin/python
export MAX_WORKERS=4
LOG=results/gpt51_full.log

log() { echo "[gpt51] $(date '+%F %T') $*" >> "$LOG"; echo "[gpt51] $*"; }

log "开始 PHASE=infer（A×1 + P，5 seeds，1,740 次调用）"
if ! PHASE=infer $PY routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_infer.log 2>&1; then
  log "infer 失败，重试一轮"
  PHASE=infer $PY routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_infer.log 2>&1 || { log "infer 重试仍失败，终止"; exit 1; }
fi
log "完成 PHASE=infer"

for seed in 1 2 3 4 5; do
  log "开始 MDT seed $seed（522 次调用）"
  if ! MDT_OUTDIR=topn_mdt_gpt51 MDT_SEED=$seed \
       MDT_ROLE_PROVIDERS=openai,openai,openai,openai,openai \
       MDT_SYNTH_PROVIDER=openai MDT_SKIP_JUDGE=1 \
       $PY routing_study/scripts/mdt_cpc.py >> /tmp/gpt51_mdt_s$seed.log 2>&1; then
    log "MDT seed $seed 失败，重试一轮"
    MDT_OUTDIR=topn_mdt_gpt51 MDT_SEED=$seed \
      MDT_ROLE_PROVIDERS=openai,openai,openai,openai,openai \
      MDT_SYNTH_PROVIDER=openai MDT_SKIP_JUDGE=1 \
      $PY routing_study/scripts/mdt_cpc.py >> /tmp/gpt51_mdt_s$seed.log 2>&1 || { log "MDT seed $seed 重试仍失败，终止"; exit 1; }
  fi
  log "完成 MDT seed $seed"
done

log "开始 PHASE=judge（冻结 GLM 直连，与 DS ER 判定 flock 互斥）"
if ! PHASE=judge $PY routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_judge.log 2>&1; then
  log "judge 失败，重试一轮"
  PHASE=judge $PY routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_judge.log 2>&1 || { log "judge 重试仍失败，终止"; exit 1; }
fi
log "完成 PHASE=judge"

log "开始 PHASE=analyze"
PHASE=analyze $PY routing_study/scripts/gpt51_family_cpc.py >> /tmp/gpt51_analyze.log 2>&1 || { log "analyze 失败，终止"; exit 1; }
log "全部完成：results/gpt51_family_cpc.{json,md}"
