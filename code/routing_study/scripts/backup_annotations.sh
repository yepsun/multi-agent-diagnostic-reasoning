#!/bin/bash
# judge validity 标注备份：每 10 分钟快照一次，保留最近 20 份
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
SRC=routing_study/results/judge_validity
BAK=$SRC/backups
mkdir -p "$BAK"
while true; do
  for w in a b; do
    f="$SRC/annotations_$w.json"
    [ -f "$f" ] && cp "$f" "$BAK/annotations_$w_$(date +%Y%m%d_%H%M%S).json"
  done
  ls -t "$BAK"/annotations_a_* 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null
  ls -t "$BAK"/annotations_b_* 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null
  sleep 600
done
