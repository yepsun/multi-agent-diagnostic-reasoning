#!/bin/zsh
cd /Users/Yepsun/Mywork/Vscodeprojects/programs/MDT
echo "== $(date '+%H:%M:%S') =="
for f in routing_study/results/topn_ax5_mod_er/ErAx5_s5.jsonl \
         routing_study/results/topn_ax5_mod_er/ErAx5Mod_s5.jsonl \
         routing_study/results/topn_p_split/Pgen_s3.jsonl \
         routing_study/results/moderator_anatomy/lists_only_s1.jsonl \
         routing_study/results/topn_ax5_mod_mcr/McrAx5_s1.jsonl; do
  [ -f "$f" ] && echo "$(basename $f): $(wc -l < $f) 行 (写入于 $(stat -f %Sm -t %H:%M:%S $f))"
done
echo "运行中: $(ps aux | grep -E '[p]_split_cpc|[a]x5_mod_er|[m]oderator_anatomy|[a]x5_mod_mcr' | grep python | awk '{print $13}' | sort -u | tr '\n' ' ')"
for lg in /Users/yepsun/.zcode/cli/exec/sess_e7bebf96-ed89-4f8a-ac48-31e73584f2c7/call_ea6e5a6f876a46e69eae8cc4-stdout.log \
          /Users/yepsun/.zcode/cli/exec/sess_e7bebf96-ed89-4f8a-ac48-31e73584f2c7/call_8ebe37c2415b4b4d9a924f9c-stdout.log \
          /Users/yepsun/.zcode/cli/exec/sess_e7bebf96-ed89-4f8a-ac48-31e73584f2c7/call_9921ec03fc37426eba999512-stdout.log; do
  [ -f "$lg" ] && grep -c "失败\|Error\|Traceback" "$lg" 2>/dev/null | xargs -I{} echo "  $(basename $log 2>/dev/null): 失败/错误行 {}" 2>/dev/null
done
