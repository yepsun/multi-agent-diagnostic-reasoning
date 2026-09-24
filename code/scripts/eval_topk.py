#!/usr/bin/env python3
"""Evaluate top-1, top-3, top-10 accuracy using LLM judge."""
import json, sys, os, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from run_inference import call_llm

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else 'data/scheme_a_results.json'
OUT_FILE = RESULTS_FILE.replace('.json', '_topk.json')

with open(RESULTS_FILE) as f:
    results = json.load(f)

with open('data/mgh_qa_dataset.json') as f:
    dataset = json.load(f)

def get_gold(idx):
    case = dataset[idx]
    a = case.get('A', {})
    if isinstance(a, dict):
        for key in ['final_diagnosis', 'gold_standard_diagnosis', 'diagnosis']:
            val = a.get(key)
            if val:
                return str(val)
    return str(a)

MATCH_PROMPT = """Determine if the GOLD STANDARD diagnosis matches the PREDICTED diagnosis.

GOLD: {gold}
PREDICTED: {pred}

Answer MATCH or NO_MATCH. Consider synonyms and equivalent medical terminology as a match."""

print("Running LLM evaluation...")
per_case = []

for r in results:
    idx = r['idx'] if 'idx' in r else r.get('case_idx')
    ranked = r.get('ranked_diagnoses', [])
    gold = get_gold(idx)

    cr = {'idx': idx, 'gold': gold[:60], 'ranked': ranked[:10]}

    for k in [1, 3, 10]:
        match_at_k = False
        for pred in ranked[:k]:
            prompt = MATCH_PROMPT.format(gold=gold, pred=pred)
            resp, _ = call_llm(prompt, temperature=0.0, max_tokens=32, disable_thinking=True)
            verdict = resp.strip().upper()
            m = re.search(r'\b(MATCH|NO_MATCH)\b', verdict)
            if m and m.group(1) == 'MATCH':
                match_at_k = True
                break
            time.sleep(0.05)
        cr[f'top{k}'] = match_at_k

    marker = ''.join('✓' if cr[f'top{k}'] else '✗' for k in [1, 3, 10])
    print(f'{marker} Case {idx:2d}: gold="{gold[:40]}" top3="{ranked[1][:30] if len(ranked)>1 else "-"}"')
    per_case.append(cr)

top1 = sum(1 for c in per_case if c['top1'])
top3 = sum(1 for c in per_case if c['top3'])
top10 = sum(1 for c in per_case if c['top10'])
n = len(per_case)
print(f'\n{"="*50}')
print(f'Results from: {RESULTS_FILE}')
print(f'{"="*50}')
print(f'  Top-1:  {top1}/{n} = {top1/n*100:.0f}%')
print(f'  Top-3:  {top3}/{n} = {top3/n*100:.0f}%')
print(f'  Top-10: {top10}/{n} = {top10/n*100:.0f}%')
print(f'  Avg ranked_diagnoses length: {sum(len(r.get("ranked_diagnoses",[])) for r in results)/n:.1f}')

with open(OUT_FILE, 'w') as f:
    json.dump(per_case, f, indent=2)
print(f'Saved per-case results to {OUT_FILE}')
