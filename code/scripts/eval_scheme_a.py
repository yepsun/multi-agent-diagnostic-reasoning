#!/usr/bin/env python3
"""Evaluate scheme_a results with LLM matching."""
import json, sys, os, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from run_inference import call_llm

with open('data/scheme_a_results.json') as f:
    results = json.load(f)

with open('data/mgh_qa_dataset.json') as f:
    data = json.load(f)

MATCH_PROMPT = """You are evaluating a medical diagnosis system. Given the GOLD STANDARD diagnosis and the SYSTEM'S PREDICTED diagnosis, determine if they match.

GOLD STANDARD: {gold}
SYSTEM PREDICTION: {pred}

Does the prediction match the gold standard? Consider:
- Synonyms and equivalent medical terminology count as a match
- Different levels of specificity count as a match if the core diagnosis is the same
- The prediction should capture the PRIMARY diagnosis (not just a feature/symptom)

Answer with exactly one word: MATCH or NO_MATCH"""

correct = 0
failed_parse = 0
for r in results:
    idx = r['idx']
    case = data[idx]
    a = case.get('A', {})
    if isinstance(a, dict):
        for key in ['final_diagnosis', 'gold_standard_diagnosis', 'diagnosis']:
            val = a.get(key)
            if val:
                gold = str(val)
                break
        else:
            gold = ''
    else:
        gold = str(a)

    pred = r['pred']
    if pred.startswith('ERROR'):
        print(f'✗ Case {idx:2d}: ERROR')
        continue

    prompt = MATCH_PROMPT.format(gold=gold, pred=pred)
    resp, usage = call_llm(prompt, temperature=0.0, max_tokens=32, disable_thinking=True)
    verdict = resp.strip().upper()
    m = re.search(r'\b(MATCH|NO_MATCH)\b', verdict)
    if not m:
        failed_parse += 1
        is_match = False
        print(f'? Case {idx:2d}: unparseable: {verdict}')
    else:
        is_match = m.group(1) == 'MATCH'

    if is_match:
        correct += 1
    marker = '✓' if is_match else '✗'
    print(f'{marker} Case {idx:2d}: gold="{gold[:50]}" pred="{pred[:50]}"')
    time.sleep(0.3)

print(f'\nScheme A: {correct}/{len(results)} = {correct/len(results)*100:.0f}% (failed_parse={failed_parse})')
