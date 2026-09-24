#!/usr/bin/env python3
"""Run scheme A on all cases with concurrency and resume support."""
import json, sys, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scheme_a import run_scheme_a

with open('data/mgh_qa_dataset.json') as f:
    data = json.load(f)

out_path = 'data/scheme_a_results.json'
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    done_idx = {r['idx'] for r in results}
    print(f"Resuming — {len(done_idx)} cases already in {out_path}")
else:
    results = []
    done_idx = set()

cases = [c for i, c in enumerate(data) if i not in done_idx]
if not cases:
    print("All cases done.")
    sys.exit(0)

print(f"Running scheme A on {len(cases)} cases with concurrency 10...")

def process(case):
    idx = next(i for i, c in enumerate(data) if c is case)
    t0 = time.time()
    try:
        result = run_scheme_a(case)
        gold = case.get('A', {})
        if isinstance(gold, dict):
            gd = gold.get('final_diagnosis', gold.get('gold_standard_diagnosis', gold.get('diagnosis', ''))) or ''
        else:
            gd = str(gold) or ''
        return {'idx': idx, 'gold': gd, 'pred': result.get('final_diagnosis', ''),
                'ranked_diagnoses': result.get('ranked_diagnoses', []), 'time': time.time() - t0}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'idx': idx, 'gold': '', 'pred': f'ERROR: {e}', 'time': time.time() - t0}

new_results = []
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(process, c): c for c in cases}
    for f in as_completed(futures):
        r = f.result()
        new_results.append(r)
        g, p = r['gold'][:40], r['pred'][:40]
        is_match = g.lower().strip()[:30] in p.lower() or p.lower().strip()[:30] in g.lower()
        mark = '✓' if is_match else '✗'
        print(f"  {mark} Case {r['idx']:2d}: gold=\"{g}\" pred=\"{p}\" ({r['time']:.0f}s)")

results.extend(new_results)
results.sort(key=lambda x: x['idx'])

with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

total = len(results)
correct = sum(1 for r in results if r['gold'].lower().strip()[:30] in r['pred'].lower() or r['pred'].lower().strip()[:30] in r['gold'].lower())
wall = sum(r['time'] for r in results)
print(f"\nDone. {total} cases ({len(new_results)} new). Estimated: {correct}/{total} = {correct/total*100:.0f}%, wall: {wall:.0f}s")
print(f"Saved to {out_path}")
