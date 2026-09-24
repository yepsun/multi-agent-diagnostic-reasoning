#!/usr/bin/env python3
"""
run_think_experiment.py

A/B test: Scheme A with thinking ON vs the existing no-think results, on one
group of the MGH dataset. Appends to results/ablation_local/group{N}_A_think.jsonl
(resume-safe) and prints a per-case comparison against group{N}_A.jsonl.

Usage:
    python run_think_experiment.py <group_number> [light]
"""

import os
import sys
import json
import time

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from scheme_a import run_scheme_a
from batch_ablation import check_match
from run_inference import get_current_model

GROUP = sys.argv[1] if len(sys.argv) > 1 else "3"
LIGHT = len(sys.argv) > 2 and sys.argv[2] == "light"
TAG = "A_think_light" if LIGHT else "A_think"
DATASET = os.path.join(script_dir, "..", "data", f"mgh_qa_dataset_group_{GROUP}.json")
OUT_DIR = os.path.join(script_dir, "..", "results", "ablation_local")
OUT_PATH = os.path.join(OUT_DIR, f"group{GROUP}_{TAG}.jsonl")
BASE_PATH = os.path.join(OUT_DIR, f"group{GROUP}_A.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)
done = set()
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH):
        done.add(json.loads(line)["case_id"])

cases = json.load(open(DATASET))
base = {}
if os.path.exists(BASE_PATH):
    for line in open(BASE_PATH):
        r = json.loads(line)
        base[r["case_id"]] = r

gold_of = lambda c: (c.get("A") or {}).get("gold_standard_diagnosis", "")
correct = 0
total = 0
for case in cases:
    cid = case["case_id"]
    if cid in done:
        continue
    gold = gold_of(case)
    start = time.time()
    result = run_scheme_a(case, enable_thinking=True, think_light=LIGHT)
    elapsed = time.time() - start
    match = check_match(gold, result["final_diagnosis"]) if gold else False
    total += 1
    correct += bool(match)
    record = {
        "case_id": cid,
        "gold": gold,
        "final_diagnosis": result["final_diagnosis"],
        "match": match,
        "elapsed_time": elapsed,
        "total_llm_calls": result.get("total_llm_calls"),
        "termination_reason": result.get("termination_reason"),
        "rounds": result.get("rounds"),
        "scheme": TAG,
        "group": GROUP,
        "model": get_current_model(),
    }
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    flag = "✓" if match else "✗"
    print(f"[{TAG}] {flag} {result['final_diagnosis'][:80]} ({elapsed:.0f}s)", flush=True)

# Final tally including previously completed cases
all_recs = [json.loads(line) for line in open(OUT_PATH)] if os.path.exists(OUT_PATH) else []
n = len(all_recs)
c = sum(1 for r in all_recs if r["match"])
print(f"\nGroup {GROUP} A_think: {c}/{n} = {100*c/n if n else 0:.1f}%")
base_recs = list(base.values()) if base else []
if base_recs:
    b = sum(1 for r in base_recs if r["match"])
    print(f"Group {GROUP} A (no-think baseline): {b}/{len(base_recs)}")
    print("\nPer-case flip analysis:")
    for r in all_recs:
        br = base.get(r["case_id"])
        if not br:
            continue
        if r["match"] and not br["match"]:
            print(f"  + FLIP UP  : {r['case_id'][:50]}")
        elif not r["match"] and br["match"]:
            print(f"  - FLIP DOWN: {r['case_id'][:50]}")
