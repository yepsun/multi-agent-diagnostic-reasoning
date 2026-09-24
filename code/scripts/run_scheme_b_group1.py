#!/usr/bin/env python3
"""Run scheme B on all group 1 cases and summarize results."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scheme_b import run_scheme_b

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_path = os.path.join(project_root, "data", "mgh_qa_dataset_group_1.json")
with open(dataset_path) as f:
    dataset = json.load(f)

results = []
for i, case in enumerate(dataset):
    cid = case.get("case_id", f"case_{i}")[:60]
    print(f"\n[{i+1}/{len(dataset)}] {cid}")
    try:
        result = run_scheme_b(case)
        final_dx = result["final_diagnosis"]
        confidence = result["final_confidence"]
        sources = result["rounds"][0]["retrieval_sources_count"]
        llm_calls = result["total_llm_calls"]
        time_s = result["total_time_seconds"]
        print(f"  Final: {final_dx[:80]}")
        print(f"  Conf: {confidence} | Sources: {sources} | LLM: {llm_calls} | Time: {time_s:.0f}s")
        results.append({
            "case_id": case.get("case_id", ""),
            "final_diagnosis": final_dx,
            "confidence": confidence,
            "sources": sources,
            "llm_calls": llm_calls,
            "time": time_s,
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "case_id": case.get("case_id", ""),
            "error": str(e),
        })

# Save results
out_path = os.path.join(os.path.dirname(dataset_path), "scheme_b_group_1_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {out_path}")
