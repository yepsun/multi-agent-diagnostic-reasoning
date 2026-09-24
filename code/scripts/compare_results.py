#!/usr/bin/env python3
"""
Compare ablation result files across schemes.

Usage:
    python compare_results.py ../results/ablation/group1_A_*.json ../results/ablation/group1_P_*.json

Or compare all schemes for a group:
    python compare_results.py ../results/ablation/group1_*.json
"""

import json
import sys
import glob
from collections import defaultdict


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(scheme_results):
    """
    scheme_results: dict[str, dict] mapping scheme name to loaded result dict.
    """
    # Build case_id -> {scheme: result}
    cases = defaultdict(dict)
    scheme_names = sorted(scheme_results.keys())

    for scheme, data in scheme_results.items():
        for r in data.get("results", []):
            cases[r["case_id"]][scheme] = r

    print("=" * 80)
    print("ABLATION COMPARISON")
    print("=" * 80)

    # Overall summaries
    print("\nOverall Accuracy:")
    for scheme in scheme_names:
        data = scheme_results[scheme]
        summary = data.get("summary", {})
        acc = summary.get("accuracy", 0) * 100
        matches = summary.get("matches", 0)
        total = summary.get("total_cases", 0)
        llm = summary.get("total_llm_calls", 0)
        pubmed = summary.get("total_pubmed_queries", 0)
        time_s = summary.get("total_time", 0)
        print(f"  {scheme:12s}: {matches}/{total} = {acc:5.1f}% | "
              f"LLM={llm:3d} | PubMed={pubmed:3d} | Time={time_s:6.1f}s")

    # Per-case comparison
    print("\nPer-Case Results:")
    print(f"{'Case ID':<60s} | " + " | ".join(f"{s:>3s}" for s in scheme_names))
    print("-" * (60 + 4 + len(scheme_names) * 6))

    for case_id in sorted(cases.keys()):
        row_results = cases[case_id]
        marks = []
        for scheme in scheme_names:
            if scheme in row_results:
                r = row_results[scheme]
                if r.get("error"):
                    marks.append("ERR")
                elif r.get("match"):
                    marks.append(" ✓ ")
                else:
                    marks.append(" ✗ ")
            else:
                marks.append(" - ")
        display_id = case_id[:57] + "..." if len(case_id) > 60 else case_id
        print(f"{display_id:<60s} | " + " | ".join(marks))

    # Agreement / disagreement matrix
    print("\nAgreement Analysis:")
    if len(scheme_names) >= 2:
        for i, s1 in enumerate(scheme_names):
            for s2 in scheme_names[i+1:]:
                both_correct = 0
                both_wrong = 0
                s1_correct_only = 0
                s2_correct_only = 0
                for case_id, res in cases.items():
                    r1 = res.get(s1, {})
                    r2 = res.get(s2, {})
                    if r1.get("error") or r2.get("error"):
                        continue
                    m1 = r1.get("match", False)
                    m2 = r2.get("match", False)
                    if m1 and m2:
                        both_correct += 1
                    elif not m1 and not m2:
                        both_wrong += 1
                    elif m1:
                        s1_correct_only += 1
                    else:
                        s2_correct_only += 1
                print(f"  {s1} vs {s2}:")
                print(f"    Both correct: {both_correct}")
                print(f"    Both wrong:   {both_wrong}")
                print(f"    Only {s1} correct: {s1_correct_only}")
                print(f"    Only {s2} correct: {s2_correct_only}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: compare all schemes for group1 in ../results/ablation
        paths = sorted(glob.glob("../results/ablation/group1_*.json"))
    else:
        paths = sys.argv[1:]

    if not paths:
        print("No result files found")
        sys.exit(1)

    scheme_results = {}
    for p in paths:
        data = load_result(p)
        scheme = data.get("scheme", "unknown")
        scheme_results[scheme] = data

    compare(scheme_results)
