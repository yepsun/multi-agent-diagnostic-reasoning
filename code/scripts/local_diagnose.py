#!/usr/bin/env python3
"""
local_diagnose.py

Run Scheme A (zero-shot CoT), Scheme P (multi-perspective), and Scheme B
(adaptive retrieval) on a real local case with no gold answer.

All LLM calls go to the provider selected by LLM_PROVIDER (e.g. llamacpp);
no DeepSeek judge, no accuracy scoring — results only.

Usage:
    python local_diagnose.py case.txt                 # run all three schemes
    python local_diagnose.py case.txt --schemes A,P   # run a subset
    python local_diagnose.py case.txt --id mycase     # custom case id
    cat case.txt | python local_diagnose.py -         # read from stdin

Output:
    results/local_cases/<case_id>_<timestamp>.json and a printed summary.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from scheme_a import run_scheme_a
from scheme_perspective import run_scheme_perspective
from scheme_b import run_scheme_b
from scheme_pb import run_scheme_pb
from retrieval_module import RetrievalOrchestrator

RESULTS_DIR = os.path.join(script_dir, '..', 'results', 'local_cases')

SCHEME_RUNNERS = {
    'A': run_scheme_a,
    'P': run_scheme_perspective,
    'B': run_scheme_b,
    'PB': run_scheme_pb,
}


def load_case_text(path: str) -> str:
    if path == '-':
        return sys.stdin.read()
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_file', help="Path to the case text file, or '-' for stdin")
    parser.add_argument('--id', dest='case_id', default=None,
                        help='Case identifier (default: input file name)')
    parser.add_argument('--schemes', default='A,P,B',
                        help='Comma-separated schemes to run, e.g. "A,B" (default: A,P,B)')
    parser.add_argument('--skip-retrieval', action='store_true',
                        help="Run Scheme B without network retrieval (diagnosis-only fallback)")
    args = parser.parse_args()

    wanted = [s.strip().upper() for s in args.schemes.split(',') if s.strip()]
    for s in wanted:
        if s not in SCHEME_RUNNERS:
            parser.error(f"Unknown scheme '{s}'. Choose from A, P, B.")

    case_text = load_case_text(args.case_file).strip()
    if not case_text:
        print("Error: case text is empty.", file=sys.stderr)
        sys.exit(1)

    case_id = args.case_id or (
        os.path.splitext(os.path.basename(args.case_file))[0]
        if args.case_file != '-' else 'stdin_case'
    )
    case = {'case_id': case_id, 'Q': case_text}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {'case_id': case_id, 'timestamp': timestamp, 'schemes': {}}

    print(f"=== Local diagnosis on case: {case_id} ===")
    print(f"Schemes to run: {', '.join(wanted)}\n")

    orchestrator = None
    if 'B' in wanted and not args.skip_retrieval:
        try:
            orchestrator = RetrievalOrchestrator()
            print("[init] RetrievalOrchestrator ready (PubMed/Europe PMC available).")
        except Exception as exc:
            print(f"[init] Retrieval unavailable ({exc}); Scheme B will run without network retrieval.")
            orchestrator = None

    for scheme in wanted:
        print(f"\n--- Scheme {scheme} running... ---")
        start = time.time()
        try:
            if scheme == 'B' and orchestrator is None:
                result = run_scheme_b(case, orchestrator=None, verifier=None)
            else:
                result = SCHEME_RUNNERS[scheme](case)
        except Exception as exc:
            print(f"Scheme {scheme} failed: {exc}")
            results['schemes'][scheme] = {'error': str(exc)}
            continue
        elapsed = time.time() - start
        result['elapsed_seconds'] = round(elapsed, 1)
        results['schemes'][scheme] = result

        diagnosis = (result.get('final_diagnosis')
                     or result.get('revised_diagnosis')
                     or result.get('diagnosis')
                     or '(no diagnosis parsed)')
        confidence = result.get('final_confidence', result.get('confidence_score'))
        print(f"Scheme {scheme} done in {elapsed:.0f}s")
        print(f"  Diagnosis  : {diagnosis}")
        if confidence is not None:
            print(f"  Confidence : {confidence}")
        ranked = result.get('ranked_diagnoses') or []
        if ranked:
            print(f"  Differentials: {'; '.join(str(r) for r in ranked[:5])}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{case_id}_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull results (including reasoning traces) saved to: {os.path.abspath(out_path)}")

    print("\n=== Summary ===")
    for scheme in wanted:
        r = results['schemes'].get(scheme, {})
        if 'error' in r:
            print(f"Scheme {scheme}: ERROR - {r['error']}")
        else:
            d = (r.get('final_diagnosis') or r.get('revised_diagnosis')
                 or r.get('diagnosis') or '(no diagnosis parsed)')
            print(f"Scheme {scheme}: {d}")


if __name__ == '__main__':
    main()
