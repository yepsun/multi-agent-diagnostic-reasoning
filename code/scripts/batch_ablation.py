#!/usr/bin/env python3
"""
Ablation batch test for Scheme A and Scheme B.

Usage:
    python batch_ablation.py <group_number>
    python batch_ablation.py all

Output:
    results/ablation/group{N}_{scheme}.jsonl      append-only incremental log
    results/ablation/group{N}_{scheme}_{timestamp}.json   legacy export

The JSONL log is append-only and doubles as a checkpoint: case_ids already
present in the log are skipped on re-runs (resume-safe, no repeated LLM
calls). The timestamped JSON export is rebuilt from the log and keeps the
exact structure consumed by compare_results.py and generate_ablation_report.py.
"""

import os
import sys
import json
import re
import time
import argparse
import unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from scheme_a import run_scheme_a
from scheme_b import run_scheme_b
from scheme_perspective import run_scheme_perspective
from retrieval_module import RetrievalOrchestrator


from run_inference import call_llm, call_llm_judge


# ---------------------------------------------------------------------------
# Append-only JSONL result log + resume support.
#
# Each (group, scheme) pair owns one JSONL file: group{N}_{scheme}.jsonl.
# Every case result is appended as a single JSON line (never overwritten).
# On startup we read the log to learn which case_ids already finished so a
# re-run can skip them (no repeated LLM token spend). The full legacy JSON
# structure expected by compare_results.py / generate_ablation_report.py is
# still produced on demand by export_json().
# ---------------------------------------------------------------------------

_APPEND_LOCK = threading.Lock()
_META_FIELDS = frozenset({"scheme", "group", "timestamp"})


def log_path(output_dir, group_num, scheme):
    """Path of the append-only JSONL log for a (group, scheme) pair."""
    return os.path.join(output_dir, f"group{group_num}_{scheme}.jsonl")


def load_completed(output_dir, group_num, scheme):
    """Return the set of case_ids already logged for (group, scheme).

    Used for checkpoint/resume: any case_id present here is skipped on a
    re-run. Malformed / partial trailing lines are ignored so a crash mid-
    write never blocks resumption.
    """
    path = log_path(output_dir, group_num, scheme)
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = rec.get("case_id")
            if case_id:
                completed.add(case_id)
    return completed


def append_result(output_dir, group_num, scheme, result):
    """Append one case result to the JSONL log (append-only, thread-safe).

    The result dict is copied and tagged with scheme/group/timestamp so the
    log is self-describing; the original consumer-facing fields are untouched.
    """
    os.makedirs(output_dir, exist_ok=True)
    record = dict(result)
    record["scheme"] = scheme
    record["group"] = group_num
    record["timestamp"] = datetime.now().isoformat()
    line = json.dumps(record, ensure_ascii=False)
    with _APPEND_LOCK:
        with open(log_path(output_dir, group_num, scheme), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def export_json(output_dir, group_num, scheme):
    """Rebuild the legacy per-scheme JSON file from the JSONL log.

    The emitted structure is identical to the historical output of
    save_results(): {scheme, group, results: [...], summary: {...}}, so
    compare_results.py and generate_ablation_report.py keep working.
    """
    path = log_path(output_dir, group_num, scheme)
    results = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("scheme") != scheme or rec.get("group") != group_num:
                    continue
                result = {k: v for k, v in rec.items() if k not in _META_FIELDS}
                results.append(result)

    matches = sum(1 for r in results if r.get("match"))
    errors = sum(1 for r in results if r.get("error"))
    total_llm = sum(r.get("total_llm_calls", 0) for r in results)
    total_pubmed = sum(r.get("total_pubmed_queries", 0) for r in results)
    total_time = sum(r.get("elapsed_time", 0) for r in results)

    data = {
        "scheme": scheme,
        "group": group_num,
        "results": results,
        "summary": {
            "scheme": scheme,
            "group": group_num,
            "total_cases": len(results),
            "matches": matches,
            "errors": errors,
            "accuracy": matches / len(results) if results else 0,
            "total_llm_calls": total_llm,
            "total_pubmed_queries": total_pubmed,
            "total_time": total_time,
            "avg_time": total_time / len(results) if results else 0,
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"group{group_num}_{scheme}_{timestamp}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_file


_UNICODE_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015"), " ")

# Full-phrase synonyms observed to trip the semantic judge on equivalent
# wordings. Checked bidirectionally on normalized text; keep this list small
# and unambiguous (each entry is a true clinical synonym, not a subtype).
_ALIAS_PAIRS = [
    ("tuberculous enteritis", "intestinal tuberculosis"),
    ("tuberculous enteritis", "gastrointestinal tuberculosis"),
    ("tuberculous enteritis", "gi tuberculosis"),
]

# Filler words that carry no discriminating power for token-overlap matching.
_STOPWORDS = {
    "of", "the", "with", "and", "due", "to", "a", "an", "in", "by", "for",
    "from", "associated", "secondary", "complicated", "resulting",
}


def _normalize_dx(text: str) -> str:
    """Normalize a diagnosis string: unify dashes/quotes/accents/punctuation.

    En-dashes and hyphens become spaces ("Tay–Sachs" -> "tay sachs"), curly
    quotes become straight, accented letters fold to ascii (ü -> u), and
    punctuation becomes whitespace. Parentheses become spaces so their
    content stays available as tokens for the overlap check.
    """
    t = (text or "").lower().translate(_UNICODE_DASHES)
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[.,;:()\[\]\"']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _significant_tokens(text: str) -> set:
    """Stemmed, stopword-filtered token set for overlap matching."""

    def _stem(w: str) -> str:
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]
        return w

    return {_stem(w) for w in _normalize_dx(text).split()
            if w not in _STOPWORDS and len(w) > 1}


def _strip_parens(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def _negation_conflict(g: str, d: str) -> bool:
    """True when exactly one side carries a negation prefix ("non ...").

    Guards the containment/Jaccard branches against pairs like
    "non-small cell lung cancer" vs "small cell lung cancer", where the
    negated entity contains the non-negated one as a literal substring.
    """
    def _neg(t: str) -> bool:
        return bool(re.search(r"\bnon\b", t))
    return _neg(g) != _neg(d)


def _heuristic_match(gold: str, diagnosis: str) -> bool:
    """Fast string-level heuristic for obvious matches.

    Deliberately strict — four checks, all conservative; anything ambiguous
    is routed through the semantic LLM judge, which is the authority:

    1. Normalized substring containment (handles en-dashes, curly quotes,
       accents, and punctuation variants of the same wording).
    2. Parenthetical-stripped containment (qualifier dropped on one side).
    3. Curated full-phrase clinical aliases (_ALIAS_PAIRS).
    4. Guarded token-overlap (stemmed Jaccard >= 0.55, both sides >= 3
       significant tokens, negation-consistent). 0.55 sits above the
       "same entity, reworded" band (~0.55-0.8 in practice) and below
       confusable near-pairs (acute lymphoblastic vs myeloid leukemia = 0.50).

    The previous shared-words branch (">=2 significant common words") produced
    systematic false positives for clinically unrelated diagnoses that merely
    share words (e.g., gold "bipolar I disorder ..." vs a breast-cancer
    diagnosis that happens to mention "major depressive episode" and "bipolar
    disorder") — that failure mode is why the Jaccard branch carries a hard
    threshold and a negation guard instead of a word-count rule.
    """
    if not gold or not diagnosis:
        return False

    g = _normalize_dx(gold)
    d = _normalize_dx(diagnosis)
    if not g or not d:
        return False

    # 1. plain containment on normalized text
    if not _negation_conflict(g, d) and (g in d or d in g):
        return True

    # 2. containment ignoring parenthetical qualifiers
    gs, ds = _strip_parens(g), _strip_parens(d)
    if not _negation_conflict(gs, ds) and (gs in ds or ds in gs):
        return True

    # 3. curated aliases
    for a, b in _ALIAS_PAIRS:
        if (a in gs and b in ds) or (b in gs and a in ds):
            return True

    # 4. stemmed token-overlap (Jaccard) with negation guard
    if not _negation_conflict(g, d):
        tg, td = _significant_tokens(gold), _significant_tokens(diagnosis)
        if len(tg) >= 3 and len(td) >= 3:
            j = len(tg & td) / len(tg | td)
            if j >= 0.55:
                return True

    return False


def _semantic_match(gold: str, diagnosis: str) -> bool:
    """Use a lightweight LLM call to judge clinical equivalence."""
    prompt = f"""You are a medical expert evaluating whether two diagnostic statements refer to the same or clinically equivalent diagnosis.

Gold standard diagnosis:
{gold}

Predicted diagnosis:
{diagnosis}

Are these clinically equivalent? Answer ONLY with YES or NO.

Rules:
- YES if they refer to the same disease, syndrome, or pathophysiologic entity, even if one is more specific than the other.
- YES if one is a subtype or complication of the other that captures the core diagnosis (e.g., "influenza A pneumonia with secondary bacterial infection" vs "viral pneumonia complicated by a bacterial super infection").
- NO if they are different diseases or if the predicted diagnosis misses the essential entity described in the gold standard.
"""
    try:
        # Judge is provider-decoupled: it always uses the DeepSeek API
        # (LLM_JUDGE_PROVIDER) so a local llama.cpp inference run is not slowed
        # by a local judge call. disable_thinking is applied inside
        # call_llm_judge (deepseek-flash is a reasoner; the YES/NO verdict
        # must land in content, not reasoning_content).
        raw, _ = call_llm_judge(prompt)
        return raw.strip().upper().startswith("YES")
    except Exception as e:
        print(f"[check_match] Semantic match failed: {e}; falling back to heuristic.")
        return False


def check_match(gold: str, diagnosis: str) -> bool:
    """Check if gold standard matches diagnosis.

    Uses a fast heuristic first, then a lightweight LLM-based semantic check
    for cases where the wording differs but the clinical meaning is equivalent.
    """
    if _heuristic_match(gold, diagnosis):
        return True

    # Only spend an LLM call when the heuristic is ambiguous.
    return _semantic_match(gold, diagnosis)


def load_group_dataset(group_num):
    dataset_path = f"../data/mgh_qa_dataset_group_{group_num}.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        return None
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_main_dataset():
    """Load the full main dataset (../data/mgh_qa_dataset.json).

    Same consumer fields as the per-group files: case_id, Q,
    A.gold_standard_diagnosis.
    """
    dataset_path = "../data/mgh_qa_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        return None
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(group_num):
    """Load a dataset by group label (int 1-10 or the string 'main')."""
    if group_num == "main":
        return load_main_dataset()
    return load_group_dataset(group_num)


def run_scheme_on_case(case, scheme_name, orchestrator=None):
    """Run a single scheme on a single case."""
    case_id = case["case_id"]
    gold = case["A"].get("gold_standard_diagnosis", "N/A")

    start_time = time.time()
    result = None
    error = None

    try:
        if scheme_name == "A":
            result = run_scheme_a(case)
        elif scheme_name == "A_think":
            result = run_scheme_a(case, enable_thinking=True)
        elif scheme_name == "B":
            result = run_scheme_b(case, orchestrator)
        elif scheme_name == "P":
            result = run_scheme_perspective(case)
        elif scheme_name == "P_think":
            result = run_scheme_perspective(case, enable_thinking=True)
        else:
            raise ValueError(f"Unknown scheme: {scheme_name}")
    except Exception as e:
        error = str(e)
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time

    if error:
        return {
            "case_id": case_id,
            "gold": gold,
            "final_diagnosis": "ERROR",
            "match": False,
            "elapsed_time": elapsed,
            "error": error,
            "total_llm_calls": 0,
            "total_pubmed_queries": 0,
        }

    final_diagnosis = result.get("final_diagnosis", "N/A")
    match = check_match(gold, final_diagnosis)

    return {
        "case_id": case_id,
        "gold": gold,
        "final_diagnosis": final_diagnosis,
        "match": match,
        "elapsed_time": elapsed,
        "total_llm_calls": result.get("total_llm_calls", 0),
        "total_pubmed_queries": result.get("total_pubmed_queries", 0),
        "termination_reason": result.get("termination_reason", ""),
        "rounds": result.get("rounds", []),
    }


def test_group(group_num, schemes, orchestrator, concurrency=1,
               output_dir="../results/ablation"):
    """Run all selected schemes on all cases in a group.

    Append-only JSONL logging + resume: already-completed (group, scheme,
    case_id) triplets found in the JSONL log are skipped without re-running.
    """
    dataset = load_dataset(group_num)
    if not dataset:
        return None

    print(f"\n{'='*70}")
    print(f"Testing Group {group_num} | Schemes: {', '.join(schemes)} | Concurrency: {concurrency}")
    print(f"{'='*70}")
    print(f"Total cases: {len(dataset)}")

    # Checkpoint: which case_ids already finished per scheme?
    completed = {
        scheme: load_completed(output_dir, group_num, scheme)
        for scheme in schemes
    }

    group_results = {}
    for scheme in schemes:
        group_results[scheme] = {
            "scheme": scheme,
            "group": group_num,
            "results": [],
        }

    for i, case in enumerate(dataset):
        case_id = case["case_id"]
        print(f"\n--- Case {i+1}/{len(dataset)}: {case_id[:60]}... ---")
        for scheme in schemes:
            if case_id in completed[scheme]:
                print(f"  [{scheme}] SKIP (already completed in log)")
                continue
            res = run_scheme_on_case(case, scheme, orchestrator)
            append_result(output_dir, group_num, scheme, res)
            group_results[scheme]["results"].append(res)
            status = "✓" if res["match"] else "✗"
            if res.get("error"):
                status = "E"
            print(f"  [{scheme}] {status} {res['final_diagnosis'][:80]}... "
                  f"(LLM={res['total_llm_calls']}, PubMed={res['total_pubmed_queries']})")

    # Summaries
    summaries = []
    for scheme in schemes:
        results = group_results[scheme]["results"]
        matches = sum(1 for r in results if r["match"])
        errors = sum(1 for r in results if r.get("error"))
        total_llm = sum(r["total_llm_calls"] for r in results)
        total_pubmed = sum(r["total_pubmed_queries"] for r in results)
        total_time = sum(r["elapsed_time"] for r in results)

        summary = {
            "scheme": scheme,
            "group": group_num,
            "total_cases": len(results),
            "matches": matches,
            "errors": errors,
            "accuracy": matches / len(results) if results else 0,
            "total_llm_calls": total_llm,
            "total_pubmed_queries": total_pubmed,
            "total_time": total_time,
            "avg_time": total_time / len(results) if results else 0,
        }
        summaries.append(summary)
        group_results[scheme]["summary"] = summary

    print(f"\n{'='*70}")
    print(f"GROUP {group_num} SUMMARY")
    print(f"{'='*70}")
    for s in summaries:
        print(f"  Scheme {s['scheme']}: {s['matches']}/{s['total_cases']} = "
              f"{s['accuracy']*100:.1f}% | "
              f"LLM calls={s['total_llm_calls']} | PubMed={s['total_pubmed_queries']}")

    return group_results


def run_scheme_a_batch(dataset, concurrency=10, output_dir="../results/ablation",
                       group_num=None, enable_thinking=False, scheme="A"):
    """Run scheme A on all cases concurrently (JSONL log + resume aware).

    `scheme` names the log file (e.g. "A" or "A_think") so thinking and
    non-thinking results don't collide; `enable_thinking` toggles the
    thinking-enabled run_scheme_a path.
    """
    completed = set()
    if group_num is not None:
        completed = load_completed(output_dir, group_num, scheme)
    pending = [case for case in dataset
               if case.get("case_id") not in completed]
    if len(pending) < len(dataset):
        print(f"  Skipping {len(dataset) - len(pending)} already-completed cases "
              f"for scheme {scheme} (resume).")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_scheme_a, case, enable_thinking=enable_thinking): case
                   for case in pending}
        for i, future in enumerate(as_completed(futures)):
            case = futures[future]
            case_id = case.get("case_id", f"case_{i}")
            try:
                result = future.result()
                gold = case.get("A", {}).get("gold_standard_diagnosis", "N/A")
                final_dx = result.get("final_diagnosis", "N/A")
                match = check_match(gold, final_dx)
                status = "✓" if match else "✗"
                print(f"  {status} [{case_id[:30]}] {final_dx[:60]}")
                res = {
                    "case_id": case_id,
                    "gold": gold,
                    "final_diagnosis": final_dx,
                    "match": match,
                    "elapsed_time": result.get("total_time_seconds", 0),
                    "total_llm_calls": result.get("total_llm_calls", 0),
                }
                if group_num is not None:
                    append_result(output_dir, group_num, scheme, res)
                results.append(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  E [{case_id[:30]}] {e}")
                res = {
                    "case_id": case_id,
                    "gold": case.get("A", {}).get("gold_standard_diagnosis", "N/A"),
                    "final_diagnosis": "ERROR",
                    "match": False,
                    "error": str(e),
                }
                if group_num is not None:
                    append_result(output_dir, group_num, scheme, res)
                results.append(res)
    return results


def save_results(group_results, output_dir="../results/ablation"):
    """Export JSONL logs back to the legacy per-scheme JSON files.

    Each file is rebuilt from the append-only log, so exports reflect ALL
    logged runs (including results from previous invocations) and match the
    structure consumed by compare_results.py / generate_ablation_report.py.
    """
    if not group_results:
        return

    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for scheme, data in group_results.items():
        output_file = export_json(output_dir, data["group"], scheme)
        saved.append(output_file)
        print(f"  Saved: {output_file}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Ablation batch test")
    parser.add_argument("group", help="Group number (1-10), 'all', or 'main'")
    parser.add_argument(
        "--schemes",
        default="A,B",
        help="Comma-separated schemes to run (default: A,B)",
    )
    parser.add_argument("--output", default="../results/ablation", help="Output dir")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent workers for scheme A (default: 1)")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",")]
    orchestrator = RetrievalOrchestrator()

    if args.group.lower() == "all":
        groups = range(1, 11)
    elif args.group.lower() == "main":
        # Full main dataset (41 cases) — 'all' stays group 1-10 only.
        groups = ["main"]
    else:
        try:
            g = int(args.group)
            if g < 1 or g > 10:
                print("Error: Group number must be between 1 and 10")
                return
            groups = [g]
        except ValueError:
            print("Error: Please specify a group number (1-10), 'all', or 'main'")
            return

    for group_num in groups:
        dataset = load_dataset(group_num)
        if not dataset:
            continue

        group_schemes = schemes
        if args.concurrency > 1:
            # Concurrent batch path for scheme A and its thinking variant.
            # Each is stripped from the serial list so it is not re-run below.
            for batch_scheme in ("A", "A_think"):
                if batch_scheme not in schemes:
                    continue
                enable_thinking = batch_scheme == "A_think"
                print(f"\n{'='*70}")
                print(f"Group {group_num} | Scheme {batch_scheme} | Concurrency: {args.concurrency}")
                print(f"{'='*70}")
                print(f"Total cases: {len(dataset)}")
                results = run_scheme_a_batch(dataset, concurrency=args.concurrency,
                                             output_dir=args.output, group_num=group_num,
                                             enable_thinking=enable_thinking,
                                             scheme=batch_scheme)
                matches = sum(1 for r in results if r.get("match"))
                print(f"\n  Scheme {batch_scheme}: {matches}/{len(results)} = "
                      f"{matches/len(results)*100:.1f}%")
                save_results({
                    batch_scheme: {"scheme": batch_scheme, "group": group_num,
                                   "results": results,
                                   "summary": {"scheme": batch_scheme, "group": group_num,
                                               "total_cases": len(results), "matches": matches,
                                               "accuracy": matches/len(results) if results else 0}}
                }, args.output)
                # Local per-group list only: the outer `schemes` must stay intact so
                # the concurrent batch path also runs on groups 2+.
                group_schemes = [s for s in group_schemes if s != batch_scheme]

        if group_schemes:
            group_results = test_group(group_num, group_schemes, orchestrator,
                                       concurrency=args.concurrency,
                                       output_dir=args.output)
            if group_results:
                save_results(group_results, args.output)


if __name__ == "__main__":
    main()
