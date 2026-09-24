#!/usr/bin/env python3
"""
Re-judge existing JSONL result logs with the current check_match (the semantic
LLM judge is the authority). This tool does NOT regenerate any diagnosis: no
scheme runs happen, only judge calls. It recalibrates recorded accuracies after
the match heuristic was tightened.

Usage:
    python3 rejudge_results.py <path-to-jsonl> [<more-paths>...]

    e.g. python3 rejudge_results.py results/ablation/groupmain_A.jsonl \
                                results/ablation/group1_B.jsonl

For each input file it writes <original-stem>.rejudged.jsonl next to it: one
line per original record with "match" replaced by the rejudged verdict and a
"rejudge_source" field added ("heuristic" when the fast substring path decided
it, "semantic" when the LLM judge was called, "skipped" for records whose
final_diagnosis is empty or "ERROR", or "error" when the judge call failed).
Original files are never overwritten.

Idempotent resume: if <original-stem>.rejudged.jsonl already exists, records
whose case_id is already present are skipped (only the missing ones are
judged) and the results are merged.

Records are rejudged with a ThreadPoolExecutor (max_workers=8). Per-record
exceptions mark the record (match=False, rejudge_source="error") and continue.
LLM-call progress is printed to stderr so stdout stays parseable.
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from batch_ablation import check_match, _heuristic_match  # noqa: E402

MAX_WORKERS = 8
PROGRESS_EVERY = 20


def _rejudged_path(path):
    """<original-stem>.rejudged.jsonl next to the original file."""
    stem = path[:-6] if path.endswith(".jsonl") else path
    return stem + ".rejudged.jsonl"


def _record_is_judgeable(record):
    """Records with empty or ERROR final_diagnosis are reported as-is."""
    dx = record.get("final_diagnosis") or ""
    dx = dx.strip()
    return bool(dx) and dx.upper() != "ERROR"


def _judge(record):
    """Rejudge a single record.

    Returns (new_record, changed_line_or_None). Always calls check_match for
    the authoritative verdict; the source is "heuristic" when the fast
    substring path decided it (check_match short-circuits there, no LLM call)
    and "semantic" otherwise. Exceptions become match=False with
    rejudge_source="error" (and no changed-line entry).
    """
    gold = record.get("gold") or ""
    dx = record.get("final_diagnosis") or ""
    recorded = bool(record.get("match"))
    try:
        verdict = bool(check_match(gold, dx))
        source = "heuristic" if _heuristic_match(gold, dx) else "semantic"
    except Exception as e:
        verdict, source = False, "error"
    new_record = dict(record)
    new_record["match"] = verdict
    new_record["rejudge_source"] = source
    changed = None
    if source != "error" and recorded != verdict:
        changed = (str(record.get("case_id") or ""), recorded, verdict, gold, dx)
    return new_record, changed


def _load_done(path):
    """case_id -> record map from an existing .rejudged.jsonl (for resume)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("case_id")
            if cid is not None:
                done[cid] = rec
    return done


def rejudge_file(path):
    """Rejudge one JSONL file. Returns (summary_dict, changed_lines)."""
    stem = _rejudged_path(path)
    existing = _load_done(stem)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[rejudge] skipped malformed line in {path}", file=sys.stderr)

    total = len(records)
    recorded_matches = sum(1 for r in records if bool(r.get("match")))

    # Seed the output with existing rejudged records (resume) and tag the
    # non-judgeable records as skipped; collect the rest for judging. Every
    # index gets filled before the file is written (existing / skipped /
    # pending-verified below).
    out: list = [None] * total
    pending = []
    for i, r in enumerate(records):
        cid = r.get("case_id")
        if cid in existing:
            out[i] = existing[cid]
        elif _record_is_judgeable(r):
            pending.append(i)
        else:
            skipped = dict(r)
            skipped["rejudge_source"] = "skipped"
            out[i] = skipped

    changed = []
    judged = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_judge, records[i]): i for i in pending}
        for fut in as_completed(futures):
            i = futures[fut]
            new_record, changed_line = fut.result()
            out[i] = new_record
            if changed_line:
                changed.append(changed_line)
            judged += 1
            if judged % PROGRESS_EVERY == 0:
                print(
                    f"[rejudge] {os.path.basename(path)}: judged {judged}/{len(pending)}",
                    file=sys.stderr,
                )

    with open(stem, "w", encoding="utf-8") as f:
        for rec in out:
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rejudged_matches = sum(1 for r in out if r is not None and bool(r.get("match")))
    summary = {
        "path": path,
        "total": total,
        "recorded_matches": recorded_matches,
        "rejudged_matches": rejudged_matches,
        "delta": rejudged_matches - recorded_matches,
    }
    return summary, changed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python3 rejudge_results.py <path-to-jsonl> [<more-paths>...]")
        return 1
    rc = 0
    for path in argv:
        if not os.path.exists(path):
            print(f"[rejudge] error: file not found: {path}", file=sys.stderr)
            rc = 1
            continue
        summary, changed = rejudge_file(path)
        print(f"=== {summary['path']} ===")
        print(
            f"  total={summary['total']} "
            f"recorded_matches={summary['recorded_matches']} "
            f"rejudged_matches={summary['rejudged_matches']} "
            f"delta={summary['delta']:+d}"
        )
        for cid, recorded, new, gold, dx in changed:
            print(
                f"  CHANGED {cid[:50]} | {recorded} -> {new} | "
                f"gold: {gold[:50]} | dx: {dx[:60]}"
            )
        print(f"  wrote {_rejudged_path(path)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
