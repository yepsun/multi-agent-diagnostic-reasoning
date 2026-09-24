#!/usr/bin/env python3
"""
Extract tables from MGH case PDFs using pdfplumber and merge them into the
existing JSON datasets as a new 'table_text' field.

Only keeps tables belonging to the case presentation (not answer/discussion).

Strategy (anchored section-boundary):
  1. Scan all PDF pages for the first occurrence of "Differential Diagnosis"
     (or "Anatomical Diagnosis" / "Final Diagnosis" as fallbacks).
  2. Let N = the page number where this section header first appears.
  3. Extract tables from pages 1..N-1 unconditionally.
  4. On page N, split the extracted text at the header; only keep tables
     whose first substantive cell text appears BEFORE the header text.
  5. Tables from page N+1 onward are discarded (answer/discussion).

Usage:
    python3 extract_tables.py                               # all groups
    python3 extract_tables.py --group 1                     # single group
    python3 extract_tables.py --pdf path/to/single.pdf      # single PDF (dry-run)
"""

import os
import sys
import re
import json
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

PDF_DIR = os.path.join(project_root, "MGH 100 Cases")
DATA_DIR = os.path.join(project_root, "data")

_ANSWER_SECTION_HEADERS = [
    "Differential Diagnosis",
    "Clinical Diagnosis",
    "Pathological Discussion",
    "Final Diagnosis",
    "Anatomical Diagnosis",
    "Discussion",
]


def _find_first_answer_page(pdf) -> int:
    """Return 0-indexed page number where answer section first begins.

    Scans all pages for known answer-section headers. Returns the earliest
    page that contains any such header.
    """
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        for header in _ANSWER_SECTION_HEADERS:
            if re.search(r"\b" + re.escape(header) + r"\b", text, re.IGNORECASE):
                return page_num
    return len(pdf.pages)  # default to last page + 1


def _text_position_of(page, needle: str) -> int:
    """Return character index of needle in page's extracted text (0 if not found)."""
    text = page.extract_text() or ""
    idx = text.lower().find(needle.lower())
    return idx if idx >= 0 else len(text)


def _table_first_cell_text(table) -> str:
    """Get the first non-trivial cell text from a table."""
    for row in table:
        for c in row:
            s = str(c).strip() if c else ""
            if len(s) >= 5 and not s.lower().startswith("table"):
                return s.lower()
    return ""


def _extract_tables_from_pdf(pdf_path: str) -> str:
    """Extract case-presentation tables from PDF using anchored boundary."""
    import pdfplumber

    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        answer_page = _find_first_answer_page(pdf)

        for page_num, page in enumerate(pdf.pages):
            if page_num >= answer_page + 1:
                # Page is AFTER the answer section start — skip entirely
                continue

            tables = page.extract_tables() if hasattr(page, "extract_tables") else []
            for table in tables:
                # Filter trivial tables
                if len(table) <= 2:
                    continue
                data_rows = [r for r in table if any(c and len(str(c).strip()) > 1 for c in r)]
                if len(data_rows) <= 2:
                    continue

                # For the answer-page itself, only keep tables that appear BEFORE the answer header
                if page_num == answer_page:
                    header_text = ""
                    for h in _ANSWER_SECTION_HEADERS:
                        if re.search(r"\b" + re.escape(h) + r"\b", page.extract_text() or "", re.IGNORECASE):
                            header_text = h
                            break
                    if header_text:
                        header_pos = _text_position_of(page, header_text)
                        first_cell = _table_first_cell_text(table)
                        if first_cell and header_pos >= 0:
                            # Check table's first cell text position
                            page_text = (page.extract_text() or "").lower()
                            cell_pos = page_text.find(first_cell)
                            if cell_pos < 0 or cell_pos >= header_pos:
                                # Table appears after (or embedded in) answer section — skip
                                continue

                lines = []
                for row in table:
                    cells = [
                        str(c).replace("\n", " ").strip() if c else ""
                        for c in row
                    ]
                    lines.append("| " + " | ".join(cells) + " |")
                num_cols = max((len(row) for row in table), default=1)
                sep = "|" + "|".join([" --- " for _ in range(num_cols)]) + "|"
                lines.insert(1, sep)
                sections.append(f"### Table (Page {page_num + 1})\n" + "\n".join(lines))

    return "\n\n".join(sections)


def process_group(group_id: int) -> dict:
    """Process all PDFs for a given group."""
    dataset_path = os.path.join(DATA_DIR, f"mgh_qa_dataset_group_{group_id}.json")
    if not os.path.exists(dataset_path):
        print(f"[SKIP] Group {group_id} dataset not found: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    prefix_to_case = {}
    for case in cases:
        cid = case.get("case_id", "")
        m = re.match(r"(\d+)_", cid)
        if m:
            prefix_to_case[m.group(1)] = cid

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    results = {}

    for pdf_path in pdfs:
        fname = os.path.basename(pdf_path)
        prefix = re.match(r"(\d+)_", fname)
        prefix_str = prefix.group(1) if prefix else ""
        if prefix_str in prefix_to_case:
            case_id = prefix_to_case[prefix_str]
            print(f"  [{prefix_str}] {case_id[:60]}...")
            table_text = _extract_tables_from_pdf(pdf_path)
            if table_text:
                results[case_id] = table_text
                print(f"    -> {len(table_text)} chars of table data")
            else:
                print(f"    -> no case-relevant tables found")

    return results


def merge_tables_into_dataset(group_id: int, table_map: dict):
    dataset_path = os.path.join(DATA_DIR, f"mgh_qa_dataset_group_{group_id}.json")
    if not os.path.exists(dataset_path):
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    modified = 0
    for case in cases:
        cid = case.get("case_id", "")
        if cid in table_map:
            case["table_text"] = table_map[cid]
            modified += 1

    if modified:
        backup_path = dataset_path.replace(".json", "_backup.json")
        if not os.path.exists(backup_path):
            os.rename(dataset_path, backup_path)
            print(f"  Backup saved: {backup_path}")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"  Updated {modified}/{len(cases)} cases in {dataset_path}")
    else:
        print(f"  No changes for {dataset_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract tables from MGH case PDFs")
    parser.add_argument("--group", type=int, default=None, help="Single group")
    parser.add_argument("--pdf", type=str, default=None, help="Single PDF (dry-run)")
    args = parser.parse_args()

    if args.pdf:
        import pdfplumber

        with pdfplumber.open(args.pdf) as pdf:
            answer_page = _find_first_answer_page(pdf)
            print(f"First answer section appears on page: {answer_page + 1}")
        table_text = _extract_tables_from_pdf(args.pdf)
        print(table_text[:3000] if table_text else "(no tables found)")
        sys.exit(0)

    if args.group:
        groups = [args.group]
    else:
        groups = sorted({
            int(re.search(r"group_(\d+)", f).group(1))
            for f in os.listdir(DATA_DIR)
            if f.startswith("mgh_qa_dataset_group_") and f.endswith(".json")
        })

    for g in groups:
        print(f"\n=== Processing Group {g} ===")
        table_map = process_group(g)
        if table_map:
            merge_tables_into_dataset(g, table_map)
        else:
            print("  No tables extracted for this group.")
