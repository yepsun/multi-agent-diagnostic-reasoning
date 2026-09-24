"""
Case text extraction and preprocessing module.
Optimized for NEJM CPC case presentations.
Borrows structuring approach from Scheme B but without retrieval dependency.

Core idea: clean the raw case text and organize key clinical information
so the model's diagnostic reasoning starts from a clearer signal.
"""

import re
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Noise patterns to remove
# =============================================================================
#
# IMPORTANT: All multi-line patterns use line-based matching (^(?!\n).*)*
# instead of .*? with re.DOTALL to avoid catastrophic backtracking on
# large case texts (~30K chars). The line-based approach is O(n).
#

# NEJM copyright / editorial boilerplate
# NOTE: Only remove the copyright line itself, not content that follows.
_COPYRIGHT_PATTERN = re.compile(
    r'^Copyright © \d{4} Massachusetts Medical Society\..*?\n',
    re.MULTILINE
)

# NEJM "Founded by Richard C. Cabot" editor list
# NOTE: This must be careful not to match beyond the editor list into clinical content.
# The editor list typically ends with "Production Editors" or similar.
_EDITORIAL_BOILERPLATE = re.compile(
    r'^Founded by Richard\s*C\.\s*Cabot.*?Production Editors.*?\n',
    re.MULTILINE | re.DOTALL | re.IGNORECASE
)

# Figure / table legends — match "Figure X" / "Table X" through to blank line
# NOTE: In NEJM CPC cases, Figure and Table legends contain CRITICAL clinical
# information (imaging findings, lab results). We should NOT remove them.
# Only remove pure figure captions that don't contain clinical data.
# _FIGURE_LEGEND = re.compile(
#     r'^(?:Figure|Table)\s+\d+.*(?:\n(?!\n).*)*',
#     re.MULTILINE | re.IGNORECASE
# )

# Instead, only remove lines that are clearly just captions without clinical data
# (e.g., "Figure 1. ECG and Imaging Studies.") - short captions without clinical details
_FIGURE_CAPTION_ONLY = re.compile(
    r'^(?:Figure|Table)\s+\d+\.\s*[A-Z][^.]{0,50}\.?$',
    re.MULTILINE | re.IGNORECASE
)

# Video references (no clinical content)
_VIDEO_REF = re.compile(
    r'\(see Video[s]?\s+\d[^)]*\)',
    re.IGNORECASE
)

# Footnote markers in tables (e.g., "*\tTo convert the values for...")
_FOOTNOTE_TABLE = re.compile(
    r'^\*\s*\t?To convert the values for.*(?:\n(?!\n).*)*',
    re.MULTILINE | re.IGNORECASE
)

# Reference range boilerplate paragraphs
_REF_RANGE_BOILERPLATE = re.compile(
    r'^Reference values are affected by many variables.*(?:\n(?!\n).*)*',
    re.MULTILINE | re.IGNORECASE
)

# DOI / URL references (single-line, safe)
_DOI_REF = re.compile(
    r'\b(?:DOI|doi|https?://doi\.org|https?://www\.nejm\.org)[^\s]*',
    re.IGNORECASE
)


# =============================================================================
# Main preprocessing
# =============================================================================

def preprocess_case_text(raw_text: str) -> str:
    """Clean and structure raw CPC case text for better LLM comprehension.

    Removes editorial noise, normalizes formatting, and preserves ALL
    clinical information. The output is a cleaner version of the input
    with the same structure.

    NOTE: Quick string guards before each regex to avoid catastrophic
    backtracking on large texts that don't contain the pattern.
    """
    text = raw_text

    # Remove noise patterns — guard with fast substring check first
    if 'Copyright' in text:
        text = _COPYRIGHT_PATTERN.sub('', text)
    if 'Founded by Richard' in text:
        text = _EDITORIAL_BOILERPLATE.sub('', text)
    # NOTE: Figure/Table legends contain critical clinical data in NEJM CPC cases.
    # Only remove very short captions (e.g., "Figure 1. ECG Studies."), keep detailed legends.
    if 'Figure' in text or 'Table' in text:
        text = _FIGURE_CAPTION_ONLY.sub('', text)
    if 'Video' in text:
        text = _VIDEO_REF.sub('', text)
    if 'To convert the values for' in text:
        text = _FOOTNOTE_TABLE.sub('', text)
    if 'Reference values are affected' in text:
        text = _REF_RANGE_BOILERPLATE.sub('', text)
    text = _DOI_REF.sub('', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    return text


# =============================================================================
# Structured case breakdown (reference Scheme B's approach)
# =============================================================================

def extract_case_sections(raw_text: str) -> Dict[str, str]:
    """Break down a case into its key clinical sections.

    Returns dict with keys like: presentation, labs, imaging, history, exam.
    Falls back gracefully when sections can't be found.
    """
    cleaned = preprocess_case_text(raw_text)
    sections = {}

    # 1. Presentation / chief complaint (first ~300 chars)
    first_para = cleaned.split('\n\n')[0] if '\n\n' in cleaned else cleaned[:500]
    sections['presentation'] = first_para.strip()

    # 2. Lab values section — find the main lab table block
    lab_block = _extract_lab_block(cleaned)
    if lab_block:
        sections['labs'] = lab_block

    # 3. Imaging findings
    imaging = _extract_imaging_findings(cleaned)
    if imaging:
        sections['imaging'] = imaging

    # 4. History and exposures (often found in mid-case paragraphs)
    history = _extract_history_exposures(cleaned)
    if history:
        sections['history'] = history

    # 5. Physical exam findings
    exam = _extract_exam_findings(cleaned)
    if exam:
        sections['exam'] = exam

    return sections


def _extract_lab_block(text: str) -> Optional[str]:
    """Extract the main lab values section."""
    # Pattern 1: Variable table with reference ranges
    patterns = [
        r'(Variable\s*\n.*?(?:Reference Range|Ref Range|Normal Range).*?)(?=\n\n[A-Z]|\Z)',
        r'(Laboratory.*?(?:Test|Data|Values).*?(?:Reference|Result|Normal).*?)(?=\n\n[A-Z]|\Z)',
        r'(Blood\n.*?(?:Sodium|Potassium|Calcium|Glucose).*?(?:mmol|mg|g).*?)(?=\n\n[A-Z]|\Z)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            block = m.group(1).strip()
            if len(block) > 100:
                return block[:2000]  # cap at 2000 chars
    return None


def _extract_imaging_findings(text: str) -> Optional[str]:
    """Extract key imaging findings as bullet-like statements."""
    sentences = []
    patterns = [
        r'(?:CT|MRI|X-ray|ultrasound|echocardiogram|angiography|PET|radiograph)\s+(?:of\s+)?(?:the\s+)?[^.]*?(?:showed|revealed|demonstrated|identified|confirmed)[^.]*\.',
        r'(?:CT|MRI|X-ray|ultrasound|echocardiogram|angiogram|radiograph).*?(?:was|were)\s+(?:normal|unremarkable|negative|positive|consistent with)[^.]*\.',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            clean = m.strip()
            if clean and clean not in sentences:
                sentences.append(clean)

    if sentences:
        return '\n'.join(sentences[:6])  # max 6 findings
    return None


def _extract_history_exposures(text: str) -> Optional[str]:
    """Extract history, exposure, medication, travel, diet info."""
    # Find the paragraph(s) that discuss history
    history_para = re.search(
        r'(?:Medical history|History of present illness|He had|She had|There was no history of)[^.]*(?:\.(?:[^.]*\.){1,15})',
        text, re.IGNORECASE
    )
    if history_para:
        return history_para.group(0).strip()

    # Fallback: look for exposure keywords
    exposure_match = re.search(
        r'(?:diet|consumption|travel|medication|drug|exposure|occupation|smok|alcohol).*?(?:\.(?:[^.]*\.){1,8})',
        text, re.IGNORECASE
    )
    if exposure_match:
        return exposure_match.group(0).strip()
    return None


def _extract_exam_findings(text: str) -> Optional[str]:
    """Extract physical exam findings."""
    exam_match = re.search(
        r'(?:On examination|Physical examination|Examination revealed|Vital signs were)[^.]*(?:\.(?:[^.]*\.){1,10})',
        text, re.IGNORECASE
    )
    if exam_match:
        return exam_match.group(0).strip()
    return None


# =============================================================================
# Key facts extraction (what Scheme B does with Question Generation)
# =============================================================================

def extract_key_clinical_facts(case_text: str) -> Dict[str, List[str]]:
    """Extract key clinical facts that anchor the diagnosis.

    Returns categorized findings that help narrow the differential,
    similar to what Scheme B's question generation prompts do internally.
    This is a heuristic extraction — the LLM will do deeper reasoning.
    """
    facts = {
        "abnormal_labs": [],
        "imaging_findings": [],
        "exposures": [],
        "negatives": [],  # notable negative findings
    }

    # Abnormal lab patterns
    lab_pats = [
        (r'potassium\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'potassium'),
        (r'sodium\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'sodium'),
        (r'calcium\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'calcium'),
        (r'glucose\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'glucose'),
        (r'creatinine\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'creatinine'),
        (r'hemoglobin\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'hemoglobin'),
        (r'white.cell\s+count\s*(?:was|of)?\s*(\d+\.?\d*)', 'wbc'),
        (r'platelet\s+count\s*(?:was|of)?\s*(\d+\.?\d*)', 'platelet'),
        (r'lactate\s+(?:level|concentration)?\s*(?:was|of)?\s*(\d+\.?\d*)', 'lactate'),
    ]
    for pattern, name in lab_pats:
        matches = re.findall(pattern, case_text, re.IGNORECASE)
        if matches:
            vals = [f"{name} {m}" for m in matches]
            facts["abnormal_labs"].extend(vals)

    # Exposures
    exposure_pats = [
        r'(?:diet|consumption|eating|drank|drank|ingested?)\s+(?:of\s+)?([^.]{10,80})',
        r'(?:medication|drug|prescribed|taking|took|received)\s+([^.]{10,80})',
        r'(?:travel|traveled|visited|lived? in)\s+([^.]{10,80})',
        r'(?:smok|alcohol|tobacco|heroin|cocaine|marijuana)[^.]{0,60}\.',
        r'(?:occupation|worked? as|construction|farmer|teacher)[^.]{0,60}\.',
    ]
    for pat in exposure_pats:
        matches = re.findall(pat, case_text, re.IGNORECASE)
        for m in matches[:2]:
            clean = m.strip().rstrip('.')
            if clean and clean not in facts["exposures"]:
                facts["exposures"].append(clean)

    # Notable negative findings
    neg_patterns = [
        r'(?:no|without|denies|denied|absence of|no history of|no evidence of)\s+([^.]{10,60})',
    ]
    for pat in neg_patterns:
        matches = re.findall(pat, case_text, re.IGNORECASE)
        for m in matches[:5]:
            facts["negatives"].append(m.strip())

    return facts


def format_structured_case(case_text: str) -> str:
    """Create a structured summary of the case suitable for prompt injection.

    Cleans the raw text and wraps it with section markers for clarity.
    This gives the LLM a cleaner signal without any API call overhead.
    """
    cleaned = preprocess_case_text(case_text)
    sections = extract_case_sections(case_text)
    facts = extract_key_clinical_facts(case_text)

    # Build a structured presentation
    parts = ["## Clinical Case Presentation\n"]

    # Full cleaned text
    parts.append(cleaned)
    parts.append("")

    # Highlight key findings (if extraction found them)
    has_labs = bool(facts["abnormal_labs"]) or bool(sections.get("labs"))
    has_exposures = bool(facts["exposures"])
    has_negatives = bool(facts["negatives"])

    if has_labs or has_exposures or has_negatives:
        parts.append("---")
        parts.append("## Key Facts Summary")

        if facts["abnormal_labs"]:
            parts.append("**Abnormal Lab Values:**")
            for l in facts["abnormal_labs"][:8]:
                parts.append(f"- {l}")

        if facts["exposures"]:
            parts.append("**Exposures / History:**")
            for e in facts["exposures"][:5]:
                parts.append(f"- {e}")

        if facts["negatives"]:
            parts.append("**Notable Negative Findings:**")
            for n in facts["negatives"][:5]:
                parts.append(f"- No {n}")

    return "\n".join(parts)
