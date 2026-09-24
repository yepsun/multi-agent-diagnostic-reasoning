#!/usr/bin/env python3
"""
Scheme B: Adaptive high-quality retrieval diagnosis for small/weaker models.

Design philosophy: 强检索 + 严过滤 + 简推理 + 自适应触发
  - Strong retrieval: BM25 via PubMed/Europe PMC, with optional MedCPT reranking.
  - Strict filtering: NLI-based verifier (or keyword heuristic fallback).
  - Simple reasoning: short-format initial assessment with self-consistency
    sampling (majority vote), no verbose multi-perspective simulation.
  - Adaptive triggering: retrieve only when model confidence is low,
    self-consistency is unstable, or knowledge gap is detected.

Result dict is compatible with batch_ablation.py and other scheme outputs.
"""

import os
import sys
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from run_inference import call_llm, parse_llm_output, get_current_model, run_harness_check
from case_extraction import preprocess_case_text, format_structured_case
from retrieval_module import (
    RetrievalOrchestrator,
    PubMedClient,
    CacheManager,
    RetrievalStrategy,
    RetrievedSource,
    CACHE_DB_PATH,
    HybridRetriever,
    extract_findings_and_build_queries,
    retrieve_multi_query_with_fallback,
    StatPearlsClient,
    StatPearlsArticle,
)
from evidence_verifier import NLIEvidenceVerifier, get_default_verifier


# =============================================================================
# Configuration knobs
# =============================================================================

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("SCHEME_B_HIGH_CONFIDENCE", "0.85"))
GAP_THRESHOLD = float(os.environ.get("SCHEME_B_GAP_THRESHOLD", "0.20"))
ENTAILMENT_THRESHOLD = float(os.environ.get("SCHEME_B_ENTAILMENT_THRESHOLD", "0.45"))
MAX_EVIDENCE_PIECES = int(os.environ.get("SCHEME_B_MAX_EVIDENCE", "5"))
MIN_EVIDENCE_PIECES = int(os.environ.get("SCHEME_B_MIN_EVIDENCE", "2"))
CASE_SUMMARY_MAX_TOKENS = int(os.environ.get("SCHEME_B_SUMMARY_TOKENS", "800"))
ASSESSMENT_TEMPERATURE = float(os.environ.get("SCHEME_B_ASSESSMENT_TEMPERATURE", "0.2"))
FINAL_TEMPERATURE = float(os.environ.get("SCHEME_B_FINAL_TEMPERATURE", "0.1"))

# Query generation knobs
SCHEME_B_USE_STATPEARLS = os.environ.get("SCHEME_B_USE_STATPEARLS", "true").lower() == "true"
SCHEME_B_USE_HARNESS = os.environ.get("SCHEME_B_USE_HARNESS", "false").lower() == "true"
SCHEME_B_FORCE_RETRIEVAL = os.environ.get("SCHEME_B_FORCE_RETRIEVAL", "false").lower() == "true"
SCHEME_B_MAX_FINDING_QUERIES = int(os.environ.get("SCHEME_B_MAX_FINDING_QUERIES", "3"))
SCHEME_B_MAX_DX_QUERIES = int(os.environ.get("SCHEME_B_MAX_DX_QUERIES", "3"))
SCHEME_B_MAX_TOTAL_QUERIES = int(os.environ.get("SCHEME_B_MAX_TOTAL_QUERIES", "8"))
SCHEME_B_MAX_STATPEARLS_ARTICLES = int(os.environ.get("SCHEME_B_MAX_STATPEARLS_ARTICLES", "3"))

# Diagnosis-centric retrieval knobs
SCHEME_B_MAX_DX_CANDIDATES = int(os.environ.get("SCHEME_B_MAX_DX_CANDIDATES", "5"))
SCHEME_B_PER_DX_RETRIEVE_K = int(os.environ.get("SCHEME_B_PER_DX_RETRIEVE_K", "5"))
SCHEME_B_PER_DX_KEEP = int(os.environ.get("SCHEME_B_PER_DX_KEEP", "2"))
SCHEME_B_DX_EVIDENCE_SNIPPET_TOKENS = int(os.environ.get("SCHEME_B_DX_EVIDENCE_SNIPPET_TOKENS", "60"))
SCHEME_B_USE_STRUCTURED_DX_EVAL = os.environ.get("SCHEME_B_USE_STRUCTURED_DX_EVAL", "true").lower() == "true"
SCHEME_B_FORCE_RETRIEVAL_UNSTABLE = os.environ.get("SCHEME_B_FORCE_RETRIEVAL_UNSTABLE", "true").lower() == "true"

# Self-consistency knobs
SCHEME_B_SELF_CONSISTENCY_SAMPLES = int(os.environ.get("SCHEME_B_SELF_CONSISTENCY_SAMPLES", "3"))
SCHEME_B_SELF_CONSISTENCY_TEMPERATURE = float(os.environ.get("SCHEME_B_SELF_CONSISTENCY_TEMPERATURE", "0.5"))

# =============================================================================
# Prompts
# =============================================================================

_INITIAL_ASSESSMENT_PROMPT = """You are an expert diagnostic clinician analyzing a complex CPC case.

## Case Presentation
{structured_case}

## Task
Provide your top-3 differential diagnoses with confidence scores, and indicate whether external retrieval is needed.

Before finalizing your list, briefly challenge yourself: is there an alternative diagnosis that could explain MORE findings with FEWER contradictions than your leading diagnosis? List 1-2 such alternatives under SKEPTIC_ALTERNATIVES, especially dangerous or easily missed entities.

Respond in EXACTLY this format:
DIAGNOSIS_1: [most likely diagnosis]
CONFIDENCE_1: [0-100]
DIAGNOSIS_2: [alternative diagnosis]
CONFIDENCE_2: [0-100]
DIAGNOSIS_3: [alternative diagnosis]
CONFIDENCE_3: [0-100]
MUST_NOT_MISS:
- [a diagnosis that is dangerous and must not be missed, even if less likely]
- [another must-not-miss diagnosis]
SKEPTIC_ALTERNATIVES:
- [alternative diagnosis]: [key supporting feature]
NEEDS_RETRIEVAL: [YES or NO]
KNOWLEDGE_GAP: [brief explanation]

Rules:
- Confidence 80+ means highly confident; retrieval probably not needed.
- Confidence 50-79 means moderate uncertainty; retrieval would help.
- Confidence <50 means high uncertainty; retrieval is strongly needed.
- Set NEEDS_RETRIEVAL=YES if any diagnosis confidence is <80 or if the top two are close.
- MUST_NOT_MISS should include dangerous entities (e.g., malignancy, serious infection, vasculitis, immunodeficiency-related opportunistic infection) that the presentation could represent, even if they are not your top differential.
- SKEPTIC_ALTERNATIVES must NOT repeat any diagnosis already listed in DIAGNOSIS_1/2/3; give 1-2 entries, each with one key supporting feature.
- When the case involves infantile developmental regression, hypotonia, startle response, or other neurodegenerative signs, explicitly include lysosomal and storage disorders (e.g., Tay-Sachs disease/GM2 gangliosidosis, GM1 gangliosidosis, Krabbe disease, Niemann-Pick disease, metachromatic leukodystrophy) and neurodegeneration with brain iron accumulation (e.g., infantile neuroaxonal dystrophy/PLA2G6-related) in the differential or must-not-miss list.
"""


_STRUCTURED_FINAL_DIAGNOSIS_PROMPT = """You are an expert diagnostic clinician analyzing a complex CPC case.

## Case Summary
{case_summary}

## Candidate Diagnoses
{candidate_list}

## Evidence Evaluation
For each candidate diagnosis below, the retrieved literature has been pre-classified as SUPPORT or CONCERN.
Use these classifications, not your general knowledge, to evaluate each candidate.

{evidence_text}

## Your Task
Evaluate each candidate diagnosis systematically, then choose the single most likely diagnosis.

CRITICAL RULES:
1. Do NOT automatically prefer the initial leading diagnosis. The candidates are listed in random order; treat each one equally.
2. For each candidate, first state its strongest clinical fit to the case (key features it explains).
3. Then state whether the retrieved evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT for that candidate.
4. A candidate with strong clinical fit AND strong SUPPORT should be preferred.
5. A candidate with mostly CONCERNS, missing key facts, or poor clinical fit should be down-ranked.
6. Cite specific evidence items by number when they support or argue against a candidate.
7. Do NOT switch to a diagnosis unless both the clinical fit and the retrieved evidence favor it. General knowledge alone is NOT enough.
8. If evidence is insufficient for ALL candidates, choose the candidate with the best clinical fit and note uncertainty.
9. Ask yourself: "If the initial leading diagnosis is wrong, which alternative best explains the case?"

Respond with ONLY a valid JSON object using EXACTLY these keys:

{{
  "final_diagnosis": "[single best diagnosis]",
  "differential_diagnosis": ["[alternative 1]", "[alternative 2]", "[alternative 3]"],
  "reasoning": "[concise reasoning citing evidence by number; first briefly evaluate each candidate as SUPPORTED/CONTRADICTED/INSUFFICIENT]",
  "confidence_score": [0-100]
}}
"""


_CONFIRMATION_RETRIEVAL_PROMPT = """You are an expert diagnostic clinician analyzing a complex CPC case.

## Case Summary
{case_summary}

## Leading Diagnosis
{leading_diagnosis}

## Differential Diagnoses
{differential}

## Retrieved Medical Evidence
{evidence_text}

## Your Task
The leading diagnosis was proposed with moderate confidence. Evaluate whether the retrieved evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT for the leading diagnosis.

CRITICAL RULES:
1. If the evidence strongly supports the leading diagnosis and no alternative is better, KEEP it.
2. If the evidence contradicts the leading diagnosis AND strongly supports a differential diagnosis, switch to that alternative.
3. If evidence is insufficient, KEEP the leading diagnosis but note uncertainty.
4. Cite evidence by number.

Respond in EXACTLY this format:

DECISION: [KEEP or SWITCH]
SELECTED_DIAGNOSIS: [single best diagnosis]
REASONING: [concise reasoning citing evidence by number]
CONFIDENCE_SCORE: [0-100]
"""


_FINAL_DIAGNOSIS_PROMPT = """You are an expert diagnostic clinician analyzing a complex CPC case.

## Case Summary
{case_summary}

## Initial Assessment
{assessment_summary}

## Retrieved Medical Evidence
{evidence_text}

## Your Task
Synthesize the case, the initial assessment, and the retrieved evidence to produce a final diagnosis.

CRITICAL RULES:
1. Start from the initial leading diagnosis but remain open to alternatives if the retrieved evidence strongly supports a competing diagnosis.
2. Cite specific evidence items by number when they support or argue against a diagnosis.
3. If the evidence is insufficient or contradictory, keep the initial leading diagnosis but note uncertainty.

Respond in EXACTLY this format:
MOST_LIKELY_DIAGNOSIS: [single best diagnosis]
DIFFERENTIAL_DIAGNOSIS: [alternative 1], [alternative 2], [alternative 3]
REASONING: [concise reasoning citing evidence by number]
CONFIDENCE_SCORE: [0-100]
"""


# =============================================================================
# Helpers
# =============================================================================

def _truncate_text(text: str, max_tokens: int = CASE_SUMMARY_MAX_TOKENS, chars_per_token: int = 4) -> str:
    """Rough token-aware truncation."""
    max_chars = max_tokens * chars_per_token
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.7:
        return truncated[:last_period + 1]
    return truncated


def _build_case_summary(structured_case: str) -> str:
    """Return a concise case summary suitable for small-model context."""
    # For now, reuse the structured case but truncate aggressively.
    return _truncate_text(structured_case, max_tokens=CASE_SUMMARY_MAX_TOKENS)


def _parse_initial_assessment(raw: str) -> Optional[Dict]:
    """Parser for the plain initial assessment output."""
    if not raw:
        return None

    text = raw.strip()
    diagnoses = []

    for i in range(1, 4):
        diag_match = re.search(
            rf'DIAGNOSIS_{i}\s*:\s*(.+?)(?=\n\s*(?:DIAGNOSIS_|CONFIDENCE_|REASONING_|NEEDS_RETRIEVAL|KNOWLEDGE_GAP)|\Z)',
            text, re.IGNORECASE | re.DOTALL,
        )
        conf_match = re.search(rf'CONFIDENCE_{i}\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)

        if diag_match:
            diagnosis = diag_match.group(1).strip().split('\n')[0].strip()
            diagnosis = diagnosis.rstrip('.').rstrip(',')
            confidence = 50.0
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except ValueError:
                    pass
            if diagnosis and diagnosis.lower() not in ('none', 'n/a', ''):
                diagnoses.append({"diagnosis": diagnosis, "confidence": confidence})

    if not diagnoses:
        return None

    diagnoses.sort(key=lambda x: x["confidence"], reverse=True)

    if not diagnoses:
        return None

    needs_match = re.search(r'NEEDS_RETRIEVAL\s*:\s*(\w+)', text, re.IGNORECASE)
    needs_retrieval = True
    if needs_match:
        val = needs_match.group(1).lower()
        needs_retrieval = val in ("yes", "true", "y", "1")

    gap_match = re.search(
        r'KNOWLEDGE_GAP\s*:\s*(.+?)(?=\n\s*(?:DIAGNOSIS_|CONFIDENCE_|REASONING_|NEEDS_RETRIEVAL|KNOWLEDGE_GAP|MUST_NOT_MISS|SKEPTIC_ALTERNATIVES)|\Z)',
        text, re.IGNORECASE | re.DOTALL,
    )
    knowledge_gap = gap_match.group(1).strip() if gap_match else ""

    # Parse MUST_NOT_MISS list
    must_not_miss = []
    mm_section = re.search(
        r'(?:\*\*)?MUST_NOT_MISS(?:\*\*)?\s*:\s*\n((?:\s*[-*]\s*[^\n]+\n?)+)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if mm_section:
        for line in mm_section.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            dx = re.sub(r"^[-*]\s+", "", line).strip()
            dx = dx.rstrip('.').rstrip(',')
            if dx and dx.lower() not in ("none", "n/a", ""):
                must_not_miss.append(dx)

    # Parse SKEPTIC_ALTERNATIVES list ("diagnosis: supporting feature" bullets)
    skeptic_alternatives = []
    sk_section = re.search(
        r'(?:\*\*)?SKEPTIC_ALTERNATIVES(?:\*\*)?\s*:\s*\n((?:\s*[-*]\s*[^\n]+\n?)+)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if sk_section:
        for line in sk_section.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            alt = re.sub(r"^[-*]\s+", "", line).strip()
            alt = alt.rstrip('.').rstrip(',')
            if alt and alt.lower() not in ("none", "n/a", ""):
                skeptic_alternatives.append(alt)

    return {
        "top_diagnoses": diagnoses[:3],
        "needs_retrieval": needs_retrieval,
        "knowledge_gap": knowledge_gap,
        "skeptic_alternatives": skeptic_alternatives,
        "must_not_miss": must_not_miss,
    }


def _should_retrieve(assessment: Dict) -> bool:
    """Adaptive retrieval trigger."""
    diagnoses = assessment.get("top_diagnoses", [])
    if not diagnoses:
        return True

    confidences = [d["confidence"] / 100.0 for d in diagnoses]
    max_conf = max(confidences)
    second_conf = sorted(confidences, reverse=True)[1] if len(confidences) > 1 else 0.0
    gap = max_conf - second_conf

    needs_retrieval = assessment.get("needs_retrieval", True)

    # Trigger if: explicitly requested, low confidence, or top two are close
    if needs_retrieval:
        print(f"[Scheme B] Trigger retrieval: model requested it (max_conf={max_conf:.2f})")
        return True
    if max_conf < HIGH_CONFIDENCE_THRESHOLD:
        print(f"[Scheme B] Trigger retrieval: max confidence {max_conf:.2f} < {HIGH_CONFIDENCE_THRESHOLD}")
        return True
    if gap <= GAP_THRESHOLD:
        print(f"[Scheme B] Trigger retrieval: top-two gap {gap:.2f} <= {GAP_THRESHOLD}")
        return True

    print(f"[Scheme B] Skip retrieval: max_conf={max_conf:.2f}, gap={gap:.2f}")
    return False


def _extract_confidence(raw: str) -> float:
    """Extract confidence score from raw output, default 50."""
    m = re.search(r'CONFIDENCE_SCORE\s*:\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 50.0


def _extract_candidate_diagnoses(assessment: Dict) -> List[str]:
    """Build a deduplicated list of candidate diagnoses for per-diagnosis retrieval."""
    candidates = []

    for d in assessment.get("top_diagnoses", [])[:SCHEME_B_MAX_DX_CANDIDATES]:
        dx = d.get("diagnosis", "").strip()
        if dx:
            candidates.append(dx)

    # Add MUST_NOT_MISS diagnoses (dangerous entities the model flagged)
    for dx in assessment.get("must_not_miss", [])[:3]:
        dx = dx.strip()
        if dx and dx.lower() not in {c.lower() for c in candidates}:
            candidates.append(dx)

    # If top diagnoses are sparse, include more from the assessment.
    if len(candidates) < 3:
        for d in assessment.get("top_diagnoses", [])[SCHEME_B_MAX_DX_CANDIDATES:5]:
            dx = d.get("diagnosis", "").strip()
            if dx and dx.lower() not in {c.lower() for c in candidates}:
                candidates.append(dx)

    for alt in assessment.get("skeptic_alternatives", [])[:SCHEME_B_MAX_DX_CANDIDATES]:
        if ":" in alt:
            dx = alt.split(":", 1)[0].strip()
        else:
            dx = alt.strip()
        if dx and dx.lower() not in {c.lower() for c in candidates}:
            candidates.append(dx)

    return candidates[:SCHEME_B_MAX_DX_CANDIDATES]


def _extract_candidate_diagnoses_from_sources(
    sources: List[RetrievedSource],
    existing_candidates: List[str],
    case_summary: str,
    max_new: int = 3,
) -> List[str]:
    """Extract likely diagnosis names from retrieved literature titles/abstracts.

    Uses a lightweight LLM prompt to list disease entities mentioned in the
    sources. This helps discover diagnoses the initial assessment missed.
    """
    if not sources:
        return []

    # Build a compact text from titles and first 300 chars of abstracts
    snippets = []
    for i, src in enumerate(sources[:10], 1):
        text = f"{src.title or ''} {src.abstract or ''}".strip()
        if text:
            snippets.append(f"[{i}] {text[:300]}")
    if not snippets:
        return []

    prompt = f"""You are a medical expert reading retrieved PubMed abstracts.
Below are snippets from literature retrieved for a complex diagnostic case.

Case summary:
{case_summary}

Literature snippets:
{chr(10).join(snippets)}

Existing candidate diagnoses:
{chr(10).join(f"- {d}" for d in existing_candidates)}

Task: List up to {max_new} additional distinct disease or syndrome names that appear in the literature and could explain the case. Be broad: include rare genetic/metabolic disorders, infections, malignancies, and autoimmune conditions if the literature mentions them. Do NOT repeat any existing candidate. Return ONLY a bulleted list, one diagnosis per line.

Format:
- [disease name 1]
- [disease name 2]
- [disease name 3]
"""
    try:
        raw, _ = call_llm(prompt, temperature=0.0, max_tokens=200, disable_thinking=True)
    except Exception as e:
        print(f"[Scheme B] Candidate extraction from sources failed: {e}")
        return []

    new_candidates = []
    non_dx_prefixes = (
        "based on", "here are", "therefore", "in conclusion", "additional",
        "two additional", "the following", "possible diagnoses", "disease name",
    )
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading bullets/dashes
        dx = re.sub(r"^[-*\d]+[\.\)]?\s*", "", line).strip()
        dx = dx.rstrip('.').rstrip(',')
        lower = dx.lower()
        if (
            dx
            and len(dx) >= 3
            and len(dx) <= 120
            and not any(lower.startswith(p) for p in non_dx_prefixes)
            and dx.lower() not in {c.lower() for c in existing_candidates}
            and dx.lower() not in {c.lower() for c in new_candidates}
        ):
            new_candidates.append(dx)

    return new_candidates[:max_new]


def _discover_additional_candidates(
    case_text: str,
    existing_candidates: List[str],
    case_summary: str,
    retriever: HybridRetriever,
    max_new: int = 3,
) -> Tuple[List[str], List[str]]:
    """Broaden the candidate list by retrieving on clinical findings.

    Runs disease-agnostic clinical-finding queries, then asks a lightweight
    LLM to name any diseases in the retrieved literature that could explain
    the case but are not already in the candidate list.

    Returns:
        (new_candidates, finding_queries)
    """
    if len(existing_candidates) >= SCHEME_B_MAX_DX_CANDIDATES:
        return [], []

    queries = []
    try:
        findings_result = extract_findings_and_build_queries(case_text, max_chars=5000)
        queries = findings_result.get("pubmed_queries", [])[:SCHEME_B_MAX_FINDING_QUERIES]
        if not queries:
            return [], []
        print(f"[Scheme B] Discovery retrieval: {len(queries)} finding-based queries")
        sources = retriever.retrieve(queries, top_k=10, rerank_top_k=10)
        time.sleep(0.2)
    except Exception as e:
        print(f"[Scheme B] Discovery retrieval failed: {e}")
        return [], []

    new_candidates = _extract_candidate_diagnoses_from_sources(
        sources, existing_candidates, case_summary, max_new=max_new
    )
    return new_candidates, queries


def _run_self_consistent_assessment(
    prompt: str,
    n_samples: int = SCHEME_B_SELF_CONSISTENCY_SAMPLES,
    temperature: float = SCHEME_B_SELF_CONSISTENCY_TEMPERATURE,
) -> Optional[Dict]:
    """Run the initial assessment multiple times in parallel and return the
    most common parseable result.

    Uses a simple majority vote over the top diagnosis. If no clear majority,
    returns the sample with the highest average confidence. The samples are
    independent LLM calls, so they are issued concurrently.
    """
    if n_samples <= 1:
        raw, _ = call_llm(prompt, temperature=ASSESSMENT_TEMPERATURE, max_tokens=1200, disable_thinking=True)
        return _parse_initial_assessment(raw)

    def _draw_sample(idx: int) -> Optional[Dict]:
        raw, _ = call_llm(prompt, temperature=temperature, max_tokens=1200, disable_thinking=True)
        return _parse_initial_assessment(raw)

    samples = []
    with ThreadPoolExecutor(max_workers=n_samples) as executor:
        futures = {executor.submit(_draw_sample, i): i for i in range(n_samples)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                parsed = future.result()
                if parsed:
                    samples.append(parsed)
            except Exception as e:
                print(f"[Scheme B] Self-consistency sample {idx+1} failed: {e}")

    if not samples:
        return None

    # Count top-diagnosis votes (case-insensitive, ignore parenthetical modifiers)
    from collections import Counter

    def _core(dx: str) -> str:
        return re.sub(r"\s*\([^)]*\)", "", dx).strip().lower()

    top_votes = Counter(_core(s["top_diagnoses"][0]["diagnosis"]) for s in samples if s.get("top_diagnoses"))
    if top_votes:
        majority_core, majority_count = top_votes.most_common(1)[0]
        print(f"[Scheme B] Self-consistency: majority='{majority_core}' ({majority_count}/{len(samples)})")
        # Return the sample with the most confident majority diagnosis
        best_sample = None
        best_conf = -1.0
        for s in samples:
            if not s.get("top_diagnoses"):
                continue
            if _core(s["top_diagnoses"][0]["diagnosis"]) == majority_core:
                conf = s["top_diagnoses"][0]["confidence"]
                if conf > best_conf:
                    best_conf = conf
                    best_sample = s
        if best_sample:
            # Merge alternatives from other samples to broaden differential
            merged_alts = {majority_core}
            merged_diagnoses = list(best_sample["top_diagnoses"])
            for s in samples:
                for d in s.get("top_diagnoses", []):
                    core = _core(d["diagnosis"])
                    if core not in merged_alts:
                        merged_alts.add(core)
                        merged_diagnoses.append(d)
            merged_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
            best_sample["top_diagnoses"] = merged_diagnoses[:5]
            best_sample["_consistency"] = {
                "majority_count": majority_count,
                "total_samples": len(samples),
                "ratio": majority_count / len(samples) if samples else 0.0,
            }
            return best_sample

    # Fallback: highest confidence sample
    samples.sort(key=lambda s: s["top_diagnoses"][0]["confidence"] if s.get("top_diagnoses") else 0.0, reverse=True)
    best = samples[0]
    best["_consistency"] = {
        "majority_count": 1,
        "total_samples": len(samples),
        "ratio": 1.0 / len(samples) if samples else 0.0,
    }
    return best


def _retrieve_confirmation_evidence(
    diagnosis: str,
    differentials: List[str],
    retriever: HybridRetriever,
    verifier: NLIEvidenceVerifier,
    max_total: int = 8,
) -> Tuple[List[RetrievedSource], int]:
    """Retrieve evidence specifically to confirm/refute the leading diagnosis.

    Queries the leading diagnosis and the top differential directly. Returns
    sources that support either the leading diagnosis or any differential.
    """
    queries = []
    candidates = [diagnosis] + differentials[:2]
    for dx in candidates:
        clean_dx = _diagnosis_to_query_fragment(dx)
        if clean_dx and len(clean_dx) >= 3:
            queries.append(f'"{clean_dx}"[Title/Abstract]')
    queries = _dedup_queries(queries)

    if not queries:
        return [], 0

    try:
        sources = retriever.retrieve(queries, top_k=max_total * 2, rerank_top_k=max_total)
        time.sleep(0.2)
    except Exception as e:
        print(f"[Scheme B] Confirmation retrieval failed: {e}")
        return [], 0

    # Keep sources that support either the leading diagnosis or a differential
    kept = []
    for src in sources:
        text = f"{src.title or ''} {src.abstract or ''}".strip()
        if not text:
            continue
        for dx in candidates:
            score, label = verifier.verify(dx, text, entailment_threshold=ENTAILMENT_THRESHOLD)
            if label == "entailment":
                kept.append(src)
                break

    return kept[:max_total], len(queries)


def _format_confirmation_evidence(sources: List[RetrievedSource]) -> str:
    """Format sources for the confirmation prompt."""
    if not sources:
        return "No relevant literature retrieved."
    lines = []
    for i, src in enumerate(sources, 1):
        title = src.title or "Untitled"
        abstract = src.abstract or ""
        year = src.year or "n.d."
        snippet = _snippet(abstract) if abstract else ""
        lines.append(f"[{i}] {title} ({year}): {snippet}")
    return "\n".join(lines)


def _build_per_dx_queries(diagnosis: str, context_phrases: Optional[List[str]] = None) -> List[str]:
    """Generate supportive PubMed queries for one candidate diagnosis.

    We retrieve direct literature for the candidate and let the verifier separate
    support from contradiction. A context-aware query combines the diagnosis with
    a distinctive clinical phrase to disambiguate generic names (e.g., "Lymphoma"
    in a specific anatomic context).

    We avoid generic terms inside parentheses (e.g., "birds", "molds") and
    keep queries short enough for PubMed [Title/Abstract] search.
    """
    clean_dx = _diagnosis_to_query_fragment(diagnosis)
    if not clean_dx or len(clean_dx) < 3:
        return []

    queries = [f'"{clean_dx}"[Title/Abstract]']

    # Extract disease entities mentioned in parentheses (e.g., "(e.g., MALT or DLBCL)")
    # and issue direct queries for them. We are conservative: skip common words,
    # environmental exposures, severity modifiers, and short/generic tokens.
    parenthetical = re.search(r'\(([^)]*)\)', diagnosis)
    if parenthetical:
        inner = parenthetical.group(1)
        # Split on common separators
        tokens = re.split(r'[,;/]|\s+or\s+|\s+and\s+|\be\.g\.\b|\bi\.e\.\b', inner, flags=re.IGNORECASE)
        for token in tokens:
            token = token.strip().strip('.,;:-')
            token = re.sub(r'^(?:such as|like|including)\s+', '', token, flags=re.IGNORECASE)
            token = _diagnosis_to_query_fragment(token)
            lower = token.lower()
            # Heuristic: must look like a medical entity (multi-word or disease suffix)
            looks_medical = (
                ' ' in token
                or any(s in lower for s in ('disease', 'syndrome', 'lymphoma', 'carcinoma', 'sarcoma', 'disorder',
                                              'dystrophy', 'leukodystrophy', 'neurodegeneration', 'vasculitis',
                                              'pneumonia', 'fibrosis', 'infection', 'inflammation', 'tumor',
                                              'neoplasm', 'granulomatosis', 'polyangiitis', 'leukemia'))
            )
            if (
                token
                and len(token) >= 5
                and looks_medical
                and lower not in {'organisms', 'infection', 'disease', 'disorder', 'syndrome',
                                   'defect', 'defects', 'antigen', 'antigens', 'exposure', 'strain',
                                   'history', 'form', 'features'}
                and not re.fullmatch(r'e\.?g\.?|i\.?e\.?|and|or', token, re.IGNORECASE)
                and token.lower() != clean_dx.lower()
                and token.lower() not in {q.lower() for q in queries}
            ):
                queries.append(f'"{token}"[Title/Abstract]')

    if context_phrases:
        # Pick the first short, clean phrase as context
        for phrase in context_phrases:
            phrase_clean = _sanitize_pubmed_fragment(phrase)
            if phrase_clean and len(phrase_clean) >= 3 and len(phrase_clean) <= 40:
                queries.append(f'("{clean_dx}"[Title/Abstract]) AND ("{phrase_clean}"[Title/Abstract])')
                break

    return _dedup_queries(queries[:3])


def _verify_source_for_dx(
    source: RetrievedSource,
    diagnosis: str,
    verifier: NLIEvidenceVerifier,
    entailment_threshold: float = ENTAILMENT_THRESHOLD,
) -> Tuple[float, str]:
    """Verify a single source against a single diagnosis hypothesis."""
    text = f"{source.title or ''} {source.abstract or ''}".strip()
    if not text:
        return 0.0, "neutral"
    score, label = verifier.verify(diagnosis, text, entailment_threshold=entailment_threshold)
    return score, label


def _retrieve_evidence_for_candidate(
    diagnosis: str,
    retriever: HybridRetriever,
    verifier: NLIEvidenceVerifier,
    top_k: int = SCHEME_B_PER_DX_RETRIEVE_K,
    keep: int = SCHEME_B_PER_DX_KEEP,
    context_phrases: Optional[List[str]] = None,
) -> Tuple[List[Tuple[RetrievedSource, float]], List[Tuple[RetrievedSource, float]]]:
    """Retrieve and verify evidence for one candidate diagnosis.

    Returns (supporting_sources, contradicting_sources), each as list of (source, score).
    """
    queries = _build_per_dx_queries(diagnosis, context_phrases=context_phrases)
    if not queries:
        return [], []

    try:
        raw_sources = retriever.retrieve(queries, top_k=top_k * 2, rerank_top_k=top_k)
    except Exception as e:
        print(f"[Scheme B] Per-dx retrieval failed for '{diagnosis}': {e}")
        return [], []
    finally:
        # Throttle NCBI E-utilities calls to stay well under the 10/s API-key limit.
        time.sleep(0.2)

    supporting = []
    contradicting = []
    for src in raw_sources:
        score, label = _verify_source_for_dx(src, diagnosis, verifier)
        if label == "entailment":
            supporting.append((src, score))
        elif label == "contradiction":
            contradicting.append((src, score))

    supporting.sort(key=lambda x: x[1], reverse=True)
    contradicting.sort(key=lambda x: x[1], reverse=True)
    return supporting[:keep], contradicting[:keep]


def _snippet(text: str, max_tokens: int = SCHEME_B_DX_EVIDENCE_SNIPPET_TOKENS, chars_per_token: int = 4) -> str:
    """Return a short one-sentence snippet for prompt inclusion."""
    max_chars = max_tokens * chars_per_token
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        return truncated[: last_period + 1]
    return truncated + "..."


def _format_per_dx_evidence(
    candidate_evidence: Dict[str, Dict],
    statpearls_by_dx: Dict[str, StatPearlsArticle],
) -> str:
    """Format per-candidate evidence bundles into a structured prompt section."""
    if not candidate_evidence:
        return "No external evidence retrieved."

    global_index = 1
    index_to_source = {}
    lines = []

    for dx, bundle in candidate_evidence.items():
        lines.append(f"### Candidate: {dx}")

        support = bundle.get("support", [])
        concern = bundle.get("concern", [])
        sp_article = statpearls_by_dx.get(dx)

        if support:
            lines.append("Support:")
            for src, score in support:
                label = f"[{global_index}]"
                index_to_source[global_index] = (src, score, dx, "support")
                title = src.title or "Untitled"
                year = src.year or "n.d."
                snippet = _snippet(src.abstract or "")
                lines.append(f"  {label} {title} ({year}): {snippet}")
                global_index += 1
        else:
            lines.append("Support: none")

        if concern:
            lines.append("Concerns / Contradictions:")
            for src, score in concern:
                label = f"[{global_index}]"
                index_to_source[global_index] = (src, score, dx, "concern")
                title = src.title or "Untitled"
                year = src.year or "n.d."
                snippet = _snippet(src.abstract or "")
                lines.append(f"  {label} {title} ({year}): {snippet}")
                global_index += 1

        if sp_article:
            sp_text = sp_article.to_text(max_chars=500).replace("\n", " ")
            lines.append(f"StatPearls guidance: {sp_text}")

        lines.append("")

    return "\n".join(lines)


def _run_structured_dx_evaluation(
    case_summary: str,
    candidates: List[str],
    retriever: HybridRetriever,
    verifier: NLIEvidenceVerifier,
    statpearls_client: Optional[StatPearlsClient] = None,
    context_phrases: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Dict], Dict[str, StatPearlsArticle], int, int]:
    """Run per-candidate retrieval, verification, and formatting for final prompt.

    Returns:
        (evidence_text, candidate_evidence, statpearls_by_dx, total_pubmed_queries, total_sources_kept)
    """
    candidate_evidence: Dict[str, Dict] = {}
    statpearls_by_dx: Dict[str, StatPearlsArticle] = {}
    total_pubmed_queries = 0
    total_sources_kept = 0

    # Per-candidate retrieval and NLI verification
    for dx in candidates:
        support, concern = _retrieve_evidence_for_candidate(
            dx, retriever, verifier,
            top_k=SCHEME_B_PER_DX_RETRIEVE_K,
            keep=SCHEME_B_PER_DX_KEEP,
            context_phrases=context_phrases,
        )
        # Count queries actually executed regardless of whether the verifier kept
        # any evidence. Previously the count only accumulated inside the
        # `if support or concern` branch, so when all sources were filtered to
        # neutral the counter stayed 0 even though retrieval really happened.
        # _build_per_dx_queries is a pure string builder (no I/O), so calling it
        # again here purely for counting is safe and does not double-count.
        queries = _build_per_dx_queries(dx, context_phrases=context_phrases)
        if queries:
            total_pubmed_queries += len(queries)
        if support or concern:
            candidate_evidence[dx] = {"support": support, "concern": concern}
            total_sources_kept += len(support) + len(concern)

    # Optional StatPearls guidance per candidate
    if SCHEME_B_USE_STATPEARLS and statpearls_client is None:
        statpearls_client = StatPearlsClient()

    if SCHEME_B_USE_STATPEARLS and statpearls_client is not None:
        for dx in candidates:
            try:
                clean_dx = _clean_diagnosis_name(dx)
                if not clean_dx or len(clean_dx) < 3:
                    continue
                article = statpearls_client.lookup(clean_dx)
                if article:
                    statpearls_by_dx[dx] = article
                    if len(statpearls_by_dx) >= SCHEME_B_MAX_STATPEARLS_ARTICLES:
                        break
            except Exception as e:
                print(f"[Scheme B] StatPearls lookup failed for '{dx}': {e}")
        print(f"[Scheme B] Fetched {len(statpearls_by_dx)} StatPearls article(s)")

    evidence_text = _format_per_dx_evidence(candidate_evidence, statpearls_by_dx)
    return evidence_text, candidate_evidence, statpearls_by_dx, total_pubmed_queries, total_sources_kept


def _candidate_list_text(candidates: List[str]) -> str:
    """Format candidate diagnosis list for the final prompt."""
    return "\n".join(f"{i}. {dx}" for i, dx in enumerate(candidates, 1))


def _format_evidence(sources: List[RetrievedSource]) -> str:
    """Format retrieved sources for the final diagnosis prompt (legacy flat format)."""
    if not sources:
        return "No relevant literature retrieved."
    lines = []
    for i, src in enumerate(sources, 1):
        title = src.title or "Untitled"
        abstract = src.abstract or ""
        year = src.year or "n.d."
        lines.append(f"[{i}] {title} ({year})")
        if abstract:
            lines.append(f"    {abstract}")
        if src.url:
            lines.append(f"    URL: {src.url}")
        lines.append("")
    return "\n".join(lines)


def _diagnosis_to_query_fragment(diagnosis: str) -> str:
    """Clean a diagnosis string for use in a PubMed query.

    Strips trailing modifiers/complications (e.g., 'with ...', 'due to ...',
    'complicated by ...') so the core disease name is searchable.
    """
    if not diagnosis:
        return ""
    clean = diagnosis.strip().rstrip('.').rstrip(',')
    # Remove markdown bold/italic wrappers
    clean = re.sub(r'\*+([^*]+)\*+', r'\1', clean).strip()
    # Remove parenthetical content
    clean = re.sub(r'\s*\([^)]*\)', '', clean).strip()
    # Remove trailing modifiers/complications to keep the core diagnosis
    clean = re.sub(
        r'\s+(?:with|due to|secondary to|complicated by|associated with|and|or)\s+.*$',
        '',
        clean,
        flags=re.IGNORECASE,
    ).strip()
    # Remove common severity/modifier prefixes
    clean = re.sub(r'^(?:severe|acute|chronic|idiopathic|primary|secondary|advanced|early|late)\s+', '', clean, flags=re.IGNORECASE)
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def _clean_diagnosis_name(diagnosis: str) -> str:
    """Strip artifacts and noise from a diagnosis name for StatPearls lookup.

    Uses a shorter core term when the full diagnosis is unlikely to match a
    StatPearls book title (e.g., 'AL amyloidosis' -> 'amyloidosis').
    """
    if not diagnosis:
        return ""
    clean = diagnosis.strip()
    # Strip literal NEEDS_RETRIEVAL marker if the parser leaked it
    clean = re.sub(r'\bNEEDS_RETRIEVAL\b', '', clean, flags=re.IGNORECASE).strip()
    # Strip markdown bold/italic wrappers
    clean = re.sub(r'\*+([^*]+)\*+', r'\1', clean).strip()
    # Strip parenthetical content and leading/trailing punctuation
    clean = re.sub(r'\s*\([^)]*\)', '', clean).strip()
    clean = clean.strip('.,;:-')
    # Remove common severity/modifier prefixes that are rarely in StatPearls titles
    clean = re.sub(r'^(?:severe|acute|chronic|idiopathic|primary|secondary|advanced|early|late)\s+', '', clean, flags=re.IGNORECASE)
    # Collapse extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Map verbose or subtype-heavy names to shorter core disease names that
    # are more likely to match a StatPearls title.
    lower = clean.lower()
    core_mappings = {
        "gm2 gangliosidosis": "tay-sachs disease",
        "infantile gm2 gangliosidosis": "tay-sachs disease",
        "al amyloidosis": "amyloidosis",
        "attr amyloidosis": "amyloidosis",
        "light chain deposition disease": "light chain deposition disease",
        "systemic amyloidosis": "amyloidosis",
        "igg4-related disease": "igg4-related disease",
        "pulmonary veno-occlusive disease": "pulmonary veno-occlusive disease",
        "malignant pleural effusion": "malignant pleural effusion",
        "malignant pleural mesothelioma": "malignant mesothelioma",
        "tuberculous pleurisy": "tuberculosis",
    }
    for phrase, core in core_mappings.items():
        if phrase in lower:
            return core
    return clean


def _sanitize_pubmed_fragment(fragment: str) -> str:
    """Clean a free-text fragment for use in a PubMed [Title/Abstract] query."""
    fragment = fragment.strip().strip('"').strip("'")
    fragment = re.sub(r"\s+", " ", fragment)
    fragment = fragment.strip(".,;:!?")
    # Remove any field tags the model may have included
    fragment = re.sub(r"\s*\[[^\]]+\]", "", fragment)
    return fragment


def _fragment_to_pubmed_query(fragment: str) -> Optional[str]:
    """Convert a cleaned fragment into a PubMed [Title/Abstract] query."""
    fragment = _sanitize_pubmed_fragment(fragment)
    if len(fragment) < 3:
        return None
    return f'"{fragment}"[Title/Abstract]'


def _dedup_queries(queries: List[str]) -> List[str]:
    """Deduplicate queries while preserving order."""
    seen = set()
    unique = []
    for q in queries:
        norm = q.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(q)
    return unique


def _format_statpearls_evidence(articles: List[StatPearlsArticle]) -> str:
    """Format StatPearls articles for the final prompt."""
    if not articles:
        return ""
    parts = ["## StatPearls Clinical Guidance"]
    for i, article in enumerate(articles, 1):
        parts.append(f"[{i}] {article.title}")
        parts.append(article.to_text(max_chars=800))
        parts.append("")
    return "\n".join(parts)


def _fetch_statpearls_for_diagnoses(
    diagnoses: List[str],
    client: Optional[StatPearlsClient] = None,
    max_articles: int = SCHEME_B_MAX_STATPEARLS_ARTICLES,
) -> List[StatPearlsArticle]:
    """Fetch StatPearls articles for a list of candidate diagnoses."""
    if not SCHEME_B_USE_STATPEARLS or not diagnoses:
        return []

    if client is None:
        client = StatPearlsClient()

    articles = []
    seen_urls = set()
    for dx in diagnoses:
        clean_dx = _clean_diagnosis_name(dx)
        if not clean_dx or len(clean_dx) < 3:
            continue
        try:
            article = client.lookup(clean_dx)
            if article and article.url not in seen_urls:
                articles.append(article)
                seen_urls.add(article.url)
                if len(articles) >= max_articles:
                    break
        except Exception as e:
            print(f"[Scheme B] StatPearls lookup failed for '{clean_dx}': {e}")
    print(f"[Scheme B] Fetched {len(articles)} StatPearls article(s)")
    return articles


def _build_assessment_summary(assessment: Dict) -> str:
    """Build a concise initial-assessment summary for prompts / logging."""
    parts = []
    top_dx = [d for d in assessment.get("top_diagnoses", []) if d.get("diagnosis")]
    if top_dx:
        parts.append(
            "Top differential: "
            + "; ".join(f"{d['diagnosis']} ({d.get('confidence', 50):.0f})" for d in top_dx[:5])
        )
    must_not_miss = assessment.get("must_not_miss", [])
    if must_not_miss:
        parts.append("Must-not-miss: " + "; ".join(must_not_miss[:3]))
    skeptic = assessment.get("skeptic_alternatives", [])
    if skeptic:
        parts.append("Skeptic alternatives: " + "; ".join(skeptic[:2]))
    consistency = assessment.get("_consistency", {})
    if consistency:
        parts.append(
            f"Self-consistency: {consistency.get('majority_count', 0)}/{consistency.get('total_samples', 0)} samples agreed on the leading diagnosis"
        )
    return "\n".join(parts)


def _build_queries_for_retrieval(
    case_text: str,
    assessment: Optional[Dict],
) -> List[str]:
    """Build a diverse set of PubMed queries for retrieval."""
    queries = []

    # Priority 1: Diagnosis-hypothesis queries for the top candidates.
    # These are the most direct way to retrieve evidence for the diseases the
    # model actually considered. We query the top-3 initial diagnoses and the
    # top-2 skeptic alternatives (deduplicated) so a correct diagnosis that is
    # not ranked first still gets a chance to be supported by literature.
    dx_queries = []
    if assessment:
        candidate_dx = []
        for d in assessment.get("top_diagnoses", [])[:SCHEME_B_MAX_DX_QUERIES]:
            dx = d.get("diagnosis", "")
            if dx:
                candidate_dx.append(dx)

        for alt in assessment.get("skeptic_alternatives", [])[:SCHEME_B_MAX_DX_QUERIES]:
            if ":" in alt:
                dx = alt.split(":", 1)[0].strip()
            else:
                dx = alt.strip()
            if dx and dx.lower() not in {c.lower() for c in candidate_dx}:
                candidate_dx.append(dx)

        for dx in candidate_dx:
            clean_dx = _diagnosis_to_query_fragment(dx)
            if clean_dx and len(clean_dx) >= 3:
                dx_queries.append(f'"{clean_dx}"[Title/Abstract]')

    # Priority 2: Disease-agnostic clinical-finding queries.
    finding_queries = []
    try:
        findings_result = extract_findings_and_build_queries(case_text, max_chars=5000)
        finding_queries = findings_result.get("pubmed_queries", [])[:SCHEME_B_MAX_FINDING_QUERIES]
        print(f"[Scheme B] Generated {len(finding_queries)} clinical-finding queries")
    except Exception as e:
        print(f"[Scheme B] Clinical-finding query generation failed: {e}")

    # Assemble with explicit priority: direct diagnosis queries first, then
    # disease-agnostic findings fill any remaining budget.
    queries = dx_queries + finding_queries
    unique = _dedup_queries(queries)
    final_queries = unique[:SCHEME_B_MAX_TOTAL_QUERIES]

    print(
        f"[Scheme B] Final query set: {len(final_queries)} "
        f"(dx={len(dx_queries)}, findings={len(finding_queries)}, cap={SCHEME_B_MAX_TOTAL_QUERIES})"
    )
    return final_queries


def _filter_sources_with_verifier(
    sources: List[RetrievedSource],
    hypotheses: List[str],
    verifier: NLIEvidenceVerifier,
    entailment_threshold: float = ENTAILMENT_THRESHOLD,
) -> List[Tuple[RetrievedSource, float, str]]:
    """
    Keep sources that support any of the hypotheses with sufficient NLI score.
    Strict filtering: only true entailment labels pass.
    """
    if not sources:
        return []

    hypotheses = [h for h in hypotheses if h]
    if not hypotheses:
        return []

    scored = []
    for src in sources:
        text = f"{src.title or ''} {src.abstract or ''}".strip()
        if not text.strip():
            continue

        best_score = 0.0
        best_label = "neutral"
        for hyp in hypotheses:
            score, label = verifier.verify(hyp, text, entailment_threshold=entailment_threshold)
            if label == "entailment" and score > best_score:
                best_score = score
                best_label = label
            elif label == "contradiction" and best_label != "entailment":
                if score > best_score:
                    best_score = score
                    best_label = label
            elif best_label == "neutral" and score > best_score:
                best_score = score
                best_label = label

        if best_label == "entailment":
            scored.append((src, best_score, best_label))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _get_retriever_from_orchestrator(orchestrator: Optional[RetrievalOrchestrator]) -> HybridRetriever:
    """Build a HybridRetriever, reusing the orchestrator's PubMed client if available."""
    if orchestrator is not None:
        return HybridRetriever(pubmed_client=orchestrator.pubmed, enable_dense_rerank=True)
    return HybridRetriever(enable_dense_rerank=True)


# =============================================================================
# Main entry point
# =============================================================================

def run_scheme_b(
    case: Dict,
    orchestrator: Optional[RetrievalOrchestrator] = None,
    verifier: Optional[NLIEvidenceVerifier] = None,
    initial_assessment_prompt: Optional[str] = None,
) -> Dict:
    """
    Run Scheme B adaptive retrieval diagnosis on a case.

    Args:
        case: Case dict with 'case_id', 'Q' (case text), 'A' (answer dict).
        orchestrator: Optional RetrievalOrchestrator instance (shared across cases).
        verifier: Optional NLIEvidenceVerifier instance.
        initial_assessment_prompt: Optional override for the initial assessment
            prompt (must contain a {structured_case} placeholder). Default None
            uses the built-in single-perspective prompt.

    Returns:
        Result dict compatible with batch_ablation.py.
    """
    start_time = time.time()
    case_id = case.get("case_id", "unknown")
    case_text = case.get("Q", "")

    cleaned_text = preprocess_case_text(case_text)
    structured_case = format_structured_case(cleaned_text)
    case_summary = _build_case_summary(structured_case)

    if verifier is None:
        verifier = get_default_verifier()

    total_llm_calls = 0
    total_pubmed_queries = 0
    retrieval_sources = []
    trigger_reason = "high_confidence_early_stop"

    # ------------------------------------------------------------------
    # Step 1: Self-consistent initial assessment (no retrieval)
    # ------------------------------------------------------------------
    prompt = (initial_assessment_prompt or _INITIAL_ASSESSMENT_PROMPT).format(
        structured_case=structured_case
    )
    assessment = _run_self_consistent_assessment(prompt)
    total_llm_calls += SCHEME_B_SELF_CONSISTENCY_SAMPLES

    if assessment is None:
        print("[Scheme B] Failed to parse initial assessment; falling back to direct diagnosis.")
        assessment = {
            "top_diagnoses": [{"diagnosis": "", "confidence": 50.0}],
            "needs_retrieval": True,
            "knowledge_gap": "parse failure",
            "skeptic_alternatives": [],
            "must_not_miss": [],
        }

    top_diagnoses = assessment.get("top_diagnoses", [])
    preliminary_dx = [d["diagnosis"] for d in top_diagnoses if d.get("diagnosis")]
    top_confidence = top_diagnoses[0]["confidence"] if top_diagnoses else 50.0
    leading_diagnosis = preliminary_dx[0] if preliminary_dx else ""
    differential_diagnoses = preliminary_dx[1:3]

    # Compute top-two confidence gap for fast-path decision
    confidences = [d["confidence"] for d in top_diagnoses if d.get("confidence") is not None]
    sorted_conf = sorted(confidences, reverse=True)
    top_two_gap = (sorted_conf[0] - sorted_conf[1]) if len(sorted_conf) > 1 else sorted_conf[0]

    # ------------------------------------------------------------------
    # Step 2: Adaptive retrieval trigger
    # ------------------------------------------------------------------
    retrieval_sources = []
    candidate_evidence: Dict[str, Dict] = {}
    statpearls_by_dx: Dict[str, StatPearlsArticle] = {}
    trigger_reason = "high_confidence_early_stop"

    should_retrieve = _should_retrieve(assessment)

    # Tracks whether the structured (JSON-mode) final-diagnosis path is active.
    # Only the structured path requests JSON output; the legacy confirmation /
    # no-retrieval paths keep the plain-text format as a fallback.
    structured_final_mode = False

    # ------------------------------------------------------------------
    # Fast path: very high confidence, or medium confidence with unstable
    # self-consistency (retrieval likely to add noise rather than value).
    # ------------------------------------------------------------------
    consistency = assessment.get("_consistency", {})
    ratio = consistency.get("ratio", 1.0)
    total_samples = consistency.get("total_samples", 1)
    is_stable = ratio >= 1.0 and total_samples >= 2

    if top_confidence >= 90.0 and not SCHEME_B_FORCE_RETRIEVAL:
        trigger_reason = "high_confidence_early_stop"
        print(f"[Scheme B] Fast path (>=90): max_conf={top_confidence:.2f}")
        final_diagnosis = leading_diagnosis
    elif 85.0 <= top_confidence < 90.0 and not is_stable and not SCHEME_B_FORCE_RETRIEVAL and not SCHEME_B_FORCE_RETRIEVAL_UNSTABLE:
        trigger_reason = "medium_confidence_unstable_consistency"
        print(f"[Scheme B] Fast path (unstable {ratio:.0%}): max_conf={top_confidence:.2f}")
        final_diagnosis = leading_diagnosis
    else:
        if 85.0 <= top_confidence < 90.0 and not is_stable and SCHEME_B_FORCE_RETRIEVAL_UNSTABLE:
            print(f"[Scheme B] Unstable consistency forces retrieval: ratio={ratio:.0%}, max_conf={top_confidence:.2f}")
        final_diagnosis = None  # will proceed to retrieval below

    if final_diagnosis is not None:
        total_time = time.time() - start_time
        return {
            "case_id": case_id,
            "final_diagnosis": final_diagnosis,
            "final_confidence": top_confidence,
            "rounds": [
                {
                    "round_number": 1,
                    "diagnosis": final_diagnosis,
                    "differential": differential_diagnoses,
                    "confidence_score": top_confidence,
                    "confidence_tier": "high",
                    "retrieval_sources_count": 0,
                    "retrieval_sources": [],
                    "reflection_summary": (assessment.get("knowledge_gap", "") + "\n" + _build_assessment_summary(assessment)).strip(),
                    "revised_diagnosis": final_diagnosis,
                    "revision_made": False,
                    "harness_verdict": "KEEP",
                    "harness_contradictions": [],
                    "harness_revision_accepted": False,
                    "reasoning": f"High-confidence leading diagnosis accepted without retrieval (max_conf={top_confidence:.1f}, gap={top_two_gap:.1f}).",
                }
            ],
            "total_llm_calls": total_llm_calls,
            "total_pubmed_queries": 0,
            "total_time_seconds": total_time,
            "termination_reason": trigger_reason,
            "harness_verdict": "KEEP",
            "harness_contradictions": [],
            "model": get_current_model(),
            "provider": os.environ.get("LLM_PROVIDER", "deepseek-flash"),
        }

    if should_retrieve:
        trigger_reason = "adaptive_retrieval"
        retriever = _get_retriever_from_orchestrator(orchestrator)

        if SCHEME_B_USE_STRUCTURED_DX_EVAL:
            # ------------------------------------------------------------------
            # Step 3 (structured): Per-candidate retrieval + verification
            # ------------------------------------------------------------------
            candidates = _extract_candidate_diagnoses(assessment)
            print(f"[Scheme B] Initial candidates ({len(candidates)}): {candidates}")

            # Broaden candidate list using disease-agnostic finding queries
            discovered, finding_queries = _discover_additional_candidates(
                case_text=case_text,
                existing_candidates=candidates,
                case_summary=case_summary,
                retriever=retriever,
                max_new=SCHEME_B_MAX_DX_CANDIDATES - len(candidates),
            )
            if discovered:
                candidates = candidates + discovered
                print(f"[Scheme B] Expanded candidates ({len(candidates)}): {candidates}")
            total_llm_calls += 1  # candidate discovery uses one LLM call

            # Derive short context phrases from the finding queries for per-dx retrieval
            context_phrases = []
            for q in finding_queries:
                # Strip PubMed field tags and boolean operators
                cleaned = re.sub(r"\[[^\]]+\]", "", q)
                cleaned = re.sub(r"\b(AND|OR|NOT)\b", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"[()\"']", "", cleaned).strip()
                if cleaned and len(cleaned) >= 3:
                    context_phrases.append(cleaned)
            # Also include supporting features from skeptic alternatives
            for alt in assessment.get("skeptic_alternatives", [])[:2]:
                if ":" in alt:
                    feat = alt.split(":", 1)[1].strip()
                    if feat and len(feat) >= 3:
                        context_phrases.append(feat)
            context_phrases = list(dict.fromkeys(context_phrases))[:3]

            evidence_text, candidate_evidence, statpearls_by_dx, qcount, src_count = _run_structured_dx_evaluation(
                case_summary=case_summary,
                candidates=candidates,
                retriever=retriever,
                verifier=verifier,
                statpearls_client=None,
                context_phrases=context_phrases,
            )
            total_pubmed_queries += qcount
            print(f"[Scheme B] Per-cx evidence: {src_count} source(s) across {len(candidate_evidence)} candidate(s)")

            # Shuffle candidate order in the final prompt to reduce leading-diagnosis anchoring.
            import random as _random
            _shuffled_candidates = list(candidates)
            _random.shuffle(_shuffled_candidates)

            final_prompt = _STRUCTURED_FINAL_DIAGNOSIS_PROMPT.format(
                case_summary=case_summary,
                candidate_list=_candidate_list_text(_shuffled_candidates),
                evidence_text=evidence_text,
            )
            # Structured path: request JSON output and parse it with
            # parse_llm_output (JSON-first with legacy regex fallback).
            structured_final_mode = True
        else:
            # ------------------------------------------------------------------
            # Step 3 (legacy): Confirmation retrieval for leading + top differentials
            # ------------------------------------------------------------------
            retrieval_sources, qcount = _retrieve_confirmation_evidence(
                leading_diagnosis,
                differential_diagnoses,
                retriever,
                verifier,
                max_total=8,
            )
            total_pubmed_queries += qcount
            print(f"[Scheme B] Confirmation retrieval: {len(retrieval_sources)} sources for '{leading_diagnosis}'")

            if retrieval_sources:
                evidence_text = _format_confirmation_evidence(retrieval_sources)
                final_prompt = _CONFIRMATION_RETRIEVAL_PROMPT.format(
                    case_summary=case_summary,
                    leading_diagnosis=leading_diagnosis,
                    differential=", ".join(differential_diagnoses) if differential_diagnoses else "none",
                    evidence_text=evidence_text,
                )
            else:
                final_prompt = _FINAL_DIAGNOSIS_PROMPT.format(
                    case_summary=case_summary,
                    assessment_summary=_build_assessment_summary(assessment),
                    evidence_text="No external evidence retrieved.",
                )
    else:
        final_prompt = _FINAL_DIAGNOSIS_PROMPT.format(
            case_summary=case_summary,
            assessment_summary=_build_assessment_summary(assessment),
            evidence_text="No external evidence retrieved.",
        )

    raw_final, _ = call_llm(
        final_prompt,
        temperature=FINAL_TEMPERATURE,
        max_tokens=1200,
        use_json_mode=structured_final_mode,
    )
    total_llm_calls += 1

    parsed = parse_llm_output(raw_final)
    diagnosis = parsed.get("most_likely_diagnosis", "").strip()
    differential = parsed.get("differential_diagnosis", [])
    reasoning = parsed.get("reasoning", "").strip()
    # JSON mode surfaces confidence_score through parse_llm_output; for legacy
    # plain-text outputs, fall back to the CONFIDENCE_SCORE: regex.
    confidence = parsed.get("confidence_score")
    if confidence is None:
        confidence = _extract_confidence(raw_final)

    # If confirmation prompt produced a DECISION/SWITCH, prefer SELECTED_DIAGNOSIS.
    # This override only applies to legacy plain-text outputs; the structured
    # (JSON-mode) path is already fully parsed by parse_llm_output above.
    if not structured_final_mode:
        m = re.search(r'(?:SELECTED_DIAGNOSIS|FINAL_DIAGNOSIS|MOST_LIKELY_DIAGNOSIS)\s*:\s*(.+?)(?=\n|$)', raw_final, re.IGNORECASE)
        if m:
            diagnosis = m.group(1).strip().rstrip('.').rstrip(',')

    if not diagnosis:
        if preliminary_dx:
            diagnosis = preliminary_dx[0]
        else:
            diagnosis = "Unable to determine diagnosis"

    # ------------------------------------------------------------------
    # Optional Step 6: Harness check / revision
    # ------------------------------------------------------------------
    harness_verdict = "KEEP"
    harness_contradictions = []
    harness_revision_accepted = False
    revised_diagnosis = diagnosis

    if SCHEME_B_USE_HARNESS:
        try:
            candidate_literature = {}
            if candidate_evidence:
                for dx, bundle in candidate_evidence.items():
                    candidate_literature[dx] = [src.title for src, _ in bundle.get("support", [])]
            harness = run_harness_check(
                case_text=case_text,
                original=diagnosis,
                differential=[d for d in differential if d] or differential_diagnoses,
                candidate_literature=candidate_literature,
            )
            total_llm_calls += 1
            harness_verdict = harness.get("verdict", "KEEP")
            harness_contradictions = harness.get("contradictions", [])
            if harness_verdict == "REVISE":
                revised = harness.get("final_diagnosis", "").strip()
                if revised:
                    revised_diagnosis = revised
                    harness_revision_accepted = True
        except Exception as e:
            print(f"[Scheme B] Harness check failed: {e}")

    total_time = time.time() - start_time

    # Aggregate structured-evidence sources for result logging
    if candidate_evidence:
        retrieval_sources = []
        seen_urls = set()
        for dx, bundle in candidate_evidence.items():
            for src, _ in bundle.get("support", []) + bundle.get("concern", []):
                if src.url and src.url not in seen_urls:
                    retrieval_sources.append(src)
                    seen_urls.add(src.url)
                elif not src.url:
                    retrieval_sources.append(src)

    return {
        "case_id": case_id,
        "final_diagnosis": revised_diagnosis,
        "final_confidence": confidence,
        "rounds": [
            {
                "round_number": 1,
                "diagnosis": revised_diagnosis,
                "differential": differential,
                "confidence_score": confidence,
                "confidence_tier": "high" if confidence >= 85 else "medium" if confidence >= 50 else "low",
                "retrieval_sources_count": len(retrieval_sources),
                "retrieval_sources": [
                    {
                        "title": s.title,
                        "source": s.source,
                        "year": s.year,
                        "url": s.url,
                        "abstract": s.abstract,
                    }
                    for s in retrieval_sources
                ],
                "reflection_summary": (assessment.get("knowledge_gap", "") + "\n" + _build_assessment_summary(assessment)).strip(),
                "revised_diagnosis": revised_diagnosis,
                "revision_made": trigger_reason == "adaptive_retrieval",
                "harness_verdict": harness_verdict,
                "harness_contradictions": harness_contradictions,
                "harness_revision_accepted": harness_revision_accepted,
                "reasoning": reasoning,
            }
        ],
        "total_llm_calls": total_llm_calls,
        "total_pubmed_queries": total_pubmed_queries,
        "total_time_seconds": total_time,
        "termination_reason": trigger_reason,
        "harness_verdict": harness_verdict,
        "harness_contradictions": harness_contradictions,
        "model": get_current_model(),
        "provider": os.environ.get("LLM_PROVIDER", "deepseek-flash"),
    }


if __name__ == "__main__":
    import json

    dataset_path = os.environ.get("DATASET_PATH", "../data/mgh_qa_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        print("No cases found")
        sys.exit(1)

    case = dataset[0]
    print(f"Running Scheme B on case: {case.get('case_id', 'unknown')}")
    result = run_scheme_b(case)
    print(f"Final diagnosis: {result['final_diagnosis']}")
    print(f"Confidence: {result['final_confidence']}")
    print(f"LLM calls: {result['total_llm_calls']}")
    print(f"PubMed queries: {result['total_pubmed_queries']}")
    print(f"Retrieved sources: {result['rounds'][0]['retrieval_sources_count']}")
    print(f"Time: {result['total_time_seconds']:.1f}s")


_STRUCTURED_OBJECTIVE_CLUES_PROMPT = """You are an expert clinician extracting objective clinical findings from a CPC case.

Extract ONLY the following structured information from the case below.

Case:
{case_text}

Respond EXACTLY in this format:

ABNORMAL_LABS:
- [lab name]: [value] (reference range if available)

SENTINEL_EVENTS:
- [key event 1]
- [key event 2]

RISK_FACTORS:
- [risk factor 1]

KEY_ABNORMAL_FINDINGS:
- [finding 1]
- [finding 2]

TREATMENT_PATTERNS:
- [treatment and response 1]

MOST_STRIKING_ABNORMALITY:
- [single most striking abnormality]

Be objective: only list things EXPLICITLY stated as abnormal in the text. Do not infer or guess.
If a category has no findings, write "None."
"""


def _extract_objective_clues(case_text: str) -> str:
    """Extract structured objective clues from case text."""
    prompt = _STRUCTURED_OBJECTIVE_CLUES_PROMPT.format(case_text=case_text[:8000])
    try:
        raw, _ = call_llm(prompt, temperature=0.0, max_tokens=1000, disable_thinking=True)
    except Exception as e:
        print(f"[Scheme B] Objective clue extraction failed: {e}")
        return ""
    
    # Parse structured sections
    required_sections = [
        "ABNORMAL_LABS:", "SENTINEL_EVENTS:", "RISK_FACTORS:",
        "KEY_ABNORMAL_FINDINGS:", "TREATMENT_PATTERNS:", "MOST_STRIKING_ABNORMALITY:",
    ]
    
    result_parts = []
    all_found = True
    for section in required_sections:
        m = re.search(rf'{re.escape(section)}\s*(.*?)(?=\n[A-Z][A-Z_]+:|\Z)', raw, re.DOTALL)
        if m:
            result_parts.append(f"{section}\n{m.group(1).strip()}")
        else:
            all_found = False
    
    if all_found:
        return "\n\n".join(result_parts)
    
    # Fallback: return raw output
    print(f"[Scheme B] Objective clue extraction: missing sections, using raw output")
    return raw
