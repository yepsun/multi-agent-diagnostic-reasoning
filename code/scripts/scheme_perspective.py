#!/usr/bin/env python3
"""
Scheme P (Perspective): Lightweight STORM-style multi-perspective diagnosis.

Inspired by Stanford STORM (Synthesis of Topic Outlines through Retrieval and
Multi-perspective question asking), but reduced to a SINGLE LLM call with
explicit clinical perspectives.  No retrieval, no multi-agent orchestration,
no article generation.  The goal is to capture the cognitive benefit of
"looking at the case from multiple angles" without importing STORM's
retrieval-noise and coordination overhead.

Perspectives used:
  1. Attending physician - primary diagnostic synthesis
  2. Pathologist/physiologist - mechanism and tissue-level explanation
  3. Imaging/laboratory specialist - objective findings interpretation
  4. Epidemiologist - exposure, geography, demographics, risk factors
  5. Skeptic/challenger - actively seeks contradictions and alternatives

NOTE: this scheme requests PLAIN TEXT output (legacy format). deepseek-flash
is a reasoner model — with thinking enabled the pure-text answer lands in
reasoning_content and content stays empty. All call_llm calls therefore pass
disable_thinking=True.
"""

import os
import sys
import re
import time
from typing import Dict, List

from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from run_inference import call_llm, parse_llm_output, get_current_model
from case_extraction import preprocess_case_text, format_structured_case

# Thinking-ON mode needs a larger token budget: the reasoner spends tokens on
# deliberation and must still have room to conclude with the final answer
# block. 2048 (the default) gets consumed by thinking alone, truncating the
# reasoning before the model writes MOST_LIKELY_DIAGNOSIS:.
THINKING_MAX_TOKENS = 8192


PERSPECTIVE_PROMPT = """You are an expert diagnostic team analyzing a complex medical case from multiple complementary perspectives. Each perspective contributes a brief analysis; you then synthesize them into a single final diagnosis.

## Case Presentation
{structured_case}

## Diagnostic Perspectives

Analyze the case from each of the following perspectives. Be concise (2-4 sentences each).

**1. Attending Physician Perspective**
What is the most coherent unifying diagnosis? Which clinical features are most discriminating?

**2. Pathophysiology Perspective**
What underlying mechanism could produce this constellation of findings? Are there hallmark laboratory, histologic, or molecular clues?

**3. Imaging and Laboratory Specialist Perspective**
How should the objective data (imaging, labs, vitals, procedures) be interpreted? What patterns or paradoxes stand out?

**4. Epidemiology and Exposure Perspective**
What role do demographics, geography, travel, diet, medications, toxins, occupational exposures, or comorbidities play? Are there hidden risk factors in the narrative?

**5. Skeptic / Challenger Perspective**
What is the strongest argument AGAINST the leading diagnosis? What alternative diagnoses could explain MORE findings with FEWER contradictions? What findings remain unexplained?

## Synthesis Task
Integrate the five perspectives above and provide:
- The single most likely diagnosis
- A ranked differential diagnosis (3-5 alternatives)
- Brief reasoning explaining how the multi-perspective analysis led to your choice

Respond in EXACTLY this format:
MOST_LIKELY_DIAGNOSIS: [final best diagnosis]
DIFFERENTIAL_DIAGNOSIS: [differential 1], [differential 2], [differential 3], [differential 4]
REASONING: [concise multi-perspective synthesis]
CONFIDENCE_SCORE: [0-100]
"""

# Appended to PERSPECTIVE_PROMPT ONLY for the thinking-enabled mode (the
# non-thinking prompt must stay byte-identical). Tells the reasoner to end
# with a single clean final answer block so the tail-extraction in
# run_inference can reliably find the FINAL diagnosis instead of a
# mid-thinking slice.
THINKING_CLOSING_INSTRUCTION = (
    "\n\nIMPORTANT: Think freely during your analysis, but your response MUST "
    "conclude with a final answer block as the very last section, in exactly "
    "this format:\n"
    "MOST_LIKELY_DIAGNOSIS: <final best diagnosis>\n"
    "DIFFERENTIAL_DIAGNOSIS: <d1>, <d2>, <d3>, <d4>\n"
    "REASONING: <concise reasoning>\n"
    "CONFIDENCE_SCORE: <0-100>\n"
    "Nothing may follow this final block."
)


def _extract_confidence(raw: str) -> float:
    m = re.search(r'CONFIDENCE_SCORE\s*:\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 50.0


def run_scheme_perspective(case: Dict, temperature: float = 0.1,
                           enable_thinking: bool = False) -> Dict:
    """Run single-call STORM-style multi-perspective diagnosis on a case.

    Args:
        case: Case dict with 'Q' (case text) and 'A' (answer dict).
        temperature: LLM sampling temperature (default 0.1).
        enable_thinking: If True, run with the reasoner's thinking ON (no
            disable_thinking). The prompt is already plain-text marker format,
            so the final answer is pulled from reasoning_content. Default False
            preserves the current disable_thinking=True behavior.

    Returns:
        Result dict compatible with run_scheme_c/run_diagnosis output.
    """
    start_time = time.time()
    case_id = case.get("case_id", "unknown")
    case_text = case.get("Q", "")

    cleaned_text = preprocess_case_text(case_text)
    structured_case = format_structured_case(cleaned_text)

    prompt = PERSPECTIVE_PROMPT.format(structured_case=structured_case)
    if enable_thinking:
        # Thinking ON: plain-text markers survive reasoning extraction. The
        # closing instruction tells the model to end with the final block, and
        # a larger max_tokens gives the reasoner room to conclude.
        prompt = prompt + THINKING_CLOSING_INSTRUCTION
        raw, _ = call_llm(prompt, temperature=temperature, max_tokens=THINKING_MAX_TOKENS)
    else:
        # disable_thinking=True: pure-text answer must go in `content`, not the
        # reasoner's reasoning_content (see module docstring).
        raw, _ = call_llm(prompt, temperature=temperature, max_tokens=2048,
                          disable_thinking=True)

    parsed = parse_llm_output(raw)
    diagnosis = parsed.get("most_likely_diagnosis", "").strip()
    differential = parsed.get("differential_diagnosis", [])
    reasoning = parsed.get("reasoning", "").strip()
    # Plain-text responses carry CONFIDENCE_SCORE: in the text, not in JSON.
    confidence = _extract_confidence(raw)

    if not diagnosis:
        # Fallback: use the first line that looks like a diagnosis. Hardened:
        # skip meta-commentary so "We need answer medical case..." is never
        # surfaced as a diagnosis.
        skip_prefixes = (
            "most", "differential", "reasoning", "confidence",
            "we need", "let's", "let us", "question", "actually",
            "think", "i think", "ok", "so ",
        )
        for line in raw.splitlines():
            line = line.strip()
            if line and len(line) > 3 and not line.lower().startswith(skip_prefixes):
                diagnosis = line.rstrip('.').rstrip(',')
                break
        if not diagnosis:
            diagnosis = "Unable to determine diagnosis"

    total_time = time.time() - start_time

    return {
        "case_id": case_id,
        "final_diagnosis": diagnosis,
        "final_confidence": confidence,
        "ranked_diagnoses": _ranked_from(diagnosis, differential),
        "rounds": [
            {
                "round_number": 1,
                "diagnosis": diagnosis,
                "differential": differential,
                "confidence_score": confidence,
                "confidence_tier": "high" if confidence >= 85 else "medium" if confidence >= 50 else "low",
                "retrieval_sources_count": 0,
                "reflection_summary": "",
                "revised_diagnosis": diagnosis,
                "revision_made": False,
                "harness_verdict": "KEEP",
                "harness_contradictions": [],
                "harness_revision_accepted": False,
                "reasoning": reasoning,
            }
        ],
        "total_llm_calls": 1,
        "total_pubmed_queries": 0,
        "total_time_seconds": total_time,
        "termination_reason": "perspective_single_shot",
        "harness_verdict": "KEEP",
        "harness_contradictions": [],
        "model": get_current_model(),
        "provider": os.environ.get("LLM_PROVIDER", "deepseek-flash"),
    }


def _ranked_from(diagnosis: str, differential: List[str]) -> List[str]:
    """Build the top-10 ranked list: diagnosis first, then deduped differential."""
    ranked = [diagnosis]
    for d in differential:
        if d and d.lower() not in [r.lower() for r in ranked]:
            ranked.append(d)
    return ranked[:10]


if __name__ == "__main__":
    import json

    dataset_path = os.environ.get("DATASET_PATH", "../data/mgh_qa_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        print("No cases found")
        sys.exit(1)

    case = dataset[0]
    print(f"Running Scheme P (Perspective) on case: {case.get('case_id', 'unknown')}")
    result = run_scheme_perspective(case)
    print(f"Final diagnosis: {result['final_diagnosis']}")
    print(f"Confidence: {result['final_confidence']}")
    print(f"LLM calls: {result['total_llm_calls']}")
    print(f"Time: {result['total_time_seconds']:.1f}s")
