#!/usr/bin/env python3
"""
Scheme A: Pure zero-shot baseline for medical diagnosis.

No retrieval, no reflection, no multi-agent discussion.
Single LLM call with chain-of-thought prompting.
Output format is compatible with Scheme C result dicts so they can be
compared directly in batch experiments.
"""

import os
import sys
import re
import time
from typing import Dict, List, Tuple

# Load .env before other imports
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
THINKING_MAX_TOKENS = int(os.environ.get("SCHEME_A_THINK_MAX_TOKENS", "8192"))
# Light-thinking mode: cap total generation (thinking + answer) so the model
# reasons briefly instead of producing a full deliberation chain.
THINKING_LIGHT_MAX_TOKENS = int(os.environ.get("SCHEME_A_THINK_LIGHT_MAX_TOKENS", "3500"))

# Appended to the plain prompt in light-thinking mode to steer the model
# toward brief deliberation (soft lever complementing the token cap).
THINKING_LIGHT_INSTRUCTION = (
    "\nIMPORTANT: Keep your thinking SHORT — at most 200 words of "
    "deliberation. Weigh only the 2-3 decisive clinical features, then "
    "commit to the final answer block. Do NOT write an exhaustive analysis.\n"
)


def _build_scheme_a_prompt(structured_case: str) -> str:
    """Build a minimal CoT diagnosis prompt."""
    return f"""You are an expert diagnostic clinician analyzing a complex medical case.

## Case Presentation
{structured_case}

## Your Task
Think step-by-step about the most likely diagnosis.
1. Identify the key clinical features, laboratory findings, imaging results, and exposures.
2. Generate a prioritized differential diagnosis with at least 10 items.
3. Select the single most likely diagnosis.

Respond in EXACTLY this JSON format:
{{
  "final_diagnosis": "[your final best diagnosis]",
  "differential_diagnosis": ["[differential 1]", "[differential 2]", "[differential 3]", "[differential 4]", "[differential 5]", "[differential 6]", "[differential 7]", "[differential 8]", "[differential 9]", "[differential 10]"],
  "reasoning": "[concise reasoning for your choice]",
  "confidence_score": [0-100]
}}
"""


def _build_scheme_a_prompt_plain(structured_case: str, light: bool = False) -> str:
    """Build a plain-text (legacy marker) CoT diagnosis prompt.

    Used for thinking-enabled mode only: JSON mode forces the reasoner's
    thinking off, so we fall back to the plain-text marker format that
    survives reasoning extraction. The prompt explicitly instructs the model
    to conclude with a final answer block as the very last section, so the
    tail-extraction in run_inference can reliably find the FINAL diagnosis.

    light=True appends a brevity instruction for light-thinking mode.
    """
    body = f"""You are an expert diagnostic clinician analyzing a complex medical case.

## Case Presentation
{structured_case}

## Your Task
Think step-by-step about the most likely diagnosis.
1. Identify the key clinical features, laboratory findings, imaging results, and exposures.
2. Generate a prioritized differential diagnosis.
3. Select the single most likely diagnosis.

Respond in EXACTLY this format:
MOST_LIKELY_DIAGNOSIS: [your final best diagnosis]
DIFFERENTIAL_DIAGNOSIS: [differential 1], [differential 2], [differential 3]
REASONING: [concise reasoning for your choice]
CONFIDENCE_SCORE: [0-100]

IMPORTANT: Think freely during your analysis, but your response MUST conclude with a final answer block as the very last section, in exactly this format:
MOST_LIKELY_DIAGNOSIS: <final best diagnosis>
DIFFERENTIAL_DIAGNOSIS: <d1>, <d2>, <d3>
REASONING: <concise reasoning>
CONFIDENCE_SCORE: <0-100>
Nothing may follow this final block.
"""
    if light:
        body += THINKING_LIGHT_INSTRUCTION
    return body


def _extract_confidence(raw: str) -> float:
    """Extract confidence score from raw output, default 50."""
    m = re.search(r'CONFIDENCE_SCORE\s*:\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 50.0


def run_scheme_a(case: Dict, temperature: float = 0.1,
                 enable_thinking: bool = False,
                 think_light: bool = False) -> Dict:
    """Run pure zero-shot baseline diagnosis on a case.

    Args:
        case: Case dict with 'Q' (case text) and 'A' (answer dict).
        temperature: LLM sampling temperature (default 0.1).
        enable_thinking: If True, run with the reasoner's thinking ON using a
            plain-text marker prompt (JSON mode forces thinking off, so it is
            incompatible with thinking). Default False preserves the JSON-mode
            behavior.

    Returns:
        Result dict compatible with run_scheme_c/run_diagnosis output.
    """
    start_time = time.time()
    case_id = case.get("case_id", "unknown")
    case_text = case.get("Q", "")

    cleaned_text = preprocess_case_text(case_text)
    structured_case = format_structured_case(cleaned_text)

    if enable_thinking:
        # Thinking ON: plain-text markers survive reasoning extraction (the
        # final answer is pulled from reasoning_content). No use_json_mode
        # (incompatible with thinking), no disable_thinking. A larger
        # max_tokens gives the reasoner room to conclude with the final block.
        # think_light caps generation (thinking + answer) and adds a brevity
        # instruction, trading deliberation depth for wall-clock time.
        prompt = _build_scheme_a_prompt_plain(structured_case, light=think_light)
        think_budget = THINKING_LIGHT_MAX_TOKENS if think_light else THINKING_MAX_TOKENS
        # Thinking takes minutes at local inference speeds; the 120s default
        # timeout aborts mid-generation and the retry loop then burns 3x more.
        raw, _ = call_llm(prompt, temperature=temperature, max_tokens=think_budget,
                          enable_thinking=True, timeout=900)
    else:
        prompt = _build_scheme_a_prompt(structured_case)
        raw, _ = call_llm(prompt, temperature=temperature, max_tokens=2048,
                          use_json_mode=True)

    parsed = parse_llm_output(raw)
    diagnosis = parsed.get("most_likely_diagnosis", "").strip()
    differential = parsed.get("differential_diagnosis", [])
    reasoning = parsed.get("reasoning", "").strip()
    # JSON mode surfaces confidence_score through parse_llm_output; fall back
    # to the legacy text parser when the model returned non-JSON output.
    confidence = parsed.get("confidence_score")
    if confidence is None:
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

    # Build ranked list for top-1/top-3/top-10 evaluation
    ranked = [diagnosis]
    for d in differential:
        if d and d.lower() not in [r.lower() for r in ranked]:
            ranked.append(d)
    ranked_diagnoses = ranked[:10]

    return {
        "case_id": case_id,
        "final_diagnosis": diagnosis,
        "final_confidence": confidence,
        "ranked_diagnoses": ranked_diagnoses,
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
        "termination_reason": "single_shot",
        "harness_verdict": "KEEP",
        "harness_contradictions": [],
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
    print(f"Running Scheme A on case: {case.get('case_id', 'unknown')}")
    result = run_scheme_a(case)
    print(f"Final diagnosis: {result['final_diagnosis']}")
    print(f"Confidence: {result['final_confidence']}")
    print(f"LLM calls: {result['total_llm_calls']}")
    print(f"Time: {result['total_time_seconds']:.1f}s")
