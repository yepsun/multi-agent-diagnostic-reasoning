#!/usr/bin/env python3
"""
Scheme PB: Perspective-guided adaptive retrieval diagnosis.

Combination of Scheme P and Scheme B: Scheme B's single-perspective initial
assessment is replaced by Scheme P's STORM-style five-perspective analysis,
while B's structured output contract, adaptive retrieval trigger, per-
candidate retrieval, NLI verification, and structured final evaluation are
reused unchanged.

Rationale: B's retrieval queries inherit the bias of its initial diagnosis.
The five-perspective (especially skeptic) front-end diversifies the initial
candidate pool so per-candidate retrieval is not anchored on a single,
possibly wrong, leading diagnosis.
"""

import os
import sys
from typing import Dict

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, script_dir)

from scheme_b import run_scheme_b  # noqa: E402

_PERSPECTIVE_ASSESSMENT_PROMPT = """You are an expert diagnostic team analyzing a complex medical case from multiple complementary perspectives, producing a structured differential assessment.

## Case Presentation
{structured_case}

## Multi-Perspective Analysis
Be concise: 1-2 sentences per perspective. Total analysis under 400 words.

**1. Attending Physician Perspective**
What is the most coherent unifying diagnosis? Which clinical features are most discriminating?

**2. Pathophysiology Perspective**
What underlying mechanism could produce this constellation of findings? Any hallmark laboratory, histologic, or molecular clues?

**3. Imaging and Laboratory Specialist Perspective**
How should the objective data (imaging, labs, vitals, procedures) be interpreted? What patterns or paradoxes stand out?

**4. Epidemiology and Exposure Perspective**
What role do demographics, geography, exposures, medications, toxins, or comorbidities play? Are there hidden risk factors in the narrative?

**5. Skeptic / Challenger Perspective**
What is the strongest argument AGAINST the leading diagnosis? What alternative diagnoses could explain MORE findings with FEWER contradictions? What findings remain unexplained?

## Structured Assessment

Synthesize the five perspectives above into the following output. Respond in EXACTLY this format:
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
- Let each perspective inform the ranking: the imaging/laboratory perspective weighs objective data; the epidemiology perspective weighs demographics and exposures; the skeptic perspective must be reflected in SKEPTIC_ALTERNATIVES.
- Confidence 80+ means highly confident; retrieval probably not needed.
- Confidence 50-79 means moderate uncertainty; retrieval would help.
- Confidence <50 means high uncertainty; retrieval is strongly needed.
- Set NEEDS_RETRIEVAL=YES if any diagnosis confidence is <80 or if the top two are close.
- MUST_NOT_MISS should include dangerous entities (e.g., malignancy, serious infection, vasculitis, immunodeficiency-related opportunistic infection) that the presentation could represent, even if they are not your top differential.
- SKEPTIC_ALTERNATIVES must NOT repeat any diagnosis already listed in DIAGNOSIS_1/2/3; give 1-2 entries, each with one key supporting feature.
- When the case involves infantile developmental regression, hypotonia, startle response, or other neurodegenerative signs, explicitly include lysosomal and storage disorders (e.g., Tay-Sachs disease/GM2 gangliosidosis, GM1 gangliosidosis, Krabbe disease, Niemann-Pick disease, metachromatic leukodystrophy) and neurodegeneration with brain iron accumulation (e.g., infantile neuroaxonal dystrophy/PLA2G6-related) in the differential or must-not-miss list.
"""


def run_scheme_pb(case: Dict, orchestrator=None, verifier=None) -> Dict:
    """Run perspective-guided adaptive retrieval diagnosis (PB = P front-end + B pipeline).

    Result dict is compatible with run_scheme_b / batch_ablation.py, plus a
    "scheme": "PB" marker.
    """
    result = run_scheme_b(
        case,
        orchestrator=orchestrator,
        verifier=verifier,
        initial_assessment_prompt=_PERSPECTIVE_ASSESSMENT_PROMPT,
    )
    result["scheme"] = "PB"
    return result
