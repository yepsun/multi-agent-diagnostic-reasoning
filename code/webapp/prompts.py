"""Web prompts for the two selectable diagnosis modes.

A_WEB: zero-shot CoT ending in a structured JSON assessment (evolution of
study scheme A with an explicit ranked top-5 requirement), run once at
temperature 0.0.

P (the "real P"): five ISOLATED expert calls (each through its own lens,
returning a ranked top-5 with a one-line rationale) followed by one
moderator synthesis call that reads the case + the five expert lists and
returns the same assessment schema as A.

Both A and the moderator return the assessment schema produced by
parse_assessment().
"""
import json
import re
from typing import Dict, List, Optional

A_WEB_PROMPT = """你是一名经验丰富的内科会诊专家，请对以下病例进行严谨的鉴别诊断分析。

分析要求：
1. 先梳理关键阳性发现与重要阴性发现；
2. 按可能性从高到低进行鉴别诊断推理，至少覆盖5个候选诊断；
3. 给出第一诊断、前5候选（第一诊断+4个主要鉴别诊断），每个候选给出支持/反对证据及最有价值的下一步检查。

输出要求：只输出一个JSON对象，不要输出JSON以外的任何文字。结构如下：
{{
  "primary_diagnosis": "第一诊断（中文，含关键分型/分期）",
  "primary_diagnosis_en": "第一诊断规范英文名（用于PubMed检索）",
  "confidence": 0到100的整数,
  "key_findings": ["关键阳性发现1", "关键阳性发现2"],
  "key_negatives": ["重要阴性发现1"],
  "differential_diagnoses": [
    {{
      "diagnosis": "鉴别诊断（中文）",
      "diagnosis_en": "规范英文名",
      "supporting": "支持点1-2条，须引用病例中的具体依据",
      "refuting": "反对点1-2条",
      "next_test": "最有价值的下一步检查"
    }}
  ],
  "reasoning_summary": "推理过程摘要（150字以内）",
  "next_steps": ["建议检查/处置1", "建议检查/处置2"]
}}
其中 differential_diagnoses 恰好提供4项，按可能性从高到低排序。

病例资料：
{case_text}
"""

def build_a_prompt(case_text: str) -> str:
    return A_WEB_PROMPT.format(case_text=case_text)


# --- P mode: 5 isolated experts + 1 moderator ---------------------------------

# (title, brief) tuples ported verbatim from routing_study/scripts/mdt_cpc.py.
EXPERT_ROLES = [
    ("Attending Internist",
     "overall clinical synthesis: which unifying diagnosis best explains the whole case, and which features are most discriminating"),
    ("Pathophysiologist",
     "underlying mechanism: what process could produce this constellation of findings, and hallmark histologic, laboratory, or molecular clues"),
    ("Imaging & Laboratory Specialist",
     "objective data: how to interpret the imaging, laboratory, and vital-sign patterns, and which patterns or paradoxes stand out"),
    ("Epidemiologist",
     "demographics, geography, travel, exposures, medications, occupation, and comorbidities — hidden risk factors in the narrative"),
    ("Skeptic / Challenger",
     "the strongest argument AGAINST the leading hypothesis: alternatives that explain more findings with fewer contradictions, and findings that remain unexplained"),
]

EXPERT_GUIDELINES = """Guidelines:
- A diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer.
- A striking organic finding may be an incidental companion finding.
- Combination diagnoses are allowed. For infections, name specific pathogens as separate items when clinically distinct."""


def build_expert_prompt(role_title: str, role_brief: str, case_text: str) -> str:
    """Prompt for one isolated expert; asks for a ranked top-5 JSON object."""
    return f"""You are the {role_title} on an MDT panel reviewing a clinical case.

Your assigned lens: {role_brief}

Through this lens, list the 5 candidate diagnoses that best fit the case, ranked most to least likely FROM YOUR PERSPECTIVE. Include candidates another specialist might overlook if the findings support them.

{EXPERT_GUIDELINES}

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "...", "rationale": "one line through your lens"}}, ... exactly 5 items]}}

Case:
{case_text}
"""


def parse_expert_top5(raw: str) -> List[Dict]:
    """Normalize a lone expert's answer into [{"diagnosis", "rationale"}]."""
    data = extract_json(raw) or {}
    items = data.get("top5") or []
    out: List[Dict] = []
    for it in items[:5]:
        if not isinstance(it, dict):
            continue
        dx = str(it.get("diagnosis") or "").strip()
        if dx and dx.lower() not in [x["diagnosis"].lower() for x in out]:
            out.append({"diagnosis": dx,
                        "rationale": str(it.get("rationale") or "").strip()})
    return out


_MODERATOR_INSTRUCTIONS = """You are the moderator of an MDT panel. Five specialists independently reviewed the clinical case below, each through their own lens, without seeing each other's opinions. Their ranked candidate lists are given.

Integrate them into the final assessment:
- Candidates supported by multiple specialists generally rise.
- A unique candidate with specific, case-grounded support must NOT be dropped merely because only one specialist listed it.
- Resolve conflicts by re-checking against the case text.
- Combination diagnoses are allowed."""

_MODERATOR_JSON_SUFFIX = """
Output requirement: output exactly one JSON code block (the only JSON block in your answer) with this structure:
```json
{
  "primary_diagnosis": "第一诊断（中文，含关键分型/分期）",
  "primary_diagnosis_en": "第一诊断规范英文名（用于PubMed检索）",
  "confidence": 0到100的整数,
  "key_findings": ["关键阳性发现1", "关键阳性发现2"],
  "key_negatives": ["重要阴性发现1"],
  "differential_diagnoses": [
    {"diagnosis": "鉴别诊断（中文）", "diagnosis_en": "规范英文名",
     "supporting": "支持点1-2条，须引用病例中的具体依据", "refuting": "反对点1-2条",
     "next_test": "最有价值的下一步检查"}
  ],
  "reasoning_summary": "综合推理摘要（200字以内）",
  "next_steps": ["建议检查/处置1", "建议检查/处置2"]
}
```
其中 differential_diagnoses 恰好提供4项，按可能性从高到低排序。只输出一个JSON代码块，其后不再输出任何文字。"""


def build_moderator_prompt(case_text: str, expert_opinions_text: str) -> str:
    """Prompt for the moderator: case + the five expert top-5 lists."""
    return (
        _MODERATOR_INSTRUCTIONS
        + "\n\nCase:\n" + case_text
        + "\n\nPanel opinions:\n" + expert_opinions_text
        + "\n" + _MODERATOR_JSON_SUFFIX
    )


def extract_json(text: str) -> Optional[Dict]:
    """Return the last parseable JSON object embedded in *text*, or None."""
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for blob in reversed(fenced):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_assessment(raw: str) -> Dict:
    """Normalize a model response into the shared assessment schema."""
    data = extract_json(raw) or {}

    def _s(v):
        return str(v).strip() if v is not None else ""

    conf = data.get("confidence")
    try:
        conf = max(0, min(100, int(round(float(conf)))))
    except (TypeError, ValueError):
        conf = None

    alts: List[Dict] = []
    for item in (data.get("differential_diagnoses") or [])[:4]:
        if not isinstance(item, dict):
            continue
        dx = _s(item.get("diagnosis"))
        if not dx:
            continue
        alts.append({
            "diagnosis": dx,
            "diagnosis_en": _s(item.get("diagnosis_en")),
            "supporting": _s(item.get("supporting")),
            "refuting": _s(item.get("refuting")),
            "next_test": _s(item.get("next_test")),
        })

    return {
        "primary_diagnosis": _s(data.get("primary_diagnosis")),
        "primary_diagnosis_en": _s(data.get("primary_diagnosis_en")),
        "confidence": conf,
        "key_findings": [x for x in (_s(v) for v in (data.get("key_findings") or [])) if x],
        "key_negatives": [x for x in (_s(v) for v in (data.get("key_negatives") or [])) if x],
        "differential_diagnoses": alts,
        "reasoning_summary": _s(data.get("reasoning_summary")),
        "next_steps": [x for x in (_s(v) for v in (data.get("next_steps") or [])) if x],
        "raw": (raw or "").strip(),
    }
