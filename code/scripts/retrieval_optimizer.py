#!/usr/bin/env python3
"""
Hermes-Inspired Retrieval Optimizer for Medical Diagnosis
"""

from __future__ import annotations

import os
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Configuration
RETRIEVAL_CONFIG = {
    "adaptive_rag": True,
    "complexity_threshold": 2,
    "initial_max_results": 20,
    "final_top_k": 10,
    "relevance_threshold": 30,
    "summary_max_length": 800,
    "total_context_max": 3500,
    "min_length_for_summary": 200,
    "filter_model": os.environ.get("RETRIEVAL_LLM_MODEL", "deepseek-flash"),
    "filter_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "filter_api_base": os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1"),
    "filter_rate_limit": 0.3,
}


@dataclass
class RetrievedSource:
    title: str
    url: str
    content: str
    source_type: str
    strategy_name: str = ""
    relevance_score: float = 0.0
    summary: str = ""


@dataclass
class CaseComplexity:
    score: int
    abnormal_indicators: int
    imaging_count: int
    case_length: int
    has_rare_features: bool
    
    def should_trigger_retrieval(self) -> bool:
        if not RETRIEVAL_CONFIG["adaptive_rag"]:
            return True
        return self.score >= RETRIEVAL_CONFIG["complexity_threshold"]


_last_filter_call = 0


def _call_filter_llm(prompt: str, max_tokens: int = 500) -> str:
    global _last_filter_call
    api_key = RETRIEVAL_CONFIG["filter_api_key"]
    if not api_key:
        return ""
    
    elapsed = time.time() - _last_filter_call
    if elapsed < RETRIEVAL_CONFIG["filter_rate_limit"]:
        time.sleep(RETRIEVAL_CONFIG["filter_rate_limit"] - elapsed)
    _last_filter_call = time.time()
    
    try:
        import requests
        base = RETRIEVAL_CONFIG["filter_api_base"].rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": RETRIEVAL_CONFIG["filter_model"],
            "messages": [
                {"role": "system", "content": "You are a medical literature relevance scorer. Be concise."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            # "thinking": {"type": "disabled"}
        }
        
        proxies = {"http": None, "https": None}
        resp = requests.post(url, headers=headers, json=payload, timeout=30, proxies=proxies)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"].get("content", "").strip()
        return ""
    except Exception as e:
        print(f"[Filter LLM Error] {e}")
        return ""


def assess_case_complexity(case_text: str) -> CaseComplexity:
    text_lower = case_text.lower()
    
    abnormal_markers = [
        "elevated", "decreased", "abnormal", "positive", "high", "low",
        "阳性", "阴性", "升高", "降低", "异常", "增高", "减少"
    ]
    abnormal_count = sum(1 for marker in abnormal_markers if marker in text_lower)
    
    imaging_keywords = [
        "ct", "mri", "x-ray", "ultrasound", "pet", "angiography",
        "radiograph", "tomography", "echocardiogram",
        "影像", "超声", "CT", "MRI", "X线", "造影"
    ]
    imaging_count = sum(1 for kw in imaging_keywords if kw in text_lower)
    
    case_length = len(case_text)
    length_score = min(case_length // 1000, 5)
    
    rare_features = [
        "rare", "unusual", "atypical", "unexpected", "zebra",
        "罕见", "不典型", "少见", "特殊"
    ]
    has_rare = any(f in text_lower for f in rare_features)
    
    score = min(
        (abnormal_count // 2) +
        (imaging_count // 2) +
        length_score +
        (2 if has_rare else 0),
        10
    )
    
    complexity = CaseComplexity(
        score=score,
        abnormal_indicators=abnormal_count,
        imaging_count=imaging_count,
        case_length=case_length,
        has_rare_features=has_rare
    )
    
    print(f"[Complexity] Score={score}/10 (abnormal={abnormal_count}, imaging={imaging_count}, length={case_length}, rare={has_rare})")
    return complexity


def score_relevance(source: RetrievedSource, case_text: str, diagnosis_hypothesis: str = "") -> float:
    """Score relevance of a source to the case text.

    HERMES-INSPIRED FIX: diagnosis_hypothesis is IGNORED for scoring.
    Using a diagnosis hypothesis creates confirmation bias:
    If the initial diagnosis is wrong, relevant literature gets filtered out.
    Instead, we score based on case text overlap only (disease-agnostic).
    """
    if len(source.content) < 50:
        return 0.0

    case_snippet = case_text[:500] if len(case_text) > 500 else case_text
    content_snippet = source.content[:800] if len(source.content) > 800 else source.content

    # HERMES FIX: Do NOT include diagnosis_hypothesis in the prompt
    # to avoid confirmation bias in filtering.
    # HERMES FIX 2: Escape source title/content to prevent prompt injection.
    safe_title = source.title.replace("\n", " ").replace("\r", " ")[:200]
    safe_content = content_snippet.replace("\n", " ").replace("\r", " ")[:1000]
    prompt = "Rate how relevant this medical article is to the following case.\n\nCase presentation (first 500 chars):\n" + case_snippet + "\n\nArticle title: " + safe_title + "\nArticle content: " + safe_content + "\n\nRate on a scale of 0-100 where:\n- 0-20: Completely irrelevant\n- 21-40: Marginally relevant\n- 41-60: Somewhat relevant\n- 61-80: Highly relevant\n- 81-100: Extremely relevant\n\nOutput ONLY a single integer (0-100). No explanation."

    response = _call_filter_llm(prompt, max_tokens=10)

    try:
        numbers = re.findall(r'\d+', response)
        if numbers:
            # HERMES FIX: Find the number most likely to be the score.
            # The score should be 0-100, so prefer numbers in that range.
            # If multiple numbers exist, pick the one closest to the middle (50)
            # to avoid edge cases like "2024" or "0" from scale descriptions.
            valid_scores = [(int(n), abs(int(n) - 50)) for n in numbers if 0 <= int(n) <= 100]
            if valid_scores:
                # Pick the number closest to 50 (avoids edge cases)
                score = min(valid_scores, key=lambda x: x[1])[0]
            else:
                # Fallback: take the last number (usually the actual score after explanation)
                score = int(numbers[-1])
            score = max(0, min(100, score))
            return float(score)
    except Exception:
        pass

    return _fallback_relevance_score(source, case_text)


def _fallback_relevance_score(source: RetrievedSource, case_text: str) -> float:
    case_words = set(case_text.lower().split())
    content_words = set(source.content.lower().split())
    overlap = case_words & content_words
    if not case_words:
        return 0.0
    score = min(len(overlap) / len(case_words) * 200, 100)
    return score


def filter_top_sources(sources: List[RetrievedSource], case_text: str, top_k: int = None, threshold: float = None, initial_diagnosis: str = "") -> List[RetrievedSource]:
    if not sources:
        return []

    top_k = top_k or RETRIEVAL_CONFIG["final_top_k"]
    threshold = threshold or RETRIEVAL_CONFIG["relevance_threshold"]

    print(f"[Filter] Starting with {len(sources)} sources, target top-{top_k}")

    # HERMES-INSPIRED FIX: Don't use diagnosis hypothesis for filtering
    # Using a diagnosis hypothesis creates confirmation bias:
    # If the initial diagnosis is wrong, relevant literature gets filtered out.
    # Instead, score based on case text overlap only (disease-agnostic).
    diagnosis_hypothesis = ""

    scored_sources = []
    for source in sources:
        score = score_relevance(source, case_text, diagnosis_hypothesis)
        source.relevance_score = score
        scored_sources.append(source)
        print(f"  [Score] {source.title[:60]}... -> {score:.1f}")

    scored_sources.sort(key=lambda x: x.relevance_score, reverse=True)
    filtered = [s for s in scored_sources if s.relevance_score >= threshold]

    if len(filtered) < top_k and len(scored_sources) >= top_k:
        filtered = scored_sources[:top_k]

    result = filtered[:top_k]
    print(f"[Filter] Retained {len(result)} sources (threshold={threshold})")
    return result


def summarize_article(source: RetrievedSource, case_text: str, max_length: int = None) -> str:
    max_length = max_length or RETRIEVAL_CONFIG["summary_max_length"]
    content = source.content
    
    if len(content) < RETRIEVAL_CONFIG["min_length_for_summary"]:
        return content
    
    content_snippet = content[:1500] if len(content) > 1500 else content
    case_snippet = case_text[:300] if len(case_text) > 300 else case_text
    
    prompt = "Extract ONLY the clinically relevant information from this medical article that would help diagnose the following case.\n\nCase key features: " + case_snippet + "\n\nArticle: " + content_snippet + "\n\nExtract in bullet points:\n- Key clinical features mentioned\n- Differential diagnosis considerations\n- Recommended confirmatory tests\n- Any pathognomonic findings\n\nKeep under " + str(max_length) + " characters. Be concise."

    summary = _call_filter_llm(prompt, max_tokens=max_length // 2)
    
    if summary and len(summary) > 50:
        source.summary = summary
        print(f"  [Summary] {len(content)} -> {len(summary)} chars")
        return summary
    else:
        return _fallback_summarize(content, max_length)


def _fallback_summarize(content: str, max_length: int) -> str:
    sentences = re.split(r'[.!?。！？]\s+', content)
    medical_keywords = [
        "diagnosis", "diagnostic", "differential", "confirmatory",
        "pathognomonic", "clinical", "symptom", "sign", "finding",
        "诊断", "鉴别", "确诊", "临床", "症状", "体征"
    ]
    
    selected = []
    current_length = 0
    
    for sent in sentences:
        if any(kw in sent.lower() for kw in medical_keywords):
            if current_length + len(sent) < max_length:
                selected.append(sent)
                current_length += len(sent) + 1
    
    if selected:
        return ". ".join(selected) + "."
    else:
        return content[:max_length] + "..." if len(content) > max_length else content


def compress_sources(sources: List[RetrievedSource], case_text: str) -> List[RetrievedSource]:
    print(f"[Compress] Summarizing {len(sources)} sources...")
    for source in sources:
        summarize_article(source, case_text)
    return sources


def build_retrieval_context(sources: List[RetrievedSource], max_length: int = None) -> str:
    """
    Build retrieval context with smart truncation.

    Enhanced version: Uses intelligent truncation that preserves the most
    relevant parts of each source rather than hard truncation.
    """
    max_length = max_length or RETRIEVAL_CONFIG["total_context_max"]

    if not sources:
        return ""

    parts = []
    current_length = 0

    for i, source in enumerate(sources, 1):
        content = source.summary if source.summary else source.content
        part = f"[{i}] {source.title}\n{content}\n\n"

        if current_length + len(part) > max_length:
            # Smart truncation: preserve head and tail of the content
            remaining = max_length - current_length - 50
            if remaining > 200:
                # Truncate content intelligently
                title_len = len(source.title) + 10
                content_budget = remaining - title_len - 20
                if content_budget > 100:
                    head_len = int(content_budget * 0.6)
                    tail_len = content_budget - head_len - 25
                    truncated = content[:head_len] + "\n...[truncated]...\n" + content[-tail_len:] if len(content) > content_budget else content
                    part = f"[{i}] {source.title}\n{truncated}\n\n"
                    parts.append(part)
            break

        parts.append(part)
        current_length += len(part)

    context = "".join(parts)
    print(f"[Context] Built context: {len(context)} chars from {len(parts)} sources")
    return context


def optimize_retrieval(raw_sources: List[Dict], case_text: str, case_id: str = "", initial_diagnosis: str = "") -> Tuple[str, List[RetrievedSource]]:
    """
    Enhanced retrieval optimization with medical context compression.

    Args:
        raw_sources: Raw retrieved sources
        case_text: Case presentation text
        case_id: Case identifier for logging
        initial_diagnosis: Initial diagnosis from Pass 1 (used for focus-guided compression)
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"[Optimize] Case {case_id}: Starting retrieval optimization")
    print(f"{separator}")

    sources = []
    for raw in raw_sources:
        source = RetrievedSource(
            title=raw.get("title", "Unknown"),
            url=raw.get("url", ""),
            content=raw.get("content", raw.get("abstract", "")),
            source_type=raw.get("source_type", "pubmed"),
            strategy_name=raw.get("strategy_name", ""),
            relevance_score=raw.get("relevance_score", 0.0),
        )
        sources.append(source)

    print(f"[Optimize] Received {len(sources)} raw sources")

    # Step 1: Relevance filtering
    # HERMES-INSPIRED FIX: Pass initial_diagnosis to filter, but the filter
    # should NOT use it for scoring (to avoid confirmation bias).
    # The initial_diagnosis is only used for focus-guided compression later.
    filtered = filter_top_sources(sources, case_text, initial_diagnosis=initial_diagnosis)

    # Step 2: Per-source summarization
    compressed = compress_sources(filtered, case_text)

    # Step 3: Smart context compression with focus topic (NEW)
    try:
        from medical_context_compressor import (
            MedicalContextCompressor,
            CompressionConfig,
            RetrievedSource as CompressedSource,
        )

        compressor = MedicalContextCompressor(
            config=CompressionConfig(
                threshold_percent=0.55,  # Trigger earlier for medical context
                tail_token_budget=5000,   # Preserve recent high-relevance sources
                total_context_max_chars=RETRIEVAL_CONFIG["total_context_max"],
                protect_last_n=3,
                quiet_mode=False,
            )
        )

        # Convert to compression format
        cs_sources = []
        for s in compressed:
            cs_sources.append(CompressedSource(
                title=s.title,
                url=s.url,
                content=s.summary if s.summary else s.content,
                source_type=s.source_type,
                relevance_score=s.relevance_score,
            ))

        # Apply compression with focus topic from initial diagnosis
        focus = initial_diagnosis if initial_diagnosis else None
        result = compressor.compress(cs_sources, case_text, focus_topic=focus)

        print(f"[Optimize] Compression: {result.compression_method}")
        print(f"[Optimize] Sources: {len(cs_sources)} -> {len(result.compressed_sources)}")
        print(f"[Optimize] Saved: {result.saved_chars} chars, {result.saved_tokens} tokens")
        if result.focus_topic:
            print(f"[Optimize] Focus topic: {result.focus_topic}")

        # Convert back to RetrievedSource format
        optimized_sources = []
        for s in result.compressed_sources:
            optimized_sources.append(RetrievedSource(
                title=s.title,
                url=s.url,
                content=s.content,
                source_type=s.source_type,
                relevance_score=s.relevance_score,
                summary=s.summary if hasattr(s, 'summary') else "",
            ))

        context = result.context_text

    except ImportError as e:
        print(f"[Optimize] medical_context_compressor not available ({e}), using legacy formatting")
        # Fallback to legacy build_retrieval_context
        optimized_sources = compressed
        context = build_retrieval_context(compressed)
    except Exception as e:
        print(f"[Optimize] Context compression failed ({e}), using legacy formatting")
        optimized_sources = compressed
        context = build_retrieval_context(compressed)

    print(f"[Optimize] Final context: {len(context)} chars")
    print(f"{separator}\n")

    return context, optimized_sources


def should_trigger_retrieval(case_text: str) -> Tuple[bool, CaseComplexity]:
    complexity = assess_case_complexity(case_text)
    should_trigger = complexity.should_trigger_retrieval()
    
    if not should_trigger:
        print(f"[Adaptive RAG] Case complexity {complexity.score} < threshold {RETRIEVAL_CONFIG['complexity_threshold']}, skipping retrieval")
    
    return should_trigger, complexity


def adapt_to_existing_format(sources: List[RetrievedSource]) -> List[Dict]:
    return [
        {
            "title": s.title,
            "url": s.url,
            "content": s.summary if s.summary else s.content,
            "source_type": s.source_type,
            "relevance_score": s.relevance_score,
        }
        for s in sources
    ]


if __name__ == "__main__":
    test_case = """
    A 54-year-old man with sudden cardiac arrest. 
    Elevated troponin. ST elevation on ECG.
    History of hypertension and diabetes.
    CT scan shows pulmonary embolism.
    """
    
    complexity = assess_case_complexity(test_case)
    print(f"Complexity: {complexity.score}/10")
    print(f"Should trigger retrieval: {complexity.should_trigger_retrieval()}")
