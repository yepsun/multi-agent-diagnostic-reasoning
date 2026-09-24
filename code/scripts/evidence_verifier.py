#!/usr/bin/env python3
"""
evidence_verifier.py

Lightweight evidence verifier for Scheme B.

Primary mode: NLI-based entailment with a small local model
(e.g., roberta-large-mnli or ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli).

Fallback mode: when transformers/torch are unavailable, use a fast
keyword-overlap + negation heuristic so the Scheme B pipeline can still run.

Interface:
    verifier = NLIEvidenceVerifier()
    score, label = verifier.verify(hypothesis="Tay-Sachs disease", evidence_text=abstract)
"""

import os
import re
import warnings
from typing import Tuple, Optional, List

# Suppress the expected transformers pooler-weight warning when loading
# roberta-large-mnli for sequence classification.
warnings.filterwarnings(
    "ignore",
    message="Some weights of the model checkpoint at .* were not used",
    category=UserWarning,
)


class NLIEvidenceVerifier:
    """Evidence verifier supporting local NLI model (optional) and heuristic fallback."""

    # Default model candidates in order of preference
    # roberta-large-mnli is preferred because it is mirrored on ModelScope and
    # can be cached locally without hitting HuggingFace.
    DEFAULT_MODELS = [
        "roberta-large-mnli",
        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        entailment_threshold: float = 0.60,
        contradiction_threshold: float = 0.55,
        heuristic_entailment_threshold: float = 0.35,
    ):
        """
        Args:
            model_name: HuggingFace model name. If None, tries DEFAULT_MODELS in order.
            device: 'cpu', 'cuda', or None for auto.
            entailment_threshold: minimum entailment probability to count as support (NLI mode).
            contradiction_threshold: minimum contradiction probability to count as refute (NLI mode).
            heuristic_entailment_threshold: lower threshold used in heuristic fallback mode.
        """
        self.model_name = model_name
        self.device = device
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.heuristic_entailment_threshold = heuristic_entailment_threshold
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._mode = "uninitialized"  # 'nli', 'heuristic'

    def _load_nli_model(self) -> bool:
        """Try to load a local NLI model. Returns True on success."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            print(f"[Verifier] transformers/torch not available: {e}. Using heuristic fallback.")
            return False

        candidates = [self.model_name] if self.model_name else self.DEFAULT_MODELS
        for name in candidates:
            if not name:
                continue
            try:
                print(f"[Verifier] Loading NLI model: {name}")
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(name)
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self._device = device
                self._mode = "nli"
                print(f"[Verifier] NLI model loaded on {device}")
                return True
            except Exception as e:
                print(f"[Verifier] Failed to load {name}: {e}")
                continue
        return False

    def _ensure_initialized(self):
        if self._mode == "uninitialized":
            if not self._load_nli_model():
                self._mode = "heuristic"
                print("[Verifier] Initialized heuristic fallback.")

    def verify(
        self,
        hypothesis: str,
        evidence_text: str,
        entailment_threshold: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Verify whether `evidence_text` supports `hypothesis`.

        Args:
            hypothesis: hypothesis to verify.
            evidence_text: evidence text to verify against.
            entailment_threshold: override default entailment threshold for this call.
            contradiction_threshold: override default contradiction threshold for this call.

        Returns:
            (score, label) where label is one of 'entailment', 'contradiction', 'neutral'.
            score is a probability-like number in [0, 1].
        """
        self._ensure_initialized()
        if not hypothesis or not evidence_text:
            return 0.0, "neutral"
        if self._mode == "nli":
            return self._verify_nli(
                hypothesis,
                evidence_text,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
            )
        return self._verify_heuristic(
            hypothesis,
            evidence_text,
            entailment_threshold=entailment_threshold,
        )

    def _verify_nli(
        self,
        hypothesis: str,
        evidence_text: str,
        entailment_threshold: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
    ) -> Tuple[float, str]:
        import torch

        inputs = self._tokenizer(
            evidence_text[:512],  # premise
            hypothesis[:256],     # hypothesis
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # For roberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
        contradiction, neutral, entailment = probs[0], probs[1], probs[2]

        ent_thresh = entailment_threshold if entailment_threshold is not None else self.entailment_threshold
        cont_thresh = contradiction_threshold if contradiction_threshold is not None else self.contradiction_threshold

        if entailment >= ent_thresh:
            return float(entailment), "entailment"
        if contradiction >= cont_thresh:
            return float(contradiction), "contradiction"
        return float(neutral), "neutral"

    def _verify_heuristic(
        self,
        hypothesis: str,
        evidence_text: str,
        entailment_threshold: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Fast heuristic when no NLI model is available.

        Goal: estimate whether `evidence_text` supports `hypothesis`.
        Uses:
        - Disease/finding term overlap between hypothesis and evidence
        - Title vs abstract weighting
        - Negation detection
        - Supportive clinical phrases
        """
        ent_thresh = entailment_threshold if entailment_threshold is not None else self.heuristic_entailment_threshold
        full_text = (evidence_text or "").lower()
        hypothesis_lower = (hypothesis or "").lower()

        if not full_text or not hypothesis_lower:
            return 0.0, "neutral"

        # Split evidence into title and abstract if possible
        parts = evidence_text.split("\n", 1)
        title = parts[0].lower() if parts else ""
        abstract = parts[1].lower() if len(parts) > 1 else full_text

        # Extract candidate terms from hypothesis
        terms = self._extract_terms(hypothesis_lower)
        if not terms:
            return 0.0, "neutral"

        matched = 0
        negated = 0
        title_hits = 0
        for term in terms:
            # Title matching (higher value)
            for m in re.finditer(re.escape(term), title):
                window = title[max(0, m.start() - 30):min(len(title), m.end() + 30)]
                if self._is_negated(window):
                    negated += 2  # stronger penalty for title negation
                else:
                    title_hits += 1
                    matched += 2

            # Abstract/full-text matching
            for m in re.finditer(re.escape(term), abstract):
                window = abstract[max(0, m.start() - 40):min(len(abstract), m.end() + 40)]
                if self._is_negated(window):
                    negated += 1
                else:
                    matched += 1

        total_mentions = matched + negated
        if total_mentions == 0:
            return 0.0, "neutral"

        # Coverage: how many distinct terms appear at least once (negated or not)
        covered = sum(
            1 for term in terms
            if re.search(re.escape(term), full_text)
        )
        coverage = covered / len(terms)

        # Supportive cues (only count if a hypothesis term is present)
        support_cues = [
            "case report", "case series", "reported", "associated with", "caused by",
            "presented with", "diagnosed with", "consistent with", "characterized by",
            "manifested as", "clinical features", "diagnosis of",
        ]
        support_count = sum(1 for cue in support_cues if cue in full_text)
        support_score = min(1.0, support_count / 3.0)

        # Contradiction cues
        contra_cues = [
            "unlikely", "not associated", "no evidence", "ruled out", "not caused",
            "does not cause", "rarely causes", "not consistent", "negative for",
            "not diagnosed", "not present", "absence of",
        ]
        contra_count = sum(1 for cue in contra_cues if cue in full_text)
        contra_score = min(1.0, contra_count / 2.0)

        # Penalize if term appears mostly negated
        negation_ratio = negated / total_mentions if total_mentions else 0

        # The most specific (longest) term that actually appears in the title
        # should dominate the decision. terms is sorted by length descending, so
        # the first title hit is the longest matching term. This matters for
        # complex diagnosis names whose full phrase never appears verbatim in a
        # title but whose sub-phrase (e.g., "hypersensitivity pneumonitis") does.
        title_terms = [t for t in terms if t and re.search(re.escape(t), title)]
        most_specific_term = title_terms[0] if title_terms else (terms[0] if terms else "")
        specific_in_title = bool(most_specific_term) and re.search(
            re.escape(most_specific_term), title
        ) and not self._is_negated(title)

        raw_score = (
            0.25 * (matched / total_mentions)
            + 0.20 * coverage
            + 0.15 * min(1.0, title_hits)
            + 0.10 * support_score
            - 0.30 * contra_score
            - 0.25 * negation_ratio
        )
        score = max(0.0, min(1.0, raw_score))

        # Stricter entailment rules to reduce false positives:
        # 1. Most specific term in title without negation -> entailment
        # 2. No contradiction, good coverage (>=2 distinct terms), and supportive cues -> entailment
        # 3. Otherwise neutral unless strong contradiction signals
        if specific_in_title and negation_ratio < 0.3:
            return max(score, ent_thresh), "entailment"
        if (
            coverage >= 0.5
            and support_score >= 0.3
            and contra_score < 0.3
            and negation_ratio < 0.3
        ):
            return max(score, ent_thresh), "entailment"
        if contra_score >= 0.3 or negation_ratio >= 0.5:
            return score, "contradiction"
        return score, "neutral"

    def _extract_terms(self, hypothesis_lower: str) -> List[str]:
        """Extract meaningful medical terms/phrases from hypothesis.

        Complex diagnosis names (e.g., "Hypersensitivity pneumonitis due to
        Mycobacterium avium complex") are kept whole AND broken into sub-phrases
        at connector words ("due to", "with", ...). This way evidence that only
        mentions a fragment of the diagnosis (e.g., "hypersensitivity
        pneumonitis") still produces a term match.
        """
        stop = {
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
            "with", "without", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "patient", "diagnosis", "disease",
            "disorder", "syndrome", "condition", "infection", "due", "secondary", "primary",
        }

        # Connector words at which a long diagnosis phrase is split into
        # shorter, independently matchable sub-phrases.
        connector_re = re.compile(
            r"\s+(?:due to|secondary to|with|without|associated with|and|or|complicated by)\s+"
        )

        # Strip parenthetical content
        cleaned = re.sub(r'\s*\([^)]*\)', '', hypothesis_lower).strip()

        # Split on common separators
        parts = re.split(r'[,;]+', cleaned)
        candidates = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 3:
                continue
            # Keep whole phrase if it contains a medical suffix
            medical_suffixes = ("itis", "osis", "emia", "uria", "pathy", "plasia",
                                "oma", "megaly", "lysis", "penia", "ectasia", "stenosis")
            has_medical_suffix = any(part.endswith(suffix) for suffix in medical_suffixes)
            if has_medical_suffix or len(part.split()) >= 2:
                candidates.append(part)
            else:
                # Otherwise add individual meaningful words
                for w in part.split():
                    w = w.strip('.,;:!?')
                    if len(w) >= 4 and w not in stop:
                        candidates.append(w)

            # Sub-phrase generation: split the part on connector words and add
            # each fragment as an extra candidate (e.g.,
            # "X due to Y" -> "X" and "Y").
            for frag in connector_re.split(part):
                frag = frag.strip().strip('.,;:!?')
                if frag and len(frag) >= 3 and frag != part:
                    candidates.append(frag)

        # Deduplicate and prefer longer phrases
        seen = set()
        unique = []
        for c in sorted(candidates, key=len, reverse=True):
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:8]

    def _is_negated(self, window: str) -> bool:
        """Detect negation in a text window around a term occurrence."""
        negation_terms = [
            "no ", "not ", "none", "never", "without", "absent", "negative", "unlikely",
            "ruled out", "ruled-out", "non-", "not associated", "not caused", "not consistent",
            "no evidence", "not present", "absence of", "negative for", "not diagnosed",
        ]
        window_lower = window.lower()
        for neg in negation_terms:
            if neg in window_lower:
                return True
        return False

    def batch_verify(
        self,
        hypotheses: list,
        evidence_texts: list,
    ) -> list:
        """
        Batch verify a list of (hypothesis, evidence) pairs.

        Args:
            hypotheses: list of hypothesis strings
            evidence_texts: list of evidence strings (same length)

        Returns:
            list of (score, label) tuples
        """
        if len(hypotheses) != len(evidence_texts):
            raise ValueError("hypotheses and evidence_texts must have the same length")
        return [self.verify(h, e) for h, e in zip(hypotheses, evidence_texts)]


# Convenience instance (lazy-loaded)
_default_verifier = None


def get_default_verifier() -> NLIEvidenceVerifier:
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = NLIEvidenceVerifier()
    return _default_verifier


def verify_evidence(hypothesis: str, evidence_text: str) -> Tuple[float, str]:
    """One-shot verify using the default verifier."""
    return get_default_verifier().verify(hypothesis, evidence_text)


if __name__ == "__main__":
    # Quick sanity test
    v = NLIEvidenceVerifier()
    examples = [
        ("Tay-Sachs disease", "An infant with developmental regression and cherry-red spot was diagnosed with Tay-Sachs disease."),
        ("Tay-Sachs disease", "The patient did not have Tay-Sachs disease; genetic testing was negative."),
        ("Tay-Sachs disease", "Community-acquired pneumonia is a common cause of fever and cough."),
    ]
    for hyp, ev in examples:
        score, label = v.verify(hyp, ev)
        print(f"H: {hyp}\nE: {ev}\n  -> {label} ({score:.3f})\n")
