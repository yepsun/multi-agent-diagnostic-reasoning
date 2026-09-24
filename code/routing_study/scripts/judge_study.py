"""Judge-validity study: sampling, storage, and Cohen's kappa.

Samples (gold, candidate, llm_verdict) triples from the unified judge cache
for blinded human review. The LLM verdict is NOT shown to the annotator;
after annotation, agreement with the LLM judge yields Cohen's kappa.
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from webapp.clustering import same_disease

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"
STUDY_DIR = ROOT / "routing_study" / "results" / "judge_validity"
SAMPLE_PATH = STUDY_DIR / "sample.json"


def build_sample(n_per_class: int = 50, seed: int = 42) -> list:
    """Stratified sample: n YES + n NO, plus all hard pairs (same_disease
    disagreement or generic-head near-misses) included up to the quota."""
    cache = json.loads(CACHE.read_text())
    triples = []
    for key, verdict in cache.items():
        gold, _, cand = key.partition("||")
        if not _ or len(gold) < 4 or len(cand) < 4:
            continue
        triples.append({"key": key, "gold": gold, "candidate": cand,
                        "llm_verdict": bool(verdict),
                        "heuristic_same": same_disease(gold, cand)})
    yes = [t for t in triples if t["llm_verdict"]]
    no = [t for t in triples if not t["llm_verdict"]]
    # hard subset: LLM says YES but string heuristic disagrees, or vice versa
    # with containment, and any NO where the candidate shares a long prefix
    def hardness(t):
        return (t["llm_verdict"] != t["heuristic_same"]
                or (not t["llm_verdict"] and t["heuristic_same"]))
    hard_yes = [t for t in yes if hardness(t)]
    hard_no = [t for t in no if hardness(t)]
    rng = random.Random(seed)
    for pool in (yes, no, hard_yes, hard_no):
        rng.shuffle(pool)
    sample = []
    for pool, extra, n in ((yes, hard_yes, n_per_class),
                           (no, hard_no, n_per_class)):
        picked_keys = set()
        picked = extra[:n // 2]
        picked_keys.update(id(t) for t in picked)
        for t in pool:
            if len(picked) >= n:
                break
            if id(t) in picked_keys:
                continue
            picked.append(t)
            picked_keys.add(id(t))
        sample.extend(picked)
    rng.shuffle(sample)
    for i, t in enumerate(sample):
        t["item_id"] = i
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    return sample


def load_sample() -> list:
    if not SAMPLE_PATH.exists():
        return build_sample()
    return json.loads(SAMPLE_PATH.read_text())


def cohens_kappa(a: list, b: list) -> dict:
    """Kappa over paired verdict lists (True/False or None=missing).

    Returns {"n", "agree", "kappa", "p_a", "p_b", "pooled_yes"}.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agree": None, "kappa": None, "p_a": None,
                "p_b": None, "pooled_yes": None}
    obs = sum(1 for x, y in pairs if x == y) / n
    p_a = sum(1 for x, _ in pairs if x) / n
    p_b = sum(1 for _, y in pairs if y) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (obs - p_e) / (1 - p_e) if p_e < 1 else 1.0
    return {"n": n, "agree": round(obs, 4), "kappa": round(kappa, 4),
            "p_a": round(p_a, 4), "p_b": round(p_b, 4),
            "pooled_yes": round((p_a + p_b) / 2, 4)}


def annotator_progress(ann_path: Path) -> dict:
    if not ann_path.exists():
        return {"done": 0, "total": len(load_sample())}
    marks = json.loads(ann_path.read_text())
    return {"done": len(marks), "total": len(load_sample())}
