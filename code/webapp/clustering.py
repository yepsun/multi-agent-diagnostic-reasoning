"""Disease-level clustering of diagnosis strings.

String-level majority voting underestimates agreement: on the real hospital
case, five A-samples naming iMCD variants looked like 20% string consistency
but were 4/5 at disease level. Diagnoses are normalized and greedily merged
when equal after normalization or contained in one another.
"""
import re
from typing import Dict, List

# Qualifier words stripped during normalization: they describe subtype or
# staining status and must not block disease-level matching.
_QUALIFIERS = [
    "\u591a\u4e2d\u5fc3\u578b",   # 多中心型
    "\u591a\u4e2d\u5fc3",         # 多中心
    "multicentric",
    "\u7279\u53d1\u6027",         # 特发性
    "hhv8\u9634\u6027",           # hhv8阴性
    "hhv8\u9633\u6027",           # hhv8阳性
    "hhv8\u76f8\u5173",           # hhv8相关
]

# Rewrites applied after punctuation removal: abbreviation → canonical stem,
# and family-name variants onto one token.
_ALIASES = [
    ("imcd", "castleman"),
    ("mcd", "castleman"),
    ("castlemandisease", "castleman\u75c5"),  # castlemandisease → castleman病
    ("castlemandisease", "castleman\u75c5"),
    ("\u6027\u75be\u75c5", "\u75c5"),          # 性疾病 → 病
    ("\u75be\u75c5", "\u75c5"),                # 疾病 → 病
]

_PAREN_RE = re.compile("\uff08[^\uff08\uff09]*\uff09|\\([^()]*\\)")
_PUNCT_RE = re.compile("[\\s,\uff0c\u3001;\uff1b:\uff1a.\u3002\u00b7\\-\u2014_/\uff0f\\[\\]\u3010\u3011\"']+")


def normalize_name(name: str) -> str:
    s = (name or "").lower().strip()
    s = _PAREN_RE.sub("", s)
    for q in _QUALIFIERS:
        s = s.replace(q, "")
    s = _PUNCT_RE.sub("", s)
    for src, dst in _ALIASES:
        s = s.replace(src, dst)
    return s


def same_disease(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    # containment merging: the shorter name must be a meaningful stem and
    # not a generic head word (e.g. 贫血 must not match 再生障碍性贫血)
    if len(short) < 3:
        return False
    if short in _GENERIC_HEADS:
        return False
    return short in long_


_GENERIC_HEADS = {
    "\u8d2b\u8840",               # 贫血
    "\u80ba\u708e",               # 肺炎
    "\u6dcb\u5df4\u7624",         # 淋巴瘤
    "\u767d\u8840\u75c5",         # 白血病
    "\u809d\u708e",               # 肝炎
    "\u80be\u708e",               # 肾炎
    "\u7efc\u5408\u5f81",         # 综合征
    "cancer",
    "disease",
}


def cluster_diagnoses(names: List[str]) -> List[Dict]:
    """Greedy clustering preserving first-seen order.

    Returns [{"rep", "names", "indices", "count"}] where rep is the most
    frequent raw spelling in the cluster (ties: first-seen spelling) and
    indices refer to positions of valid names in *names*.
    """
    clusters: List[Dict] = []
    for idx, name in enumerate(names):
        if not name:
            continue
        for c in clusters:
            if same_disease(c["names"][0], name):
                c["names"].append(name)
                c["indices"].append(idx)
                break
        else:
            clusters.append({"names": [name], "indices": [idx]})

    out = []
    for c in clusters:
        counts: Dict[str, int] = {}
        for n in c["names"]:
            counts[n] = counts.get(n, 0) + 1
        rep = max(counts.items(), key=lambda kv: kv[1])[0]
        out.append({"rep": rep, "names": c["names"],
                    "indices": c["indices"], "count": len(c["names"])})
    return out


def aggregate_a(samples: List[Dict]) -> Dict:
    """Aggregate parsed A-sample assessments into consensus + ranked top-5.

    Returns {"n_samples", "consensus", "agreement", "clusters", "top5",
    "minority"}; consensus/top5 entries carry the evidence of their
    highest-ranked source sample. top5[0] is the consensus primary.
    """
    pairs = [(s.get("primary_diagnosis") or "", s) for s in samples]
    valid = [(n, s) for n, s in pairs if n]
    if not valid:
        return {"n_samples": len(samples), "consensus": None, "agreement": None,
                "clusters": [], "top5": [], "minority": []}

    pclusters = cluster_diagnoses([n for n, _ in valid])
    top = max(pclusters, key=lambda c: c["count"])
    win_sample = valid[top["indices"][0]][1]
    consensus = {
        "diagnosis": top["rep"],
        "votes": top["count"],
        "n_samples": len(valid),
        "confidence": win_sample.get("confidence"),
        "primary_diagnosis_en": win_sample.get("primary_diagnosis_en") or "",
        "key_findings": win_sample.get("key_findings") or [],
        "key_negatives": win_sample.get("key_negatives") or [],
        "reasoning_summary": win_sample.get("reasoning_summary") or "",
        "next_steps": win_sample.get("next_steps") or [],
    }

    # Borda-style aggregation of the alternates, excluding consensus members.
    scores: Dict[str, Dict] = {}
    for _, s in valid:
        for rank, alt in enumerate(s.get("differential_diagnoses") or []):
            dx = alt.get("diagnosis")
            if not dx or same_disease(dx, consensus["diagnosis"]):
                continue
            key = normalize_name(dx)
            entry = scores.setdefault(key, {
                "rep": dx, "score": 0.0, "votes": 0, "best": alt, "best_rank": rank,
            })
            entry["score"] += max(0.0, (4 - rank) / 4.0)
            entry["votes"] += 1
            if rank < entry["best_rank"]:
                entry["best_rank"], entry["best"] = rank, alt

    ranked = sorted(scores.values(), key=lambda e: (e["score"], e["votes"]), reverse=True)
    top5 = [dict(consensus)]
    for e in ranked[:4]:
        b = e["best"]
        top5.append({
            "diagnosis": e["rep"],
            "diagnosis_en": b.get("diagnosis_en") or "",
            "supporting": b.get("supporting") or "",
            "refuting": b.get("refuting") or "",
            "next_test": b.get("next_test") or "",
            "votes": e["votes"],
        })

    minority = [{"rep": c["rep"], "count": c["count"]}
                for c in pclusters if c is not top]

    return {
        "n_samples": len(samples),
        "consensus": consensus,
        "agreement": (top["count"] / len(valid)) if valid else None,
        "clusters": [{"rep": c["rep"], "count": c["count"]} for c in pclusters],
        "top5": top5,
        "minority": minority,
    }
