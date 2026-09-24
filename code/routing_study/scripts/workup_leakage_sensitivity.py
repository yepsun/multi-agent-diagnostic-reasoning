"""泄漏敏感性分析：排除"参考标签字面出现在客观数据中"的病例后，重算 workup 效应与交互。

对应论文 Methods 所述判据（同 flag_answer_visible_cases.py）与 Results 中的敏感性段落。
输出：routing_study/results/workup_leakage_sensitivity.json

用法：
    ./.venv/bin/python routing_study/scripts/workup_leakage_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "routing_study" / "results" / "topn_erreason"
WORK = ROOT / "routing_study" / "results" / "topn_erreason_workup"
CACHE = ROOT / "routing_study" / "results" / "judge_cache_glm_v3.json"
FLAG = ROOT / "routing_study" / "results" / "workup_answer_visible_cases.json"
OUT = ROOT / "routing_study" / "results" / "workup_leakage_sensitivity.json"

SCHEMES = {"Ax1": "ax1", "P": "p", "MDT": "mdt_synth"}
SEEDS = ("", "s2", "s3", "s4", "s5")


def key(gold: str, cand: str) -> str:
    return gold[:150] + "||" + cand[:150]


def case_rates(root: Path, scheme: str, k: int, cache: dict) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for sub in SEEDS:
        path = (root / sub if sub else root) / f"{scheme}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            hit = any(cache.get(key(r["gold"], c)) is True for c in r["top5"][:k])
            per.setdefault(r["case_id"], []).append(1.0 if hit else 0.0)
    n_seeds = max(len(v) for v in per.values())
    return {c: float(np.mean(v)) for c, v in per.items() if len(v) == n_seeds}


def main() -> None:
    cache = json.loads(CACHE.read_text())
    flagged = {c["case_id"] for c in json.loads(FLAG.read_text())["cases"]}
    results: dict = {"n_flagged": len(flagged), "seeds": list(SEEDS), "analyses": {}}

    for label, subset in (("all", None), ("excluding_answer_visible", flagged)):
        entry = {}
        for k in (1, 3, 5):
            rates = {s: (case_rates(WORK, f, k, cache), case_rates(BASE, f, k, cache))
                     for s, f in SCHEMES.items()}
            sets = {s: sorted(set(w) & set(b)) for s, (w, b) in rates.items()}
            common = set.intersection(*(set(v) for v in sets.values()))
            if subset:
                common -= subset
            common = sorted(common)
            gains = {s: np.array([rates[s][0][c] - rates[s][1][c] for c in common])
                     for s in SCHEMES}
            diff = gains["MDT"] - gains["Ax1"]
            p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            entry[f"top{k}"] = {
                "n_cases": len(common),
                "gains_pp": {s: round(float(g.mean() * 100), 2) for s, g in gains.items()},
                "interaction_MDT_minus_Ax1_pp": round(float(diff.mean() * 100), 2),
                "interaction_p": round(p, 4),
            }
        results["analyses"][label] = entry

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for label, entry in results["analyses"].items():
        print(f"\n[{label}]（top-1/3/5）")
        for k in ("top1", "top3", "top5"):
            e = entry[k]
            g = e["gains_pp"]
            print(f"  {k}: n={e['n_cases']} | d(A×1)={g['Ax1']:+.1f} d(P)={g['P']:+.1f} "
                  f"d(MDT)={g['MDT']:+.1f} | interaction {e['interaction_MDT_minus_Ax1_pp']:+.1f}pp "
                  f"p={e['interaction_p']}")
    print(f"\n写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
