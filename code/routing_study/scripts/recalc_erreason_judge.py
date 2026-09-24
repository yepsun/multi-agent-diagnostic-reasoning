#!/usr/bin/env python3
"""ER-Reason 结果重算：A×1/P/MDT × top-1/3/5 + 分层 + 配对 McNemar。

判官口径可选：默认 GLM 主口径（judge_cache_glm_v3.json），
--judge ds 用 DS-v3 口径；两个口径并排输出。

分层：金标签按启发式分为症状级（"unspecified"/"Complains of"/症状词）
与疾病级；ER-Reason 的 ED 诊断有大量症状级标签（"Chest pain,
unspecified type"），模型答出机制级疾病名同样会被判错，故必须分层看。
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
ER = B / "topn_erreason"
CACHES = {"GLM-v3": B / "judge_cache_glm_v3.json",
          "DS-v3": B / "judge_cache_dsflash_v3.json"}

SYMPTOM = re.compile(
    r"unspecified|complains of|^pain|swelling|fever|hypoxia|syncope|dizziness|"
    r"nausea|vomiting|weakness|fatigue|fall,|suicidal|altered mental|headache|"
    r"bleeding|shortness of breath|chest pain|abdominal pain|back pain|rash|"
    r"edema|cough", re.I)

key = lambda g, c: g[:150] + "||" + c[:150]


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i)
                       for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


SCHEMES = [("ax1", "A×1"), ("p", "P"), ("mdt_synth", "MDT")]


def main():
    rows = {f: load(ER / f"{f}.jsonl") for f, _ in SCHEMES}
    sub = {c["case_id"]: c for c in json.loads(
        (ROOT / "data" / "er_reason_subset.json").read_text())}
    ids = list(rows["ax1"])
    strata = {
        "全部": ids,
        "症状级金标签": [i for i in ids if SYMPTOM.search(sub[i]["gold"])],
        "疾病级金标签": [i for i in ids if not SYMPTOM.search(sub[i]["gold"])],
    }

    for jname, cpath in CACHES.items():
        if not cpath.exists():
            continue
        J = json.loads(cpath.read_text())

        def topk(r, k):
            return any(J.get(key(r["gold"], c)) for c in r["top5"][:k])

        missing = sum(1 for r in rows["ax1"].values() for c in r["top5"][:5]
                      if key(r["gold"], c) not in J)
        print(f"\n########## 判官 {jname}"
              f"（缺失判定 {missing}）##########")
        for sname, sids in strata.items():
            print(f"\n--- {sname} (n={len(sids)}) ---")
            for f, label in SCHEMES:
                cells = [f"top{k} {sum(1 for i in sids if topk(rows[f][i], k))/len(sids)*100:.1f}%"
                         for k in (1, 3, 5)]
                print(f"  {label:4s} " + " | ".join(cells))
            print("  配对 McNemar:")
            for k in (1, 3, 5):
                line = []
                for a, b in [("mdt_synth", "ax1"), ("p", "ax1"),
                             ("mdt_synth", "p")]:
                    ao = bo = 0
                    for i in sids:
                        ha, hb = topk(rows[a][i], k), topk(rows[b][i], k)
                        if ha and not hb:
                            ao += 1
                        elif hb and not ha:
                            bo += 1
                    line.append(f"{a} {ao}:{bo} p={mcnemar(ao, bo):.4f}")
                print(f"    top-{k}: " + " | ".join(line))


if __name__ == "__main__":
    sys.exit(main())
