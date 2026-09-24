#!/usr/bin/env python3
"""构建 ER-Reason 分层子集（400 例）→ data/er_reason_subset.json。

- 来源：data/er_reason/er_reason.csv（3,984  encounters，3,437 患者）
- 过滤：HP_Note_Text 与 primaryeddiagnosisname 均非空；每位患者只保留
  首个 encounter（去重）
- 分层：按 primarychiefcomplaintname 比例抽样（cap 50/层），seed=20260916
- 输入文本：结构化头（年龄/性别/主诉）+ **ED_Provider_Notes_Text 截断于
  "Medical Decision Making" 之前**（即主诉/现病史/既往史/过敏史/体格检查
  部分），之后的 MDM、ED Course、Final Disposition 含真实诊断与处置，
  一律不进输入（防泄漏）。H&P 列**不可用**——它是患者纵向笔记池里的
  另一份笔记，与本次 encounter 不对齐（实测：Vision disturbance 那例的
  H&P 是消化科门诊记录）。无 MDM 标记的 27 例截断于 Final Disposition/
  ED Course/Assessment/Plan 的最早出现处。
- gold = primaryeddiagnosisname（ED 最终诊断，原样保留，供判官与
  症状级/疾病级分层分析）
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "er_reason" / "er_reason.csv"
OUT = ROOT / "data" / "er_reason_subset.json"
N_TARGET = 400
SEED = 20260916
CAP_PER_CC = 50


csv.field_size_limit(10**9)


def truncate_ed_note(text):
    """截断 ED 医师笔记于评估/决策段之前（防诊断泄漏）。"""
    primary = ["Medical Decision Making", "Medical Decision-Making"]
    fallback = ["Final Disposition", "ED Course", "Assessment", "Plan"]
    cuts = [text.find(m) for m in primary if text.find(m) > 0]
    if not cuts:
        cuts = [text.find(m) for m in fallback if text.find(m) > 0]
    if cuts:
        return text[:min(cuts)].rstrip()
    return text.rstrip()


def main():
    seen_patients = set()
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ed = (r["ED_Provider_Notes_Text"] or "").strip()
            gold = (r["primaryeddiagnosisname"] or "").strip()
            pid = r["patientdurablekey"]
            if not ed or not gold or pid in seen_patients:
                continue
            seen_patients.add(pid)
            rows.append(r)
    print(f"候选 encounter（ED 笔记+gold 非空、患者去重）: {len(rows)}")

    by_cc = defaultdict(list)
    for r in rows:
        by_cc[r["primarychiefcomplaintname"]].append(r)

    rng = random.Random(SEED)
    picked = []
    for cc, group in sorted(by_cc.items(), key=lambda kv: -len(kv[1])):
        quota = min(CAP_PER_CC, round(N_TARGET * len(group) / len(rows)))
        if quota < 1:
            continue
        rng.shuffle(group)
        picked.extend(group[:quota])
    rng.shuffle(picked)
    picked = picked[:N_TARGET]

    cases = []
    for r in picked:
        note = truncate_ed_note(r["ED_Provider_Notes_Text"].strip())
        text = (f"Age: {r['Age']}\nSex: {r['sex']}\n"
                f"Chief complaint: {r['primarychiefcomplaintname']}\n\n"
                f"ED note:\n{note}")
        cases.append({
            "case_id": r["encounterkey"],
            "text": text,
            "gold": r["primaryeddiagnosisname"].strip(),
            "chief_complaint": r["primarychiefcomplaintname"],
            "acuity": r["acuitylevel"],
            "disposition": r["eddisposition"],
        })

    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    cc_dist = Counter(c["chief_complaint"] for c in cases)
    print(f"已写 {OUT}：{len(cases)} 例，{len(cc_dist)} 种主诉")
    print("top 主诉:", cc_dist.most_common(10))


if __name__ == "__main__":
    sys.exit(main())
