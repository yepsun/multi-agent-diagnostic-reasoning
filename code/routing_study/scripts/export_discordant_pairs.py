#!/usr/bin/env python3
"""导出 CPC87 上 GLM 判官口径、top-1 层面 MDT 与 A×1 的全部不一致对。

- 判定逻辑复用 recalc_main_judge.py：judge_cache_glm_v3.json，键 gold[:150]+"||"+cand[:150]。
- top-1 取各方法 top5[0]。
- 输出 5 seeds 全部 (case_id, seed) 不一致对，并按 (case_id, gold, ax1_pred, mdt_pred) 去重、保留 seed 列表。
- 病例文本来自 data/mgh_qa_dataset_merged.json 的 Q 字段。
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
J = json.loads((B / "judge_cache_glm_v3.json").read_text())
key = lambda g, c: g[:150] + "||" + c[:150]


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


def top1_correct(rec):
    k = key(rec["gold"], rec["top5"][0])
    if k not in J:
        return None
    return bool(J[k])


cases_q = {r["case_id"]: r["Q"]
           for r in json.loads((ROOT / "data" / "mgh_qa_dataset_merged.json").read_text())}

pairs = []  # per (case, seed)
for s in range(1, 6):
    ax1 = load(B / f"topn_seeds/Ax1_s{s}.jsonl")
    mdt = load(B / "topn_mdt/synthesis.jsonl" if s == 1
               else B / f"topn_mdt/s{s}/synthesis.jsonl")
    for cid in sorted(set(ax1) & set(mdt)):
        a, m = ax1[cid], mdt[cid]
        ca, cm = top1_correct(a), top1_correct(m)
        if ca is None or cm is None:
            print(f"MISSING judge entry: s{s} {cid}")
            continue
        if ca != cm:
            pairs.append({
                "case_id": cid, "seed": s, "gold": a["gold"],
                "ax1_top1": a["top5"][0], "mdt_top1": m["top5"][0],
                "ax1_correct": ca, "mdt_correct": cm,
                "direction": "MDT对/A×1错" if cm else "A×1对/MDT错",
            })

print(f"原始不一致对: {len(pairs)} "
      f"(MDT对/A×1错 {sum(1 for p in pairs if p['mdt_correct'])}, "
      f"A×1对/MDT错 {sum(1 for p in pairs if p['ax1_correct'])})")

dedup = {}
for p in pairs:
    dk = (p["case_id"], p["gold"], p["ax1_top1"], p["mdt_top1"])
    if dk not in dedup:
        dedup[dk] = {**p, "seeds": [], "case_text": cases_q.get(p["case_id"], "")}
    dedup[dk]["seeds"].append(p["seed"])

out = sorted(dedup.values(), key=lambda x: (x["direction"], x["case_id"]))
for o in out:
    o["n_seeds"] = len(o["seeds"])
    o["pair_id"] = f"{o['direction'][:3]}-{out.index(o)+1:02d}"

n_mdt_win = sum(1 for o in out if o["mdt_correct"])
print(f"去重后: {len(out)} 对 (MDT对/A×1错 {n_mdt_win}, A×1对/MDT错 {len(out)-n_mdt_win})")

dst = B / "failure_taxonomy"
dst.mkdir(exist_ok=True)
(dst / "discordant_pairs.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2))
print("written:", dst / "discordant_pairs.json")
