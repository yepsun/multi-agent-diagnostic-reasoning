#!/usr/bin/env python3
"""groupmain 41 例 MGH CPC：A / P / ASC / PSC 的 top-3 正确率回算。

前三候选 = 主诊断 + rounds[0].differential 的前 2 项。
rank1 沿用 rejudged（v2 判定）的 match 结果；rank2/3 用同一 v2 judge 补判，
按 (gold, candidate) 去重。结果写 routing_study/results/groupmain_top3.json。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from run_static_routing import judge  # noqa: E402  (v2 语义判定)

RESULTS = ROOT / "results" / "ablation"
SCHEMES = ["A", "P", "ASC", "PSC"]
OUT = ROOT / "routing_study" / "results" / "groupmain_top3.json"


def load_rows(scheme):
    for suffix in (".rejudged.jsonl", ".jsonl"):
        p = RESULTS / f"groupmain_{scheme}{suffix}"
        if p.exists():
            rows = [json.loads(l) for l in open(p) if l.strip()]
            return rows, suffix
    return [], None


def gold_text(row):
    g = row.get("gold")
    return g if isinstance(g, str) else json.dumps(g, ensure_ascii=False)


def top3_candidates(row):
    """主诊断优先，后接 differential 中首个不重复的 2 项。"""
    r0 = (row.get("rounds") or [{}])[0]
    primary = (row.get("final_diagnosis")
               or r0.get("diagnosis") or "").strip()
    out = [primary] if primary else []
    for d in r0.get("differential") or []:
        d = (d or "").strip()
        if d and d.lower() not in [x.lower() for x in out]:
            out.append(d)
        if len(out) == 3:
            break
    return out


def key_of(gold_t, cand):
    return (gold_t[:200], cand[:200])


def main():
    data = {}
    jobs = {}
    for s in SCHEMES:
        rows, suffix = load_rows(s)
        data[s] = rows
        print(f"{s}: {len(rows)} 行 (来源 {suffix})", flush=True)
        for row in rows:
            gold_t = gold_text(row)
            for cand in top3_candidates(row)[1:]:  # rank2/3 需补判
                jobs.setdefault(key_of(gold_t, cand), (gold_t, cand))

    print(f"rank2/3 待判候选（去重后）: {len(jobs)}", flush=True)

    def work(item):
        key, (gold_t, cand) = item
        try:
            return key, bool(judge(gold_t, cand))
        except Exception:
            return key, False

    verdicts = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (key, verdict) in enumerate(ex.map(work, jobs.items()), 1):
            verdicts[key] = verdict
            if i % 25 == 0:
                print(f"  已判 {i}/{len(jobs)}", flush=True)

    summary = {}
    for s, rows in data.items():
        n = len(rows)
        top1 = 0
        top3 = 0
        pos = {"1": 0, "2": 0, "3": 0, "miss": 0}
        details = []
        for row in rows:
            gold_t = gold_text(row)
            cands = top3_candidates(row)
            hit = None
            for rank, cand in enumerate(cands[:3], start=1):
                if rank == 1:
                    ok = bool(row.get("match"))
                else:
                    ok = verdicts.get(key_of(gold_t, cand), False)
                if ok:
                    hit = rank
                    break
            if hit:
                pos[str(hit)] += 1
                top3 += 1
            else:
                pos["miss"] += 1
            if cands and bool(row.get("match")):
                top1 += 1
            details.append({"case_id": row.get("case_id"),
                            "top3": cands, "hit": hit})
        summary[s] = {"n": n, "top1": top1, "top3": top3,
                      "top1_acc": round(top1 / n, 3) if n else None,
                      "top3_acc": round(top3 / n, 3) if n else None,
                      "gold_at_rank": pos, "details": details}
        print(f"{s}: top1 {top1}/{n} ({top1/n:.1%}) | top3 {top3}/{n} "
              f"({top3/n:.1%}) | 位置分布 {pos}", flush=True)

    OUT.write_text(json.dumps({"summary": summary}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
