#!/usr/bin/env python3
"""MDT-异构 vs 同构 vs A×1：GLM-v3 主判官口径，CPC87，seeds 1-3。

- hetero MDT：topn_mdt_hetero/（s1 顶层、s2/s3 子目录）
- homo MDT：topn_mdt/（同 seed 子集 s1-s3，保证可比）
- A×1：topn_seeds/Ax1_s{1,2,3}.jsonl
- 判定：judge_cache_glm_v3.json（缺失键记 None，不计入该 cell 分母；
  正常应先跑 judge_mdt_hetero_glm.py 补齐）
- 输出：逐 seed + 3-seed 均值±SD 的 top-1/3/5，配对 McNemar（逐 seed +
  3-seed 合并 261 对），写 topn_mdt_hetero/hetero_vs_homo_glm.json
"""
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
J = json.loads((B / "judge_cache_glm_v3.json").read_text())
key = lambda g, c: g[:150] + "||" + c[:150]

SEEDS = (1, 2, 3)


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


def mdt_path(base, s):
    return base / "synthesis.jsonl" if s == 1 else base / f"s{s}" / "synthesis.jsonl"


runs = {}
for s in SEEDS:
    runs[("hetero", s)] = load(mdt_path(B / "topn_mdt_hetero", s))
    runs[("homo", s)] = load(mdt_path(B / "topn_mdt", s))
    runs[("Ax1", s)] = load(B / f"topn_seeds/Ax1_s{s}.jsonl")

ids = sorted(runs[("Ax1", 1)])
for m in ("hetero", "homo", "Ax1"):
    for s in SEEDS:
        assert set(runs[(m, s)]) == set(ids), f"{m} s{s} 病例集不一致"
print(f"n={len(ids)}，seeds={SEEDS}，三套病例集一致", flush=True)

missing = 0
hits = {}
for (m, s), cases in runs.items():
    for cid, rec in cases.items():
        flags = []
        for c in rec["top5"][:5]:
            k = key(rec["gold"], c)
            if k in J:
                flags.append(bool(J[k]))
            else:
                missing += 1
                flags.append(None)
        hits[(m, s, cid)] = flags
print(f"GLM 判官缓存 {len(J)} 对 | 缺失 {missing}", flush=True)


def topk(m, s, cid, k):
    f = hits[(m, s, cid)][:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


NAMES = {"hetero": "MDT-hetero", "homo": "MDT-homo", "Ax1": "A×1"}
out = {"n": len(ids), "seeds": list(SEEDS), "judge": "glm-5.3-flash v3",
       "judge_cache": "judge_cache_glm_v3.json", "missing_pairs": missing,
       "per_seed": {}, "mean_sd": {}, "mcnemar": {}}

print("\n=== 逐 seed top-1/3/5（GLM-v3 口径）===")
for m in ("hetero", "homo", "Ax1"):
    for s in SEEDS:
        row = {}
        for k in (1, 3, 5):
            vals = [topk(m, s, c, k) for c in ids]
            vals = [v for v in vals if v is not None]
            row[f"top{k}"] = round(sum(vals) / len(vals), 4)
        out["per_seed"][f"{m}_s{s}"] = row
        print(f"  {NAMES[m]:10s} s{s}: " +
              " | ".join(f"top{k} {row[f'top{k}']*100:.1f}%" for k in (1, 3, 5)))

print("\n=== 3-seed 均值±SD ===")
for m in ("hetero", "homo", "Ax1"):
    out["mean_sd"][m] = {}
    cells = []
    for k in (1, 3, 5):
        per = [out["per_seed"][f"{m}_s{s}"][f"top{k}"] for s in SEEDS]
        out["mean_sd"][m][f"top{k}"] = {"mean": round(st.mean(per), 4),
                                        "sd": round(st.stdev(per), 4)}
        cells.append(f"top{k} {st.mean(per)*100:.1f}±{st.stdev(per)*100:.1f}")
    print(f"  {NAMES[m]:10s} " + " | ".join(cells))

print("\n=== 配对 McNemar（逐 seed + 3-seed 合并）===")
for k in (1, 3, 5):
    for a, b in (("hetero", "homo"), ("hetero", "Ax1"), ("homo", "Ax1")):
        pooled = {"ao": 0, "bo": 0}
        per_seed = []
        for s in SEEDS:
            ao = bo = 0
            for c in ids:
                ha, hb = topk(a, s, c, k), topk(b, s, c, k)
                if ha is None or hb is None:
                    continue
                if ha and not hb:
                    ao += 1
                elif hb and not ha:
                    bo += 1
            per_seed.append({"seed": s, f"{a}_only": ao, f"{b}_only": bo,
                             "p": round(mcnemar(ao, bo), 4)})
            pooled["ao"] += ao
            pooled["bo"] += bo
        pv = mcnemar(pooled["ao"], pooled["bo"])
        out["mcnemar"][f"top{k}_{a}_vs_{b}"] = {
            "per_seed": per_seed,
            "pooled": {f"{a}_only": pooled["ao"], f"{b}_only": pooled["bo"],
                       "n_pairs": len(ids) * len(SEEDS), "p": round(pv, 4)}}
        print(f"  top-{k} {NAMES[a]} vs {NAMES[b]}: "
              + " | ".join(f"s{r['seed']} {r[f'{a}_only']}:{r[f'{b}_only']} "
                           f"p={r['p']:.4f}" for r in per_seed)
              + f" | 合并 {pooled['ao']}:{pooled['bo']} p={pv:.4f}")

out_path = B / "topn_mdt_hetero" / "hetero_vs_homo_glm.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
print(f"\n已写入 {out_path}", flush=True)
