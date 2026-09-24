#!/usr/bin/env python3
"""MDT 5-seed 统计：MDT s1（topn_mdt/）+ s2-s5（topn_mdt/s{seed}/），
对照 A×1 5 seeds（topn_seeds/Ax1_s1-s5.jsonl，无需重跑）。

产出（打印 + topn_mdt/seed_summary.json）：
- MDT / A×1 逐 seed top-1/3/5 与均值±SD
- MDT vs A×1 top-1 逐 seed McNemar 精确检验 + 5-seed 合并（435 对）
- seed 符号检验（5 个 seed 中 MDT top-1 赢几个）

判官 deepseek-flash，判定缓存内容寻址、跨实验共享（mdt_cpc.judge_phase）；
预载 topn_cpc 共享缓存以复用 A×1 种子的历史判定。
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import mdt_cpc  # noqa: E402
from seeds_87 import mcnemar_exact  # noqa: E402
from topn_cpc import load_done  # noqa: E402

SEEDS_DIR = ROOT / "routing_study" / "results" / "topn_seeds"
MDT_DIR = ROOT / "routing_study" / "results" / "topn_mdt"
SEEDS = range(1, 6)


def mdt_synthesis_path(seed):
    return MDT_DIR / "synthesis.jsonl" if seed == 1 else MDT_DIR / f"s{seed}" / "synthesis.jsonl"


def hit_of(row, hits_fn):
    verdicts = hits_fn(row)
    return next((r for r, v in enumerate(verdicts, start=1) if v), None)


def metrics(rows, hits_fn):
    n = len(rows)
    t = {1: 0, 3: 0, 5: 0}
    for row in rows:
        h = hit_of(row, hits_fn)
        if h:
            for k in (1, 3, 5):
                t[k] += h <= k
    return {"n": n, **{f"top{k}": t[k] for k in (1, 3, 5)},
            **{f"top{k}_acc": round(t[k] / n, 4) if n else None for k in (1, 3, 5)}}


def mean_sd(accs):
    return {"mean": round(st.mean(accs), 4),
            "sd": round(st.stdev(accs), 4) if len(accs) > 1 else 0.0}


def main():
    # 只用 deepseek-flash 统一缓存（rejudge_dsflash.py 产出），彻底剔除 v4-pro
    # 历史判定；缺失对由 judge_phase 现场用 deepseek-flash 补判。
    unified = ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"
    mdt_cpc.SEED_CACHES[:] = [unified] if unified.exists() else []

    ax1 = {s: load_done(SEEDS_DIR / f"Ax1_s{s}.jsonl") for s in SEEDS}
    mdt = {}
    for s in SEEDS:
        p = mdt_synthesis_path(s)
        if p.exists():
            mdt[s] = load_done(p)
        else:
            print(f"[跳过] s{s} 尚无 {p}", flush=True)
    if len(mdt) < 2:
        sys.exit("MDT seed 不足，先运行 MDT_SEED=2..5 mdt_cpc.py")

    all_rows = [r for rows in list(ax1.values()) + list(mdt.values())
                for r in rows.values()]
    hits_fn = mdt_cpc.judge_phase(all_rows)

    per_seed = {}
    for s in SEEDS:
        per_seed[f"Ax1_s{s}"] = metrics(list(ax1[s].values()), hits_fn)
    for s, rows in mdt.items():
        per_seed[f"MDT_s{s}"] = metrics(list(rows.values()), hits_fn)
    for name, m in per_seed.items():
        print(f"{name}: n={m['n']} top1 {m['top1_acc']:.1%} "
              f"top3 {m['top3_acc']:.1%} top5 {m['top5_acc']:.1%}", flush=True)

    print("\n===== 均值 ± SD =====", flush=True)
    stats = {}
    for scheme, keys in (("Ax1", [f"Ax1_s{s}" for s in SEEDS]),
                         ("MDT", [f"MDT_s{s}" for s in sorted(mdt)])):
        stats[scheme] = {f"top{k}": mean_sd([per_seed[key][f"top{k}_acc"]
                                             for key in keys])
                         for k in (1, 3, 5)}
        print(scheme, stats[scheme], flush=True)

    print("\n===== MDT vs A×1 配对 McNemar（top-1，逐 seed + 合并）=====",
          flush=True)
    pooled = {"mdt_only": 0, "a_only": 0, "both": 0, "neither": 0}
    per_seed_mcn = []
    wins = 0
    for s in sorted(mdt):
        b = c = both = neither = 0
        for cid, rm in mdt[s].items():
            ra = ax1[s].get(cid)
            if ra is None:
                continue
            hm = (hit_of(rm, hits_fn) or 99) == 1
            ha = (hit_of(ra, hits_fn) or 99) == 1
            both += hm and ha
            b += hm and not ha      # MDT 独对
            c += (not hm) and ha    # A×1 独对
            neither += (not hm) and (not ha)
        pv = mcnemar_exact(b, c)
        per_seed_mcn.append({"seed": s, "mdt_right_a_wrong": b,
                             "a_right_mdt_wrong": c, "p": round(pv, 4)})
        pooled["mdt_only"] += b
        pooled["a_only"] += c
        pooled["both"] += both
        pooled["neither"] += neither
        wins += per_seed[f"MDT_s{s}"]["top1"] > per_seed[f"Ax1_s{s}"]["top1"]
        print(f"s{s}: MDT独对 {b} | A×1独对 {c} | McNemar p={pv:.4f}",
              flush=True)
    n_pairs = sum(pooled.values())
    pv_pool = mcnemar_exact(pooled["mdt_only"], pooled["a_only"])
    print(f"合并({n_pairs} 对): MDT独对 {pooled['mdt_only']} | A×1独对 "
          f"{pooled['a_only']} | McNemar p={pv_pool:.4f}", flush=True)
    print(f"符号检验: MDT 在 {wins}/{len(mdt)} 个 seed 上 top-1 更高", flush=True)

    out = {"per_seed": per_seed, "stats": stats,
           "mcnemar_per_seed": per_seed_mcn,
           "mcnemar_pooled": {**pooled, "n_pairs": n_pairs, "p": round(pv_pool, 4)},
           "sign_test": {"mdt_wins": wins, "n_seeds": len(mdt)}}
    (MDT_DIR / "seed_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {MDT_DIR / 'seed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
