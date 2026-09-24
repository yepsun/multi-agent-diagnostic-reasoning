#!/usr/bin/env python3
"""敏感性分析：用 v3 判官缓存重算三数据集主结果，与 v2 口径并排。

覆盖：CPC 87（A×1/P 5 seeds qwen-flash、MDT 5 seeds、qwen-max 5 seeds）+
MCR 406（A×1/P/MDT × 5 seeds）。输出 judge_v3_sensitivity.json。
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

V3 = json.loads((ROOT / "routing_study" / "results"
                 / "judge_cache_dsflash_v3.json").read_text())
R = ROOT / "routing_study" / "results"


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def metrics(rows):
    n = len(rows)
    t = {1: 0, 3: 0, 5: 0}
    for row in rows:
        gold = row["gold"]
        for i, cand in enumerate(row["top5"][:5], 1):
            if V3.get(key_of(gold, cand)):
                t[1] += i == 1
                t[3] += i <= 3
                t[5] += 1
                break
    return {"n": n, "top1": t[1], "top3": t[3], "top5": t[5]}


def seed_stats(paths_by_scheme):
    out = {}
    for scheme, path_template in paths_by_scheme.items():
        accs = {k: [] for k in (1, 3, 5)}
        n_ref = 0
        for seed in range(1, 6):
            p = Path(path_template.format(s=seed))
            rows = list(json.loads(l) for l in open(p) if l.strip()) if p.exists() else []
            if not rows:
                continue
            m = metrics(rows)
            n_ref = m["n"]
            for k in (1, 3, 5):
                accs[k].append(m[f"top{k}"] / m["n"])
        out[scheme] = {f"top{k}": {"mean": round(st.mean(accs[k]), 4),
                                   "sd": round(st.stdev(accs[k]), 4)}
                       for k in (1, 3, 5) if accs[k]}
    return out


def fmt(stats):
    parts = []
    for scheme, s in stats.items():
        cell = " / ".join(f"{s[f'top{k}']['mean']:.1%}±{s[f'top{k}']['sd']:.1%}"
                          for k in (1, 3, 5) if f"top{k}" in s)
        parts.append(f"  {scheme:6s} top1/3/5 = {cell}")
    return "\n".join(parts)


def main():
    report = {}

    blocks = {
        "CPC87_qwenflash": {
            "Ax1": str(R / "topn_seeds" / "Ax1_s{s}.jsonl"),
            "P": str(R / "topn_seeds" / "P_s{s}.jsonl"),
            "MDT": str(R / "topn_mdt" / ("synthesis.jsonl" if True else "")
                       ),
        },
    }
    # MDT 的 s1 在顶层文件，s2-5 在子目录：构造显式列表
    cpc = {
        "Ax1": [R / "topn_seeds" / f"Ax1_s{s}.jsonl" for s in range(1, 6)],
        "P": [R / "topn_seeds" / f"P_s{s}.jsonl" for s in range(1, 6)],
        "MDT": [R / "topn_mdt" / "synthesis.jsonl"] + [
            R / "topn_mdt" / f"s{s}" / "synthesis.jsonl" for s in range(2, 6)],
    }
    mcr = {
        "Ax1": [R / "topn_mcr" / "ax1.jsonl"] + [
            R / "topn_mcr_seeds" / f"Ax1_s{s}.jsonl" for s in range(2, 6)],
        "P": [R / "topn_mcr" / "p.jsonl"] + [
            R / "topn_mcr_seeds" / f"P_s{s}.jsonl" for s in range(2, 6)],
        "MDT": [R / "topn_mcr" / "mdt_synth.jsonl"] + [
            R / "topn_mcr_seeds" / f"s{s}" / "mdt_synth.jsonl" for s in range(2, 6)],
    }
    qmax = {
        "Ax1": [R / "topn_seeds_qwenmax" / f"Ax1_s{s}.jsonl" for s in range(1, 6)],
        "P": [R / "topn_seeds_qwenmax" / f"P_s{s}.jsonl" for s in range(1, 6)],
    }

    for name, paths in (("CPC87_qwenflash_v3", cpc),
                        ("MCR406_qwenflash_v3", mcr),
                        ("CPC87_qwenmax_v3", qmax)):
        print(f"== {name} ==", flush=True)
        out = {}
        for scheme, plist in paths.items():
            accs = {k: [] for k in (1, 3, 5)}
            n_ref = None
            for p in plist:
                if not p.exists():
                    print(f"  缺文件: {p}", flush=True)
                    continue
                rows = [json.loads(l) for l in open(p) if l.strip()]
                m = metrics(rows)
                n_ref = m["n"]
                for k in (1, 3, 5):
                    accs[k].append(m[f"top{k}"] / m["n"])
            out[scheme] = {f"top{k}": {"mean": round(st.mean(v), 4),
                                       "sd": round(st.stdev(v), 4)}
                           for k, v in accs.items() if v}
            cell = " / ".join(f"{out[scheme][f'top{k}']['mean']:.1%}"
                              f"±{out[scheme][f'top{k}']['sd']:.1%}"
                              for k in (1, 3, 5) if f"top{k}" in out[scheme])
            print(f"  {scheme:6s} n={n_ref}×{len(accs[1])}seeds  {cell}",
                  flush=True)
        report[name] = out

    outp = R / "judge_v3_sensitivity.json"
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"已写 {outp}", flush=True)


if __name__ == "__main__":
    main()
