#!/usr/bin/env python3
"""为 ER 与 MCR 数据集各抽 100 条判分对，供 GLM v3 裁判的数据集特异性人工盲评。

协议与 batch-2（sample_v4check.json，CPC）完全一致：
- YES/NO 各 50（按 LLM judge 的缓存结论分层）；
- 每层一半为"难判对"（judge 结论与 same_disease 启发式不一致的边界样本）；
- 排除已在 sample.json / sample_v4check.json 中用过的键；
- 固定 rng seed 77，可复现。

输出 judge_validity/sample_er.json 与 sample_mcr.json（schema 与旧样本一致，
另加 dataset 字段）。评者通过 judge_validity_app.py 的 ?set=er / ?set=mcr 盲评。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

from caselevel_stats import RESULTS, key_of  # noqa: E402
from webapp.clustering import same_disease  # noqa: E402

VALIDITY = RESULTS / "judge_validity"
CACHE = RESULTS / "judge_cache_glm_v3.json"
N_PER_CLASS = 50

# 各数据集的行文件来源（与论文分析所用的行目录一致）
SOURCES = {
    "er": [
        RESULTS / "topn_erreason",
        RESULTS / "topn_ax5_mod_er",
        RESULTS / "topn_ax5_mod_er_ds",
    ],
    "mcr": [
        RESULTS / "topn_mcr",
        RESULTS / "topn_mcr_seeds",
        RESULTS / "topn_ax5_mod_mcr",
    ],
}


def dataset_keys(cache):
    """返回 {数据集: {key: verdict}}，只保留缓存中已判的对。"""
    out = {}
    for ds, dirs in SOURCES.items():
        pairs = {}
        for d in dirs:
            for f in sorted(d.glob("*.jsonl")):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    gold = row.get("gold")
                    if not gold:
                        continue
                    for cand in (row.get("top5") or [])[:5]:
                        if not cand:
                            continue
                        k = key_of(gold, cand)
                        if k in cache:
                            pairs[k] = (gold, cand)
        out[ds] = pairs
        print(f"[{ds}] 行文件键 {len(pairs)} 个均在缓存中")
    return out


def main():
    cache = json.loads(CACHE.read_text())
    used = set()
    for name in ("sample.json", "sample_v4check.json"):
        used |= {t["key"] for t in json.loads((VALIDITY / name).read_text())}

    for ds, pairs in dataset_keys(cache).items():
        triples = []
        for k, (gold, cand) in pairs.items():
            if k in used or len(gold) < 4 or len(cand) < 4:
                continue
            verdict = bool(cache[k])
            triples.append({"key": k, "gold": gold, "candidate": cand,
                            "llm_verdict": verdict, "dataset": ds,
                            "heuristic_same": same_disease(gold, cand)})
        yes = [t for t in triples if t["llm_verdict"]]
        no = [t for t in triples if not t["llm_verdict"]]

        def hard(t):
            return t["llm_verdict"] != t["heuristic_same"]

        rng = random.Random(77)
        for pool in (yes, no):
            rng.shuffle(pool)
        sample = []
        for pool, n in ((yes, N_PER_CLASS), (no, N_PER_CLASS)):
            h = [t for t in pool if hard(t)]
            picked = h[:n // 2]
            seen = {id(t) for t in picked}
            for t in pool:
                if len(picked) >= n:
                    break
                if id(t) in seen:
                    continue
                picked.append(t)
                seen.add(id(t))
            sample.extend(picked)
        rng.shuffle(sample)
        for i, t in enumerate(sample):
            t["item_id"] = i
        out = VALIDITY / f"sample_{ds}.json"
        out.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
        print(f"[{ds}] 样本 {len(sample)} 条（YES "
              f"{sum(1 for t in sample if t['llm_verdict'])}/NO "
              f"{sum(1 for t in sample if not t['llm_verdict'])}，难判对 "
              f"{sum(1 for t in sample if hard(t))}）→ {out}")


if __name__ == "__main__":
    main()
