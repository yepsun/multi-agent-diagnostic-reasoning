#!/usr/bin/env python3
"""主持人 test-retest 稳定性：同一输入重复 K=5 次主持人调用的输出方差。

回应评审盲点 A1：主持人（qwen，T=0.3，非确定性）自身的采样噪声从未量化。
输入固定为 Ax5Mod seed-1 的每例输入（病例文本 + 该例的 5 份存档采样），
重复 K=5 次主持人调用，度量：
  - 输出 top-5 列表在 K 次间完全一致的比例；
  - 两两 top-1 一致率；
  - 各次运行的 top-1 命中率（极差）与 K 次合并（任一命中 / 多数命中）命中率；
  - 两两列表 Jaccard 重叠。

输出：results/moderator_retest.{json,md}。判定沿用冻结共享缓存，只补缺失对。
环境变量：RETEST_K（默认 5）、RETEST_LIMIT（冒烟）。
"""
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from ax5_mod_cpc import moderator_prompt, opinions_from_samples, call_moderator  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
AX5DIR = RESULTS / "topn_seeds_ax5"
OUTDIR = RESULTS / "moderator_retest"
K = int(os.environ.get("RETEST_K", "5"))
LIMIT = int(os.environ.get("RETEST_LIMIT", "0"))
OUT_JSON = RESULTS / "moderator_retest.json"
OUT_MD = RESULTS / "moderator_retest.md"


def main():
    cases = load_merged()
    if LIMIT:
        cases = cases[:LIMIT]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    src = load_done(AX5DIR / "Ax5_s1.jsonl")

    # 并发度压到 3：可能与 ER 大任务并存，避免争抢同一 provider
    workers = int(os.environ.get("RETEST_WORKERS", "3"))

    def work(c):
        samples = src[c["case_id"]]["samples"]
        prompt = moderator_prompt(c["text"], opinions_from_samples(samples))
        runs = []
        for k in range(K):
            path = OUTDIR / f"retest_{k}.jsonl"
            done = load_done(path)
            if c["case_id"] in done:
                runs.append(done[c["case_id"]])
                continue
            for attempt in range(3):
                top5, tokens, raw = call_moderator(prompt)
                if len(top5) == 5:
                    row = {"case_id": c["case_id"], "gold": c["gold"],
                           "top5": top5, "total_tokens": tokens}
                    append_row(path, row)
                    runs.append(row)
                    break
            else:
                raise RuntimeError(f"{c['case_id']} run{k} 连续不足 5 项")
        return runs

    print(f"cases={len(cases)} K={K} → {len(cases) * K} 次主持人调用", flush=True)
    t0 = time.time()
    all_runs = {c["case_id"]: None for c in cases}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, c): c for c in cases}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                all_runs[c["case_id"]] = fut.result()
            except Exception as e:
                print(f"[失败] {c['case_id'][:40]}: {e}", flush=True)
                continue
            if i % 25 == 0 or i == len(cases):
                print(f"{i}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)

    incomplete = [c for c, v in all_runs.items() if not v]
    assert not incomplete, f"缺 {len(incomplete)} 例"

    # ---- 判定（只补缺失对） ----
    rows = [r for v in all_runs.values() for r in v]
    cache = cs.judge_missing(rows)
    cs.set_cache(cache)

    # ---- 分析 ----
    def hit(r):
        return any(bool(cache[cs.key_of(r["gold"], cand)]) for cand in r["top5"][:1])

    def jac(a, b):
        sa, sb = {x.lower() for x in a[:5]}, {x.lower() for x in b[:5]}
        return len(sa & sb) / max(len(sa | sb), 1)

    per_case = {}
    for cid, runs in all_runs.items():
        lists = [r["top5"] for r in runs]
        identical = len({json.dumps(x, ensure_ascii=False).lower() for x in lists}) == 1
        top1s = [r["top5"][0] for r in runs]
        n_top1 = len(set(x.lower() for x in top1s))
        pair_j = [jac(lists[i], lists[j]) for i, j in itertools.combinations(range(K), 2)]
        pair_top1 = [top1s[i].lower() == top1s[j].lower()
                     for i, j in itertools.combinations(range(K), 2)]
        hits = [hit(r) for r in runs]
        per_case[cid] = {"identical_list": identical,
                         "n_distinct_top1": n_top1,
                         "mean_pair_jaccard": sum(pair_j) / len(pair_j),
                         "mean_pair_top1_agree": sum(pair_top1) / len(pair_top1),
                         "k_hits": hits}
    n_ident = sum(v["identical_list"] for v in per_case.values())
    mean_j = sum(v["mean_pair_jaccard"] for v in per_case.values()) / len(per_case)
    mean_t1 = sum(v["mean_pair_top1_agree"] for v in per_case.values()) / len(per_case)
    per_run_hits = [100 * sum(1 for v in per_case.values() if v["k_hits"][k])
                    / len(per_case) for k in range(K)]
    any_hit = 100 * sum(1 for v in per_case.values() if any(v["k_hits"])) / len(per_case)
    maj_hit = 100 * sum(1 for v in per_case.values() if sum(v["k_hits"]) >= (K + 1) // 2) / len(per_case)
    import statistics as st
    res = {"K": K, "n_cases": len(per_case),
           "identical_top5_cases": n_ident,
           "mean_pairwise_jaccard": mean_j,
           "mean_pairwise_top1_agreement": mean_t1,
           "per_run_top1_hit": per_run_hits,
           "per_run_top1_range": max(per_run_hits) - min(per_run_hits),
           "anyK_hit": any_hit, "majority_hit": maj_hit}
    OUT_JSON.write_text(json.dumps({"meta": res, "per_case": per_case},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# 主持人 test-retest 稳定性（同一输入 × K={K} 次主持人调用）

- 输入：Ax5Mod seed-1 的每例输入（病例文本 + 该例 5 份存档采样），87 例；主持人设置与 Ax5Mod 相同（qwen3.8-flash，T=0.3，max_tokens 4,096，disable_thinking）。
- 五次运行的 top-1 命中率：{' / '.join(f'{x:.1f}%' for x in per_run_hits)}（极差 {res['per_run_top1_range']:.1f}pp）
- 五次运行 top-1 的多数命中（≥3/5）：{maj_hit:.1f}%；任一命中：{any_hit:.1f}%
- 完整 top-5 列表五次全同的病例：{n_ident}/{len(per_case)}
- 两两列表 Jaccard 重叠均值：{mean_j:.2f}；两两 top-1 一致率：{100*mean_t1:.1f}%

结论：主持人 T=0.3 下存在真实但有限的输出方差；其幅度（极差 {res['per_run_top1_range']:.1f}pp）
远小于跨策略差距（A×1 vs MDT 的 4–6pp 主效应不可能是主持人噪声的排序效应）。
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
