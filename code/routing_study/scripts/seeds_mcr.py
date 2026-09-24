#!/usr/bin/env python3
"""MCR 406 例：补 s2-s5（s1 = topn_mcr 既有运行），三方案 × 5 seeds 统计。

每 seed = 406 例 × (A×1 + P + MDT 全流程)。断点续跑；MDT 角色 10 并发。
判定走统一缓存（deepseek-flash，严格模式）。产出：
results/topn_mcr_seeds/seed_summary.json
"""
import json
import os
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

from topn_cpc import load_done, append_row  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_mcr import load_cases, call_top5, judge_phase, key_of  # noqa: E402
from mdt_cpc import ROLES, call_role  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

SEEDS_DIR = ROOT / "routing_study" / "results" / "topn_mcr_seeds"
S1_DIR = ROOT / "routing_study" / "results" / "topn_mcr"
ROLE_WORKERS = 10
SIMPLE_WORKERS = 6

# 与 topn_mcr.run_mdt 内部逐字一致的主持人模板
MODERATOR_PROMPT = """You are the moderator of an MDT panel. Five specialists independently reviewed the case below, each through their own lens, without seeing each other's opinions. Their ranked candidate lists are given.

Integrate them into the final ranked top-5 for the case:
- Candidates supported by multiple specialists generally rise.
- A unique candidate with specific, case-grounded support must NOT be dropped merely because only one specialist listed it.
- Resolve conflicts by re-checking against the case text.
- Combination diagnoses are allowed.

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, ... exactly 5 items]}}

Case:
{case_text}

Panel opinions:
{opinions_text}
"""


def run_simple_seed(scheme, seed, path, prompt_fn, temperature, cases):
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[{scheme} s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        top5, tokens = call_top5(prompt_fn(c), temperature)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(SIMPLE_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 25 == 0 or i == len(todo):
                print(f"[{scheme} s{seed}] {i}/{len(todo)}", flush=True)


def run_mdt_seed(seed, cases):
    seed_dir = SEEDS_DIR / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    roles_path = seed_dir / "mdt_roles.jsonl"
    synth_path = seed_dir / "mdt_synth.jsonl"

    roles_by = {}
    done_roles = set()
    if roles_path.exists():
        for line in roles_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done_roles.add((r["case_id"], r["role"]))
            roles_by.setdefault(r["case_id"], {})[r["role"]] = r["top5"]
    todo = [(c, t, b) for c in cases for t, b in ROLES
            if (c["case_id"], t) not in done_roles]
    print(f"[MDT 角色 s{seed}] 已完成 {len(done_roles)}，待跑 {len(todo)}",
          flush=True)

    def work_role(c, t, b):
        return c, t, call_role(t, b, c["text"])

    with ThreadPoolExecutor(ROLE_WORKERS) as ex:
        futs = {ex.submit(work_role, c, t, b): (c, t) for c, t, b in todo}
        done_n = 0
        for fut in as_completed(futs):
            c, t = futs[fut]
            try:
                c, t, (top5, tokens) = fut.result()
            except Exception as e:
                print(f"[MDT 角色 s{seed}] {c['case_id'][:30]} {t} 失败: {e}",
                      flush=True)
                continue
            append_row(roles_path, {"case_id": c["case_id"], "role": t,
                                    "top5": top5, "total_tokens": tokens})
            roles_by.setdefault(c["case_id"], {})[t] = top5
            done_n += 1
            if done_n % 50 == 0 or done_n == len(todo):
                print(f"[MDT 角色 s{seed}] {done_n}/{len(todo)}", flush=True)

    done = load_done(synth_path)
    todo2 = [c for c in cases
             if c["case_id"] not in done
             and len(roles_by.get(c["case_id"], {})) == len(ROLES)]
    incomplete = len(cases) - len(done) - len(todo2)
    print(f"[MDT 汇总 s{seed}] 已完成 {len(done)}，待跑 {len(todo2)}，"
          f"角色不齐跳过 {incomplete}", flush=True)

    def work_synth(c):
        lines = []
        for title, _ in ROLES:
            lines.append(f"{title}:")
            for i, item in enumerate(roles_by[c["case_id"]][title][:5], 1):
                line = f"  {i}. {item['diagnosis']}"
                if item.get("rationale"):
                    line += f" — {item['rationale']}"
                lines.append(line)
        top5, tokens = call_top5(MODERATOR_PROMPT.format(
            case_text=c["text"], opinions_text="\n".join(lines)), 0.3)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": tokens}

    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(work_synth, c): c for c in todo2}
        done_n = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                append_row(synth_path, fut.result())
            except Exception as e:
                print(f"[MDT 汇总 s{seed}] {c['case_id'][:30]} 失败: {e}",
                      flush=True)
            done_n += 1
            if done_n % 20 == 0 or done_n == len(todo2):
                print(f"[MDT 汇总 s{seed}] {done_n}/{len(todo2)}", flush=True)


def main():
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    print(f"数据集: {len(cases)} 例 × seeds s2-s5 × 3 方案", flush=True)

    for seed in range(2, 6):
        run_simple_seed("A×1", seed, SEEDS_DIR / f"Ax1_s{seed}.jsonl",
                        lambda c: A_TOPN_PROMPT.format(case_text=c["text"]),
                        0.0, cases)
        run_simple_seed("P", seed, SEEDS_DIR / f"P_s{seed}.jsonl",
                        lambda c: PERSPECTIVE_PROMPT.format(
                            structured_case=c["text"]) + P_TOPN_SUFFIX,
                        0.3, cases)
        run_mdt_seed(seed, cases)

    # s1 = topn_mcr 顶层既有运行；s2-s5 = SEEDS_DIR 子目录
    grouped = {}
    for scheme, fname in (("Ax1", "ax1.jsonl"), ("P", "p.jsonl")):
        grouped[(scheme, 1)] = list(load_done(S1_DIR / fname).values())
        for seed in range(2, 6):
            grouped[(scheme, seed)] = list(
                load_done(SEEDS_DIR / f"{scheme}_s{seed}.jsonl").values())
    grouped[("MDT", 1)] = list(load_done(S1_DIR / "mdt_synth.jsonl").values())
    for seed in range(2, 6):
        grouped[("MDT", seed)] = list(
            load_done(SEEDS_DIR / f"s{seed}" / "mdt_synth.jsonl").values())

    hits_fn = judge_phase([r for v in grouped.values() for r in v])

    print("\n===== 逐 seed =====", flush=True)
    per_seed = {}
    for (scheme, seed), rows in sorted(grouped.items()):
        n = len(rows)
        t = {1: 0, 3: 0, 5: 0}
        for row in rows:
            verdicts = hits_fn(row)
            hit = next((r for r, v in enumerate(verdicts, start=1) if v), None)
            if hit:
                t[1] += hit == 1
                t[3] += hit <= 3
                t[5] += 1
        per_seed[(scheme, seed)] = {"n": n, "top1": t[1], "top3": t[3],
                                    "top5": t[5]}
        print(f"{scheme} s{seed}: top1 {t[1]}/{n} ({t[1]/n:.1%}) "
              f"top3 {t[3]}/{n} ({t[3]/n:.1%}) top5 {t[5]}/{n} ({t[5]/n:.1%})",
              flush=True)

    print("\n===== 均值 ± SD =====", flush=True)
    stats_summary = {}
    for scheme in ("Ax1", "P", "MDT"):
        accs = {k: [per_seed[(scheme, s)][f"top{k}"] / per_seed[(scheme, s)]["n"]
                    for s in range(1, 6)] for k in (1, 3, 5)}
        stats_summary[scheme] = {
            f"top{k}": {"mean": round(st.mean(accs[k]), 4),
                        "sd": round(st.stdev(accs[k]), 4)} for k in (1, 3, 5)}
        print(scheme, stats_summary[scheme], flush=True)

    print("\n===== 配对 McNemar（top-1）=====", flush=True)
    mcn = {}
    for a_s, b_s, tag in (("MDT", "Ax1", "MDT_vs_Ax1"),
                          ("MDT", "P", "MDT_vs_P"),
                          ("Ax1", "P", "Ax1_vs_P")):
        pool_a = pool_b = 0
        per = []
        for seed in range(1, 6):
            ra = grouped[(a_s, seed)]
            rb = grouped[(b_s, seed)]
            amap = {r["case_id"]: r for r in ra}
            bmap = {r["case_id"]: r for r in rb}
            da = db = 0
            for cid, ra_row in amap.items():
                if cid not in bmap:
                    continue
                ha = any(hits_fn(ra_row)[:1])
                hb = any(hits_fn(bmap[cid])[:1])
                da += ha and not hb
                db += (not ha) and hb
            pv = mcnemar_exact(da, db)
            per.append({"seed": seed, "a_right_b_wrong": da,
                        "b_right_a_wrong": db, "p": round(pv, 4)})
            pool_a += da
            pool_b += db
            print(f"{tag} s{seed}: {a_s}独对 {da} | {b_s}独对 {db} | p={pv:.4f}",
                  flush=True)
        pv_pool = mcnemar_exact(pool_a, pool_b)
        print(f"{tag} 合并: {pool_a} vs {pool_b} | p={pv_pool:.4f}", flush=True)
        mcn[tag] = {"per_seed": per, "pooled": {"a": pool_a, "b": pool_b,
                                                "p": round(pv_pool, 4)}}

    (SEEDS_DIR / "seed_summary.json").write_text(json.dumps(
        {"per_seed": {f"{s}_s{seed}": per_seed[(s, seed)]
                      for s in ("Ax1", "P", "MDT") for seed in range(1, 6)},
         "stats": stats_summary, "mcnemar": mcn},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {SEEDS_DIR / 'seed_summary.json'}", flush=True)


def mcnemar_exact(b, c):
    import math
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


if __name__ == "__main__":
    main()
