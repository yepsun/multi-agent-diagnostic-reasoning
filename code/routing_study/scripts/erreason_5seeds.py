import os as _os
#!/usr/bin/env python3
"""ER-Reason 5-seed 判定 + 病例级统计分析。

阶段（PHASE 环境变量：judge / analyze / all，默认 all）：

1. judge：GLM-5.3-flash（ZHIPU_API_KEY，scripts/.env）× v3 提示词
   （judge_v3.V3_PROMPT 原样复用），缓存 judge_cache_glm_v3.json
   （键 gold[:150]+"||"+cand[:150]）只补缺失对。as_completed + 每轮时间预算
   （挂死连接不阻塞整轮）、6 并发、150s 超时、最多 5 轮重试。
2. analyze：5 seeds（s1 = topn_erreason/ 顶层，s2-5 = topn_erreason/s{2..5}/）
   × A×1/P/MDT，GLM 判官口径：
   - 逐 seed top-1/3/5 + 5-seed 均值±SD；
   - 病例级口径（与 CPC/MCR 对齐，仿 stats_caselevel.py）：每病例 5-seed
     命中率 + 配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次 95% CI；
   - 多数决（>=3/5 seeds 命中）精确 McNemar；
   - 分层：症状级金标签（n=168）/ 疾病级金标签（n=196）。

输出：routing_study/results/topn_erreason/seed_summary.json
     routing_study/results/erreason_5seeds_report.md
"""
import json
import math
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

import numpy as np
import requests
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

from judge_v3 import V3_PROMPT  # noqa: E402
from recalc_erreason_judge import SYMPTOM  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
ER = RESULTS / "topn_erreason"
GLM_CACHE = RESULTS / "judge_cache_glm_v3.json"
OUT_JSON = ER / "seed_summary.json"
OUT_MD = RESULTS / "erreason_5seeds_report.md"
ER_DATA = ROOT / "data" / "er_reason_subset.json"

SEEDS = (1, 2, 3, 4, 5)
SCHEMES = [("Ax1", "ax1"), ("P", "p"), ("MDT", "mdt_synth")]
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")]
GLM_WORKERS = int(os.environ.get("GLM_WORKERS", 6))
GLM_TIMEOUT = int(os.environ.get("GLM_TIMEOUT", 150))
ROUND_BUDGET = int(os.environ.get("ROUND_BUDGET", 20 * 60))
N_BOOT = 10000
RNG = np.random.default_rng(20260917)

KEY = [l.split("=", 1)[1].strip() for l in open(ROOT / "scripts" / ".env")
       if l.startswith("ZHIPU_API_KEY=")][0]


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def load(p):
    out = {}
    for l in open(p):
        if not l.strip():
            continue
        try:
            row = json.loads(l)
        except json.JSONDecodeError:
            continue  # 推理进行中可能读到写了一半的末行
        out[row["case_id"]] = row
    return out


def seed_dir(s):
    return ER if s == 1 else ER / f"s{s}"


# ---------- 阶段 1：GLM 判定（只补缺失对） ----------

def glm_verdict(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json=body, timeout=(10, GLM_TIMEOUT))
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def judge_missing(rows, max_rounds=12):
    cache = json.loads(GLM_CACHE.read_text()) if GLM_CACHE.exists() else {}
    n_before = len(cache)
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            k = key_of(row["gold"], cand)
            if k not in cache:
                jobs[k] = (row["gold"], cand)
    print(f"[GLM 判定] 缓存 {n_before}，待判 {len(jobs)}", flush=True)
    todo = list(jobs.items())

    def save():
        # 合并式写盘：先重读磁盘缓存并入本进程结果，避免并发实例互相覆盖
        try:
            disk = json.loads(GLM_CACHE.read_text()) if GLM_CACHE.exists() else {}
        except Exception:
            disk = {}
        disk.update(cache)
        cache.update(disk)
        GLM_CACHE.write_text(json.dumps(disk, ensure_ascii=False))
    for round_no in range(1, max_rounds + 1):
        if not todo:
            break
        errs = []

        def work(item):
            k, (gold, cand) = item
            try:
                return k, glm_verdict(gold, cand)
            except Exception:
                return k, None

        # wait(FIRST_COMPLETED) 轮询 + 硬性轮次截止：比 as_completed(timeout=...)
        # 更可靠（系统休眠后 as_completed 的超时可能迟迟不触发），且每 30s
        # 打一次心跳便于发现挂死。
        import concurrent.futures as cf
        ex = ThreadPoolExecutor(GLM_WORKERS)
        pending = {ex.submit(work, item): item for item in todo}
        n = 0
        deadline = time.monotonic() + ROUND_BUDGET
        while pending:
            if time.monotonic() > deadline:
                print(f"  轮{round_no} 超时（{ROUND_BUDGET}s），"
                      f"{len(pending)} 个连接挂死，放弃本轮重试之", flush=True)
                errs.extend(pending.values())
                pending = {}
                break
            done, _ = cf.wait(list(pending), timeout=30,
                              return_when=cf.FIRST_COMPLETED)
            if not done:
                print(f"  轮{round_no} 心跳：{n}/{len(todo)} 完成，"
                      f"{len(pending)} 在途", flush=True)
                continue
            for fut in done:
                item = pending.pop(fut)
                verdict = None
                try:
                    _, verdict = fut.result()
                except Exception:
                    pass
                if verdict is None:
                    errs.append(item)
                else:
                    cache[item[0]] = verdict
                n += 1
                if n % 25 == 0 or n == len(todo):
                    save()
                    print(f"  轮{round_no} {n}/{len(todo)} | 失败 {len(errs)}",
                          flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        save()
        todo = errs
        print(f"[GLM 判定] 轮{round_no} 结束：失败 {len(errs)}", flush=True)
    if todo:
        print(f"[GLM 判定] 警告：{len(todo)} 对仍未解析", flush=True)
    print(f"[GLM 判定] 缓存新增 {len(cache) - n_before} 对 "
          f"（{n_before} -> {len(cache)}）", flush=True)
    return cache, len(cache) - n_before


# ---------- 阶段 2：分析 ----------

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def boot_ci(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def main():
    phase = os.environ.get("PHASE", "all").lower()

    runs = {}
    for s in SEEDS:
        for m, f in SCHEMES:
            p = seed_dir(s) / f"{f}.jsonl"
            if p.exists():
                runs[(m, s)] = load(p)

    sub = {c["case_id"]: c for c in json.loads(ER_DATA.read_text())}
    n_cases = len(sub)
    all_rows = [r for v in runs.values() for r in v.values()]

    added = 0
    if phase in ("all", "judge"):
        _, added = judge_missing(all_rows)
    if phase == "judge":
        return

    # ---- 分析阶段需要 5 seeds × 3 方案全部齐全 ----
    for s in SEEDS:
        for m, _f in SCHEMES:
            assert (m, s) in runs, f"{m} s{s} 输出不存在"
    for (m, s), rows in runs.items():
        assert len(rows) == n_cases, f"{m} s{s} 缺行: {len(rows)}/{n_cases}"
        assert all(r["top5"] for r in rows.values()), f"{m} s{s} 有空 top5"
    print(f"[检查] 5 seeds × 3 方案 × {n_cases} 例齐全、无空 top5", flush=True)

    J = json.loads(GLM_CACHE.read_text())
    missing = sum(1 for r in all_rows for c in r["top5"][:5]
                  if key_of(r["gold"], c) not in J)
    print(f"判官缓存 {GLM_CACHE.name}: {len(J)} 对 | 缺失 {missing}", flush=True)

    ids = sorted(runs[("Ax1", 1)])

    # hits[(m, s, cid)] = 5 个候选的判定 flag（缺失 None）
    hits = {}
    for (m, s), cases in runs.items():
        for cid, rec in cases.items():
            hits[(m, s, cid)] = [J.get(key_of(rec["gold"], c))
                                 for c in rec["top5"][:5]]

    def topk(m, s, cid, k):
        f = hits[(m, s, cid)][:k]
        if not any(x is not None for x in f):
            return None
        return any(x is True for x in f)

    def seed_acc(m, s, k, sub_ids):
        vals = [topk(m, s, c, k) for c in sub_ids]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def case_rates(m, k, sub_ids):
        """cid -> 5-seed 命中率（仅 5 seeds 全部可判定的病例）。"""
        out = {}
        for cid in sub_ids:
            vals = [topk(m, s, cid, k) for s in SEEDS]
            if any(v is None for v in vals):
                continue
            out[cid] = sum(vals) / len(SEEDS)
        return out

    def case_majority(m, k, sub_ids):
        out = {}
        for cid in sub_ids:
            vals = [topk(m, s, cid, k) for s in SEEDS]
            if any(v is None for v in vals):
                continue
            out[cid] = int(sum(vals) >= 3)
        return out

    def analyze_stratum(name, sub_ids):
        res = {"n": len(sub_ids), "per_seed": {}, "mean_sd": {}, "topk": {}}
        for m, _ in SCHEMES:
            per = {k: [seed_acc(m, s, k, sub_ids) for s in SEEDS]
                   for k in (1, 3, 5)}
            res["per_seed"][m] = {str(k): per[k] for k in (1, 3, 5)}
            res["mean_sd"][m] = {
                str(k): {"mean": st.mean(per[k]),
                         "sd": st.stdev(per[k]) if len(per[k]) > 1 else 0.0}
                for k in (1, 3, 5)}
        for k in (1, 3, 5):
            rates = {m: case_rates(m, k, sub_ids) for m, _ in SCHEMES}
            maj = {m: case_majority(m, k, sub_ids) for m, _ in SCHEMES}
            entry = {"case_rate_mean": {
                m: (st.mean(rates[m].values()) if rates[m] else None)
                for m, _ in SCHEMES}, "comparisons": {}}
            for a, b in PAIRS:
                common = sorted(set(rates[a]) & set(rates[b]))
                ra = np.array([rates[a][c] for c in common])
                rb = np.array([rates[b][c] for c in common])
                diff = ra - rb
                wp = (1.0 if np.all(diff == 0)
                      else float(wilcoxon(ra, rb, zero_method="wilcox").pvalue))
                md, lo, hi = boot_ci(diff)
                cm = sorted(set(maj[a]) & set(maj[b]))
                ao = sum(1 for c in cm if maj[a][c] and not maj[b][c])
                bo = sum(1 for c in cm if maj[b][c] and not maj[a][c])
                entry["comparisons"][f"{a}_vs_{b}"] = {
                    "n_cases": len(common),
                    "mean_rate_a": float(ra.mean()) if len(ra) else None,
                    "mean_rate_b": float(rb.mean()) if len(rb) else None,
                    "mean_diff": md, "boot95_ci": [lo, hi],
                    "wilcoxon_p": wp,
                    "majority": {"a_only": ao, "b_only": bo,
                                 "mcnemar_p": mcnemar(ao, bo)},
                }
            res["topk"][str(k)] = entry
        return res

    strata = {
        "all": ("全部", ids),
        "symptom": ("症状级金标签",
                    [i for i in ids if SYMPTOM.search(sub[i]["gold"])]),
        "disease": ("疾病级金标签",
                    [i for i in ids if not SYMPTOM.search(sub[i]["gold"])]),
    }
    results = {"judge": "GLM-5.3-flash × v3", "n_cases": n_cases,
               "judge_cache_size": len(J), "judge_missing": missing,
               "judge_pairs_added_this_run": added,
               "strata": {k: analyze_stratum(name, sids)
                          for k, (name, sids) in strata.items()}}
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"写出 {OUT_JSON}", flush=True)

    # ---------- Markdown 报告 ----------
    L = []
    L.append("# ER-Reason 364 例 × 5 seeds：A×1 / P / MDT（GLM-5.3-flash × v3 判官）\n")
    L.append("- 推理 qwen3.8-flash：A×1 T=0、P T=0.3、MDT 角色与主持人 T=0.3"
             "（与 seed 1 完全相同的提示词与温度，提示词一字未改）。")
    L.append("- seed 1 = `topn_erreason/` 顶层；seeds 2–5 = `topn_erreason/s{2..5}/`。")
    L.append(f"- 判官缓存 {GLM_CACHE.name}：{len(J)} 对（本次新增 {added}），"
             f"缺失判定 {missing}。")
    L.append("- 统计口径（与 CPC/MCR 对齐）：每病例 5-seed 命中率（0–1）→ 跨病例"
             "配对 Wilcoxon 双侧；病例级 cluster bootstrap 10,000 次 95% CI；"
             "多数决（≥3/5 seeds 命中）精确 McNemar。")
    L.append(f"- 分层：症状级金标签 n={results['strata']['symptom']['n']} / "
             f"疾病级金标签 n={results['strata']['disease']['n']}。\n")

    label = {"Ax1": "A×1", "P": "P", "MDT": "MDT"}
    for skey, (sname, _) in strata.items():
        res = results["strata"][skey]
        L.append(f"\n## {sname}（n={res['n']}）\n")
        # 逐 seed 表
        L.append("### 逐 seed top-k 命中率\n")
        L.append("| 方案 | 指标 | s1 | s2 | s3 | s4 | s5 | 均值±SD |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, _ in SCHEMES:
            for k in (1, 3, 5):
                per = res["per_seed"][m][str(k)]
                ms = res["mean_sd"][m][str(k)]
                cells = " | ".join(f"{v*100:.1f}" for v in per)
                L.append(f"| {label[m]} | top-{k} | {cells} | "
                         f"{ms['mean']*100:.1f} ± {ms['sd']*100:.1f} |")
        # 统计检验表
        L.append("\n### 病例级统计检验\n")
        L.append("| top-k | 对比 | 命中率 A vs B | 均值差 [95% CI] | "
                 "Wilcoxon p | 多数决 McNemar (a:b) p |")
        L.append("|---|---|---|---|---|---|")
        for k in (1, 3, 5):
            entry = res["topk"][str(k)]
            for pair, cmp in entry["comparisons"].items():
                a, b = pair.split("_vs_")
                lo, hi = cmp["boot95_ci"]
                mj = cmp["majority"]
                L.append(
                    f"| top-{k} | {label[a]} vs {label[b]} | "
                    f"{cmp['mean_rate_a']*100:.1f}% vs "
                    f"{cmp['mean_rate_b']*100:.1f}% | "
                    f"{cmp['mean_diff']*100:+.1f}pp "
                    f"[{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{fmt_p(cmp['wilcoxon_p'])}{sig(cmp['wilcoxon_p'])} | "
                    f"{mj['a_only']}:{mj['b_only']} "
                    f"p={fmt_p(mj['mcnemar_p'])}{sig(mj['mcnemar_p'])} |")
            cm = entry["case_rate_mean"]
            L.append(f"| top-{k} | 方案命中率 | "
                     f"A×1 {cm['Ax1']*100:.1f}% / P {cm['P']*100:.1f}% / "
                     f"MDT {cm['MDT']*100:.1f}% | — | — | — |")
        L.append("\n（* p<0.05；均值差方向 = 前者 − 后者）")

    # 核心问题
    L.append("\n## 核心问题：5 seeds 下「A×1 显著优于 MDT」是否保持\n")
    for skey, (sname, _) in strata.items():
        cmp = results["strata"][skey]["topk"]["1"]["comparisons"]["MDT_vs_Ax1"]
        wp = cmp["wilcoxon_p"]
        lo, hi = cmp["boot95_ci"]
        mj = cmp["majority"]
        direction = ("A×1 显著优于 MDT" if (wp < 0.05 and cmp["mean_diff"] < 0)
                     else "MDT 显著优于 A×1" if (wp < 0.05 and cmp["mean_diff"] > 0)
                     else "差异不显著")
        L.append(
            f"- **{sname}**（top-1，病例级 5-seed 命中率）：MDT "
            f"{cmp['mean_rate_a']*100:.1f}% vs A×1 {cmp['mean_rate_b']*100:.1f}%，"
            f"均值差 {cmp['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}]，"
            f"Wilcoxon p={fmt_p(wp)}{sig(wp)}，多数决 McNemar "
            f"{mj['a_only']}:{mj['b_only']} p={fmt_p(mj['mcnemar_p'])}"
            f" → **{direction}**")
    allk = results["strata"]["all"]["topk"]
    for k in (3, 5):
        cmp = allk[str(k)]["comparisons"]["MDT_vs_Ax1"]
        wp = cmp["wilcoxon_p"]
        direction = ("A×1 显著优于 MDT" if (wp < 0.05 and cmp["mean_diff"] < 0)
                     else "MDT 显著优于 A×1" if (wp < 0.05 and cmp["mean_diff"] > 0)
                     else "差异不显著")
        L.append(
            f"- 全部（top-{k}）：MDT {cmp['mean_rate_a']*100:.1f}% vs A×1 "
            f"{cmp['mean_rate_b']*100:.1f}%，Wilcoxon p={fmt_p(wp)}{sig(wp)}"
            f" → {direction}")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写出 {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
    # 判定阶段若有挂死的网络线程（非 daemon），正常退出会被 join 卡住；
    # 所有产出已落盘，直接退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
