#!/usr/bin/env python3
"""A×1 @ T=0.3 敏感性实验（回应"温度混杂"评审意见）。

主实验 A×1 用 T=0、P/MDT 用 T=0.3；本实验把 A×1 也放到 T=0.3：
- CPC 87 例 × 5 seeds → results/topn_seeds_ax1t03/Ax1t03_s{1..5}.jsonl
- ER-Reason 364 例 × 1 seed → results/topn_erreason_ax1t03.jsonl
提示词 A_TOPN_PROMPT（promptv2）原样冻结；推理 qwen3.8-flash，与主实验一致。

seed 列表可用 AX1T03_SEEDS 覆盖（默认 "1,2,3,4,5"，逗号或空格分隔）。
推理断点续跑（load_done）：已有的 jsonl 行自动跳过，所以把 3 seeds 扩到 5
seeds 只新跑缺失病例。

判定：GLM-5.3-flash × v3 规则（V3_PROMPT 复用 judge_v3.py），缓存
judge_cache_glm_v3.json（键 gold[:150]+"||"+cand[:150]）只补缺失对；
6 并发、150s 超时、最多 3 轮重试。

分析（GLM 口径 = 逐 seed 均值±SD + 跨 seed 合并配对 McNemar）：
- 第 1 节固定为 seeds 1–3，标题与数字与历史版本一致 —— 不删除、不改写，
  以便 paper 里已经引用过的 3-seed 结论继续可查。
- 若 AX1T03_SEEDS 含 1–3 以外的种子（默认含 4、5），追加"5-seed 口径补充"
  一节：同一口径重算 A×1(T=0.3) vs 原 A×1(T=0) / P / MDT 的 top-1/3/5 与
  配对 McNemar，并逐条说明 3-seed 口径的结论是否改变。
- ER-Reason 364 例与 seed 数无关，其小节与结论节保持原样。
产出 results/ax1_t03_sensitivity.md。

阶段可用 PHASE 环境变量控制：infer / judge / analyze / all（默认 all）。
各阶段均断点续跑；推理阶段空 top5 视为失败不落盘（下轮重跑）。
SMOKE=1 只跑前 3 例到 /tmp/ax1t03_smoke（不判定、不分析）。
"""
import json
import math
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import requests

from topn_cpc import load_done, append_row, call_top5, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from topn_mcr import call_top5 as call_top5_er  # noqa: E402
from judge_v3 import V3_PROMPT  # noqa: E402
from seeds_87 import mcnemar_exact  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
CPC_OUTDIR = RESULTS / "topn_seeds_ax1t03"
ER_OUT = RESULTS / "topn_erreason_ax1t03.jsonl"
GLM_CACHE = RESULTS / "judge_cache_glm_v3.json"
ER_DATA = ROOT / "data" / "er_reason_subset.json"
MD = RESULTS / "ax1_t03_sensitivity.md"

# seeds 1–3 是历史口径，固定单独出一节；SEEDS 是本次运行的 seeds
LEGACY_SEEDS = (1, 2, 3)
SEEDS = tuple(int(x) for x in
              os.environ.get("AX1T03_SEEDS", "1,2,3,4,5").replace(",", " ").split())
if not SEEDS:
    raise SystemExit("AX1T03_SEEDS 为空")
SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/ax1t03_smoke"))
GLM_WORKERS = 6
GLM_TIMEOUT = 150
ROUND_BUDGET = 20 * 60  # 单轮判定的时间预算（秒），超时放弃挂死连接

KEY = [l.split("=", 1)[1].strip() for l in open(ROOT / "scripts" / ".env")
       if l.startswith("ZHIPU_API_KEY=")][0]


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


# ---------- 阶段 1：推理 ----------

def run_cpc_seed(seed, cases, outdir=None):
    outdir = outdir or CPC_OUTDIR
    path = outdir / f"Ax1t03_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[Ax1t03 s{seed}] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        for attempt in range(3):
            top5, tokens = call_top5(
                A_TOPN_PROMPT.format(case_text=c["text"]), temperature=0.3)
            if top5:
                return {"case_id": c["case_id"], "gold": c["gold"],
                        "top5": top5, "total_tokens": tokens}
            print(f"[Ax1t03 s{seed}] {c['case_id'][:40]} 空输出，"
                  f"重试 {attempt + 1}/3", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续空输出")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                append_row(path, fut.result())
            except Exception as e:
                print(f"[Ax1t03 s{seed}] {c['case_id'][:40]} 失败: {e}",
                      flush=True)
                continue
            if i % 20 == 0 or i == len(todo):
                print(f"[Ax1t03 s{seed}] {i}/{len(todo)}", flush=True)


def run_er(cases):
    done = load_done(ER_OUT)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[ER Ax1t03] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c):
        for attempt in range(3):
            top5, tokens = call_top5_er(
                A_TOPN_PROMPT.format(case_text=c["text"]), 0.3)
            if top5:
                return {"case_id": c["case_id"], "gold": c["gold"],
                        "top5": top5, "total_tokens": tokens}
            print(f"[ER Ax1t03] {c['case_id'][:40]} 空输出，"
                  f"重试 {attempt + 1}/3", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续空输出")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                append_row(ER_OUT, fut.result())
            except Exception as e:
                print(f"[ER Ax1t03] {c['case_id'][:40]} 失败: {e}", flush=True)
                continue
            if i % 20 == 0 or i == len(todo):
                print(f"[ER Ax1t03] {i}/{len(todo)}", flush=True)


# ---------- 阶段 2：GLM 判定（只补缺失对） ----------

def glm_verdict(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json=body, timeout=GLM_TIMEOUT)
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def judge_missing(rows):
    cache = json.loads(GLM_CACHE.read_text()) if GLM_CACHE.exists() else {}
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            k = key_of(row["gold"], cand)
            if k not in cache:
                jobs[k] = (row["gold"], cand)
    print(f"[GLM 判定] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
    todo = list(jobs.items())
    for round_no in (1, 2, 3):
        if not todo:
            break
        errs = []

        def work(item):
            k, (gold, cand) = item
            try:
                return k, glm_verdict(gold, cand)
            except Exception:
                return k, None

        # as_completed + 轮次时间预算：个别挂死的连接不阻塞整轮
        # （ex.map 按提交顺序产出，一个挂死 future 会卡住后面全部）
        ex = ThreadPoolExecutor(GLM_WORKERS)
        futs = {ex.submit(work, item): item for item in todo}
        done_keys = set()
        try:
            n = 0
            for fut in as_completed(futs, timeout=ROUND_BUDGET):
                item = futs[fut]
                done_keys.add(item[0])
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
                if n % 100 == 0 or n == len(todo):
                    GLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                    print(f"  轮{round_no} {n}/{len(todo)} | 失败 {len(errs)}",
                          flush=True)
        except TimeoutError:
            stuck = [it for f, it in futs.items() if it[0] not in done_keys]
            print(f"  轮{round_no} 超时（{ROUND_BUDGET}s），"
                  f"{len(stuck)} 个连接挂死，放弃本轮重试之", flush=True)
            errs.extend(stuck)
        ex.shutdown(wait=False, cancel_futures=True)
        GLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        todo = errs
        print(f"[GLM 判定] 轮{round_no} 结束：失败 {len(errs)}", flush=True)
    if todo:
        print(f"[GLM 判定] 警告：{len(todo)} 对仍未解析", flush=True)
    return cache


# ---------- 阶段 3：分析 ----------

def hit_rank(row, cache):
    verdicts = [bool(cache.get(key_of(row["gold"], c)))
                for c in row["top5"][:5]]
    return next((r for r, v in enumerate(verdicts, start=1) if v), None)


def accs(rows, cache):
    n = len(rows)
    out = {}
    for k in (1, 3, 5):
        out[k] = sum(1 for r in rows
                     if (h := hit_rank(r, cache)) and h <= k) / n
    return out


def mcnemar(rows_a, rows_b, cache, k):
    """配对 McNemar：A 独对 b / B 独对 c，按 top-k 命中。"""
    b = c = 0
    by_b = {r["case_id"]: r for r in rows_b}
    for ra in rows_a:
        rb = by_b.get(ra["case_id"])
        if rb is None:
            continue
        ha = (h := hit_rank(ra, cache)) and h <= k
        hb = (h := hit_rank(rb, cache)) and h <= k
        b += bool(ha) and not hb
        c += (not ha) and bool(hb)
    return b, c, mcnemar_exact(b, c)


def fmt(m, s):
    return f"{m*100:.1f}% ± {s*100:.1f}"


def pstr(p):
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def peq(p):
    return f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"


def load_cpc(seeds):
    """各方案 × 指定 seeds 的 CPC 结果（缺失的 seed 给空列表）。"""
    paths = {
        "Ax1t03": {s: CPC_OUTDIR / f"Ax1t03_s{s}.jsonl" for s in seeds},
        "Ax1_T0": {s: RESULTS / "topn_seeds" / f"Ax1_s{s}.jsonl" for s in seeds},
        "P": {s: RESULTS / "topn_seeds" / f"P_s{s}.jsonl" for s in seeds},
        "MDT": {s: (RESULTS / "topn_mdt" / "synthesis.jsonl" if s == 1
                    else RESULTS / "topn_mdt" / f"s{s}" / "synthesis.jsonl")
                for s in seeds},
    }
    return {scheme: {s: (list(load_done(p).values()) if p.exists() else [])
                     for s, p in per.items()}
            for scheme, per in paths.items()}


def cpc_table_lines(cpc, cache, seeds, names):
    lines = ["| 方案 | top-1 | top-3 | top-5 |", "|---|---|---|---|"]
    for scheme in ("Ax1t03", "Ax1_T0", "P", "MDT"):
        per = [accs(cpc[scheme][s], cache) for s in seeds if cpc[scheme][s]]
        m = {k: st.mean(a[k] for a in per) for k in (1, 3, 5)}
        sd = {k: (st.stdev(a[k] for a in per) if len(per) > 1 else 0.0)
              for k in (1, 3, 5)}
        lines.append(f"| {names[scheme]} | {fmt(m[1], sd[1])} | "
                     f"{fmt(m[3], sd[3])} | {fmt(m[5], sd[5])} |")
    return lines


def cpc_mcnemar(cpc, cache, seeds, names, others=("Ax1_T0", "P", "MDT")):
    """A×1(T=0.3) vs 各方案的逐 seed 配对 McNemar（跨 seed 合并）。

    返回 (md 表格行, {(other, k): (b, c, p)})。
    """
    lines = ["| 对比 | top-k | A×1t03 独对 | 对方独对 | p |",
             "|---|---|---|---|---|"]
    stats = {}
    for other in others:
        for k in (1, 3, 5):
            b = c = 0
            for s in seeds:
                bi, ci, _ = mcnemar(cpc["Ax1t03"][s], cpc[other][s], cache, k)
                b += bi
                c += ci
            p = mcnemar_exact(b, c)
            stats[(other, k)] = (b, c, p)
            lines.append(f"| vs {names[other]} | top-{k} | {b} | {c} | "
                         f"{pstr(p)}{' *' if p < 0.05 else ''} |")
    return lines, stats


def verdict_change(old, new):
    """old/new = (b, c, p)：显著性状态与方向是否改变。"""
    b0, c0, p0 = old
    b1, c1, p1 = new
    if (b0 - c0) * (b1 - c1) < 0:
        return "方向反转"
    s0, s1 = p0 < 0.05, p1 < 0.05
    if s0 == s1:
        return "与 3-seed 一致" if s0 else "均不显著"
    return "3-seed 显著→5-seed 不显著" if s0 else "3-seed 不显著→5-seed 显著"


def smoke():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()[:3]
    seed = SEEDS[0]
    path = SMOKE_DIR / f"Ax1t03_s{seed}.jsonl"
    for p in (path, SMOKE_DIR / "failures.jsonl"):
        if p.exists():
            p.unlink()
    print(f"[冒烟] {len(cases)} 例 → {SMOKE_DIR}（seed {seed}，T=0.3）",
          flush=True)
    run_cpc_seed(seed, cases, SMOKE_DIR)
    rows = load_done(path)
    print(f"\n[冒烟] 产出 {len(rows)}/{len(cases)} 行")
    ok = len(rows) == len(cases)
    for c in cases:
        r = rows.get(c["case_id"])
        if not r or not r["top5"]:
            ok = False
            print(f"  {c['case_id'][:50]} | 失败：无 top5")
            continue
        print(f"  {c['case_id'][:50]}\n"
              f"    gold  : {c['gold'][:80]}\n"
              f"    top5[0]: {r['top5'][0]} | tokens={r['total_tokens']}")
    print(f"[冒烟] 通过: {ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    if os.environ.get("SMOKE") == "1":
        smoke()
        return

    phase = os.environ.get("PHASE", "all").lower()
    cpc_cases = load_merged()
    er_cases = json.loads(ER_DATA.read_text())
    # 分析用的 seeds = 历史口径 ∪ 本次指定（保证老一节永远可查）
    seeds = tuple(sorted(set(LEGACY_SEEDS) | set(SEEDS)))
    print(f"seeds: 运行 {list(SEEDS)} | 分析 {list(seeds)}"
          f"（历史口径 {list(LEGACY_SEEDS)}）", flush=True)

    if phase in ("all", "infer"):
        CPC_OUTDIR.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            run_cpc_seed(seed, cpc_cases)
        run_er(er_cases)
        # 完整性检查：无空输出
        for seed in seeds:
            rows = list(load_done(CPC_OUTDIR / f"Ax1t03_s{seed}.jsonl").values())
            assert len(rows) == len(cpc_cases), f"CPC s{seed} 缺行"
            assert all(r["top5"] for r in rows), f"CPC s{seed} 有空 top5"
        rows = list(load_done(ER_OUT).values())
        assert len(rows) == len(er_cases), "ER 缺行"
        assert all(r["top5"] for r in rows), "ER 有空 top5"
        print("[检查] 全部输出完整、无空 top5", flush=True)

    new_rows = []
    for seed in seeds:
        new_rows.extend(load_done(CPC_OUTDIR / f"Ax1t03_s{seed}.jsonl").values())
    new_rows.extend(load_done(ER_OUT).values())

    if phase in ("all", "judge"):
        judge_missing(new_rows)

    # ---- 分析 ----
    cache = json.loads(GLM_CACHE.read_text())
    missing = [key_of(r["gold"], c) for r in new_rows for c in r["top5"][:5]
               if key_of(r["gold"], c) not in cache]
    if missing:
        print(f"[分析] 警告：{len(missing)} 对无 GLM 判定，按 miss 计", flush=True)

    cpc = load_cpc(seeds)
    er = {"Ax1t03": list(load_done(ER_OUT).values()),
          "Ax1_T0": list(load_done(RESULTS / "topn_erreason" / "ax1.jsonl").values()),
          "MDT": list(load_done(RESULTS / "topn_erreason" / "mdt_synth.jsonl").values())}
    names = {"Ax1t03": "A×1 (T=0.3, 本实验)", "Ax1_T0": "A×1 (T=0, 原)",
             "P": "P (T=0.3)", "MDT": "MDT (T=0.3)"}

    # 本次是否追加"更多 seeds"补充节（seeds 1–3 之外有种子且结果齐备）
    extra = tuple(s for s in SEEDS if s not in LEGACY_SEEDS)
    supp_ok = bool(extra) and all(
        cpc["Ax1t03"][s] and cpc["Ax1_T0"][s] and cpc["P"][s] and cpc["MDT"][s]
        for s in SEEDS)

    lines = []
    lines.append("# A×1 @ T=0.3 敏感性分析（GLM-5.3-flash × v3 判官口径）\n")
    lines.append("主实验 A×1 用 T=0、P/MDT 用 T=0.3，被指出存在温度混杂。"
                 "本实验将 A×1 复跑于 T=0.3（CPC 87 例 × 3 seeds；"
                 "ER-Reason 364 例 × 1 seed），检验结论是否稳健。\n")
    if supp_ok:
        lines.append(f"**本次运行在此文件末尾追加了 "
                     f"{len(SEEDS)}-seed 口径补充节"
                     f"（AX1T03_SEEDS={','.join(map(str, SEEDS))}）；"
                     f"下文 ' 结论 ' 一节仍为历史 3-seed 口径。**\n")

    # --- 第 1 节：历史 3-seed 口径，原样保留 ---
    lines.append("\n## CPC 87 例（seeds 1–3，GLM 判官）\n")
    lines.extend(cpc_table_lines(cpc, cache, LEGACY_SEEDS, names))

    lines.append("\n### 配对 McNemar（CPC，A×1(T=0.3) vs 各方案，3×87=261 对合并）\n")
    mcn_lines, base_stats = cpc_mcnemar(cpc, cache, LEGACY_SEEDS, names)
    lines.extend(mcn_lines)

    lines.append("\n## ER-Reason 364 例（1 seed，GLM 判官）\n")
    lines.append("| 方案 | top-1 | top-3 | top-5 |")
    lines.append("|---|---|---|---|")
    for scheme in ("Ax1t03", "Ax1_T0", "MDT"):
        a = accs(er[scheme], cache)
        lines.append(f"| {names[scheme]} | {a[1]*100:.1f}% | "
                     f"{a[3]*100:.1f}% | {a[5]*100:.1f}% |")

    lines.append("\n### 配对 McNemar（ER，A×1(T=0.3) vs 各方案，364 对）\n")
    lines.append("| 对比 | top-k | A×1t03 独对 | 对方独对 | p |")
    lines.append("|---|---|---|---|---|")
    for other in ("Ax1_T0", "MDT"):
        for k in (1, 3, 5):
            b, c, p = mcnemar(er["Ax1t03"], er[other], cache, k)
            lines.append(f"| vs {names[other]} | top-{k} | {b} | {c} | "
                         f"{pstr(p)}{' *' if p < 0.05 else ''} |")

    # 核心问题结论（历史 3-seed 口径）
    # base_stats[(other, k)] = (A×1t03 独对, 对方独对, p)，故 MDT 独对取 c
    b1, c1, p1 = mcnemar(er["Ax1t03"], er["MDT"], cache, 1)
    b3, c3, _ = base_stats[("MDT", 3)]
    b5, c5, _ = base_stats[("MDT", 5)]
    p3 = mcnemar_exact(b3, c3)
    p5 = mcnemar_exact(b5, c5)
    er_a = accs(er["Ax1t03"], cache)
    er_m = accs(er["MDT"], cache)
    lines.append("\n## 结论\n")
    q1 = (f"ER 上 A×1(T=0.3) top-1 = {er_a[1]*100:.1f}% vs MDT "
          f"{er_m[1]*100:.1f}%，McNemar {peq(p1)} —— "
          + ("A×1 仍显著优于 MDT，原结论对温度混杂稳健。"
             if p1 < 0.05 and b1 > c1 else
             "差异不再显著，原' A×1 优于 MDT '的反转结论受温度混杂影响，需修正。"))
    q2 = (f"CPC 上 MDT vs A×1(T=0.3)：top-3 MDT 独对 {c3} / A×1t03 独对 {b3} "
          f"({peq(p3)})；top-5 MDT 独对 {c5} / A×1t03 独对 {b5} ({peq(p5)}) —— "
          + ("MDT 的 top-3/5 优势在同温度下保持。" if (p3 < 0.05 or p5 < 0.05)
             else "MDT 的 top-3/5 优势在同温度下不再显著。"))
    lines.append(f"1. {q1}")
    lines.append(f"2. {q2}")

    # --- 第 2 节：本次 seeds 口径（默认 5 seeds）补充 ---
    if supp_ok:
        n = len(cpc["Ax1t03"][SEEDS[0]])
        seed_span = (f"{SEEDS[0]}–{SEEDS[-1]}" if len(SEEDS) > 1 else str(SEEDS[0]))
        lines.append(f"\n## 补充：{len(SEEDS)}-seed 口径（seeds {seed_span}，"
                     f"AX1T03_SEEDS={','.join(map(str, SEEDS))}）\n")
        lines.append(f"上一节为历史 3-seed 口径（原文保留，供既有引用核对）。"
                     f"本节用同样的逐 seed 均值±SD + 跨 seed 合并配对 McNemar "
                     f"口径，把 A×1(T=0.3) 的重复运行从 3 seeds 扩到 "
                     f"{len(SEEDS)} seeds（{len(SEEDS)}×{n}="
                     f"{len(SEEDS) * n} 对合并），检验 3-seed 结论是否稳健。\n")
        lines.extend(cpc_table_lines(cpc, cache, SEEDS, names))
        lines.append(f"\n### 配对 McNemar（CPC，A×1(T=0.3) vs 各方案，"
                     f"{len(SEEDS)}×{n}={len(SEEDS) * n} 对合并）\n")
        new_lines, new_stats = cpc_mcnemar(cpc, cache, SEEDS, names)
        lines.extend(new_lines)

        lines.append(f"\n### 3-seed → {len(SEEDS)}-seed 结论是否改变\n")
        lines.append("| 对比 | top-k | 3 seeds（独对 b:c, p） | "
                     f"{len(SEEDS)} seeds（独对 b:c, p） | 判定 |")
        lines.append("|---|---|---|---|---|")
        changed = []
        for other in ("Ax1_T0", "P", "MDT"):
            for k in (1, 3, 5):
                ob, oc, op = base_stats[(other, k)]
                nb, nc, np_ = new_stats[(other, k)]
                v = verdict_change((ob, oc, op), (nb, nc, np_))
                if v in ("方向反转", "3-seed 显著→5-seed 不显著",
                         "3-seed 不显著→5-seed 显著"):
                    changed.append(f"{names[other]} top-{k}: {v}")
                lines.append(f"| vs {names[other]} | top-{k} | "
                             f"{ob}:{oc}, {pstr(op)} | {nb}:{nc}, {pstr(np_)} | {v} |")
        lines.append("")
        if changed:
            lines.append("改变显著性的条目：" + "；".join(changed) + "。"
                         "正文引用以本节 " + f"{len(SEEDS)}-seed 口径为准。")
        else:
            lines.append(f"逐条比对：{len(SEEDS)}-seed 口径下所有对比的显著性状态"
                         "与方向均与 3-seed 口径一致，原温度敏感性结论不变。")
        lines.append("\n注：ER-Reason 364 例为单 seed 设计，与 seed 数无关，"
                     "其小节与上方结论第 1 条不受本节影响。")
    elif extra:
        lines.append(f"\n## 补充：{len(SEEDS)}-seed 口径（未完成，跳过）\n")
        lines.append(f"请求 seeds {list(SEEDS)}，但部分结果文件缺失："
                     + "；".join(f"{s}=" + str(len(cpc['Ax1t03'][s]))
                                for s in SEEDS) + "。补跑后重跑 analyze。")

    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n已写入 {MD}", flush=True)


if __name__ == "__main__":
    main()
    # 判定阶段若有挂死的网络线程（非 daemon），正常退出会被 join 卡住；
    # 所有产出已落盘，直接退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
