#!/usr/bin/env python3
"""ER-Reason「workup-informed vs presentation-only」对照评测。

两个条件（同 364 例、同 case_id、同 gold，只有输入不同）：

- baseline（presentation-only）：`routing_study/results/topn_erreason/`
- workup（临床表现 + 本次就诊客观检查）：`routing_study/results/topn_erreason_workup/`

seed 1 在目录顶层，seeds 2–5 在 `s{2..5}/` 子目录；每目录含
`ax1.jsonl`（A×1）/ `p.jsonl`（P）/ `mdt_synth.jsonl`（MDT 综合）。

阶段（PHASE 环境变量：judge / analyze / all，默认 all）：

1. judge：GLM-5.3-flash（ZHIPU_API_KEY，`scripts/.env`）× v3 提示词
   （`judge_v3.V3_PROMPT` 原样复用，不新写），缓存
   `judge_cache_glm_v3.json`（键 `gold[:150]+"||"+cand[:150]`，与既有实验
   共享内容寻址缓存），只补缺失对、绝不重判。as_completed + 轮次时间预算
   （挂死连接不阻塞整轮）、可配并发/超时、失败多轮重试、每 25 对写盘
   （合并式写盘，避免并发实例互相覆盖）。行数不足 364 的在跑文件默认不判
   （`JUDGE_INCOMPLETE=1` 可强制）。
2. analyze：逐条件逐方案 top-1/3/5（多 seed 给均值±SD）；workup vs baseline
   同 seed 配对、病例级精确 McNemar（双侧，报告独对数 + p + bootstrap CI），
   A×1 / P / MDT 三个策略分别做；关键问题 = MDT 相对 A×1 的差距是否因
   workup 数据而缩小（Δ_workup − Δ_baseline）；分层 = objective_contains_gold
   真假 / 客观数据长度 ≥200 字符 / 金标签层级（症状级 n=168、疾病级 n=196）/
   客观数据段非空（剔除 objective 为空的 34 例，n=330，敏感性分析：零信息病例
   两条件输入按构造相同，会稀释策略 × 条件交互）。
   多 seed 另给三块（口径与 stats_caselevel.py / erreason_5seeds.py 对齐，不把
   seed 当独立样本）：
   a. 病例级命中率口径：每病例 × (条件, 方案, top-k) 先在参与 seeds 上取 0–1
      命中率，再在病例集合内做配对 Wilcoxon 双侧 + 病例级 cluster bootstrap
      10,000 次 95% CI（多数决 McNemar 作为补充）；
   b. 交互检验：每病例算 d_strategy = rate_workup − rate_baseline，对
      d_MDT − d_A×1、d_P − d_A×1、d_MDT − d_P 做跨病例配对 Wilcoxon 双侧 +
      bootstrap CI，回答「加入客观数据是否缩小 MDT 相对 A×1 的差距」；
   c. 三策略 × 两条件整体交互：Friedman（病例为区组）检验三个策略的 workup
      效应是否齐一，事后两两 Wilcoxon + Holm 校正。
   单 seed 时 b/c 仍可算（病例级 d 退化为 −1/0/1），报告中明确标注「仅 1 个
   seed、不含 seed 抽样变异」，并在结论里写明是单 seed 点估计。

环境变量：
  PHASE=judge|analyze|all       默认 all
  SEEDS="1" / "1,2,3,4,5"       默认全部可用 seed
  GLM_WORKERS=6  GLM_TIMEOUT=45  ROUND_BUDGET=1200（秒/轮）
  JUDGE_INCOMPLETE=0            是否也判定尚未跑完的 jsonl

输出：`routing_study/results/workup_vs_baseline.json`（机器可读）
     `routing_study/results/workup_vs_baseline.md`（表 + 结论）

JSON 新增字段（其余字段与旧版兼容）：每个分层下
`case_level[方案].per_k[top-k]`（病例级命中率口径的 workup vs baseline）、
`interaction.topk[top-k]`（策略间交互 + Friedman + Holm 事后）。
每格都带 `n_seeds` / `n_cases`（分母）/ `n_excluded_unjudged`；判定缺失按
既有规则剔除，绝不当作判错。
"""
import json
import math
import os
import sys
import time
from concurrent.futures import (ThreadPoolExecutor, as_completed,
                                TimeoutError as FutTimeout)
from pathlib import Path

import numpy as np
import requests
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_v3 import V3_PROMPT  # noqa: E402
from recalc_erreason_judge import SYMPTOM  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
COND_DIRS = {"baseline": RESULTS / "topn_erreason",
             "workup": RESULTS / "topn_erreason_workup"}
COND_LABEL = {"baseline": "presentation-only", "workup": "workup-informed"}
GLM_CACHE = RESULTS / "judge_cache_glm_v3.json"
OUT_JSON = RESULTS / "workup_vs_baseline.json"
OUT_MD = RESULTS / "workup_vs_baseline.md"
WORKUP_DATA = ROOT / "data" / "er_reason_workup_subset.json"

SCHEMES = [("Ax1", "ax1"), ("P", "p"), ("MDT", "mdt_synth")]
SCHEME_LABEL = {"Ax1": "A×1", "P": "P", "MDT": "MDT"}
KS = (1, 3, 5)
MAX_SEED = 5
GLM_MODEL = "glm-5.3-flash"
GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_WORKERS = int(os.environ.get("GLM_WORKERS", 6))
GLM_TIMEOUT = int(os.environ.get("GLM_TIMEOUT", 45))
ROUND_BUDGET = int(os.environ.get("ROUND_BUDGET", 20 * 60))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 12))
JUDGE_INCOMPLETE = os.environ.get("JUDGE_INCOMPLETE", "0") == "1"
N_BOOT = 10000
RNG = np.random.default_rng(20260918)
# 新增分析（病例级聚合、交互检验）走另一条随机流：既有 Bootstrap 输出的抽样序列
# 与旧版脚本逐位一致，便于回归对照。
RNG_NEW = np.random.default_rng(20260919)


# ---------- 基础工具 ----------

def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def load_jsonl(path):
    """读 jsonl；忽略推理进行中可能写了一半的末行。"""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["case_id"]] = row
    return out


def api_key():
    p = ROOT / "scripts" / ".env"
    for line in p.read_text().splitlines():
        if line.startswith("ZHIPU_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"{p} 中未找到 ZHIPU_API_KEY")


def seed_dir(cond, seed):
    d = COND_DIRS[cond]
    return d if seed == 1 else d / f"s{seed}"


def available_seeds():
    out = []
    for s in range(1, MAX_SEED + 1):
        if any(seed_dir(c, s).exists() for c in COND_DIRS):
            out.append(s)
    return out


def selected_seeds():
    """SEEDS 环境变量指定；默认取两个条件里至少有一个目录存在的 seed。

    返回 (参与比较的 seeds, 请求的 seeds)；请求但目录还不存在的会打印提示并从
    比较中剔除（并在报告里写明）。
    """
    raw = os.environ.get("SEEDS", "").strip()
    requested = ([int(x) for x in raw.replace(" ", "").split(",") if x]
                 if raw else available_seeds())
    seeds = [s for s in requested if any(seed_dir(c, s).exists()
                                         for c in COND_DIRS)]
    for s in sorted(set(requested) - set(seeds)):
        print(f"[检查] 请求的 seed {s} 在两个条件下都还没有目录，已排除："
              + "；".join(str(seed_dir(c, s)) for c in COND_DIRS), flush=True)
    if not seeds:
        raise RuntimeError("没有任何可用 seed 目录")
    for c in COND_DIRS:
        for s in seeds:
            if not seed_dir(c, s).exists():
                print(f"[检查] {COND_LABEL[c]} seed {s} 目录不存在："
                      f"{seed_dir(c, s)}（该条件下此 seed 全部记为缺失）",
                      flush=True)
    return sorted(seeds), sorted(set(requested))


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def pc(x, nd=1):
    return "NA" if x is None else f"{x * 100:.{nd}f}"


def pp(x, nd=1):
    return "NA" if x is None else f"{x * 100:+.{nd}f}"


def num(x, nd=1):
    return "NA" if x is None else f"{x * 100:.{nd}f}"


# ---------- 阶段 1：GLM 判定 ----------

def glm_verdict(gold, cand):
    body = {"model": GLM_MODEL,
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post(GLM_URL, headers={"Authorization": f"Bearer {api_key()}"},
                      json=body, timeout=(10, GLM_TIMEOUT))
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def judge_missing(rows, cache_path=GLM_CACHE):
    """只补缓存里缺的 (gold, cand) 对；返回 (cache, 本次新增数)。"""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    n_before = len(cache)
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            k = key_of(row["gold"], cand)
            if k not in cache:
                jobs[k] = (row["gold"], cand)
    print(f"[GLM 判定] 缓存 {n_before}，本次待判 {len(jobs)}", flush=True)
    if not jobs:
        return cache, 0
    todo = list(jobs.items())

    def save():
        try:
            disk = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        except Exception:
            disk = {}
        disk.update(cache)
        cache.update(disk)
        cache_path.write_text(json.dumps(disk, ensure_ascii=False))

    def work(item):
        k, (gold, cand) = item
        try:
            return k, glm_verdict(gold, cand)
        except Exception:
            return k, None

    for round_no in range(1, MAX_ROUNDS + 1):
        if not todo:
            break
        errs = []
        n = 0
        ex = ThreadPoolExecutor(GLM_WORKERS)
        futs = {ex.submit(work, item): item for item in todo}
        consumed = set()
        try:
            for fut in as_completed(futs, timeout=ROUND_BUDGET):
                consumed.add(fut)
                item = futs[fut]
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
        except FutTimeout:
            # 轮次时间预算用尽：把已完成但还没被消费的 future 结果也收进来，
            # 剩下的连接视为挂死，放弃本轮、下一轮重试。
            for fut, item in futs.items():
                if fut not in consumed and fut.done():
                    consumed.add(fut)
                    verdict = None
                    try:
                        _, verdict = fut.result(timeout=0)
                    except Exception:
                        pass
                    if verdict is None:
                        errs.append(item)
                    else:
                        cache[item[0]] = verdict
                    n += 1
            pending = [it for f, it in futs.items() if f not in consumed]
            errs.extend(pending)
            print(f"  轮{round_no} 超时（{ROUND_BUDGET}s）：{len(pending)} 个连接"
                  f"挂死，放弃本轮、重试之；已完成 {n}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        save()
        todo = errs
        print(f"[GLM 判定] 轮{round_no} 结束：失败 {len(errs)}", flush=True)
        if not todo:
            break
    if todo:
        print(f"[GLM 判定] 警告：{len(todo)} 对多轮后仍未解析", flush=True)
    added = len(cache) - n_before
    print(f"[GLM 判定] 缓存新增 {added} 对（{n_before} -> {len(cache)}）",
          flush=True)
    return cache, added


# ---------- 阶段 2：分析 ----------

def mcnemar_exact(b, c):
    """精确 McNemar（双侧二项）。b/c = 两个方向的独对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def boot_ci(vals, n_boot=N_BOOT, rng=None):
    d = np.asarray([v for v in vals if v is not None], dtype=float)
    n = len(d)
    if n == 0:
        return None, None, None
    rng = RNG if rng is None else rng
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def boot_paired_diff(pairs, n_boot=N_BOOT):
    """病例级配对 bootstrap：a-b 的均值差 95% CI。pairs = [(a, b), ...]。"""
    arr = np.asarray([(a, b) for a, b in pairs if a is not None and b is not None],
                     dtype=float)
    n = len(arr)
    if n == 0:
        return None, None, None
    idx = RNG.integers(0, n, size=(n_boot, n))
    d = arr[:, 0] - arr[:, 1]
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def wilcoxon_p(diffs):
    """配对 Wilcoxon 双侧 p（零差默认剔除）；全零差值 scipy 会报错，返回 1.0。

    返回 (p, n_zero)：n_zero = 差值为 0 的病例数（报告里说明有效配对分母）。
    """
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    if d.size == 0:
        return None, 0
    n_zero = int((d == 0).sum())
    if n_zero == d.size:
        return 1.0, n_zero
    try:
        return float(wilcoxon(d, zero_method="wilcox").pvalue), n_zero
    except Exception:
        return None, n_zero


def holm(pairs):
    """Holm–Bonferroni 校正。pairs = [(label, p), ...] -> ([(label, p, p_adj)], ...).

    p 为 None 的对比原样保留为 (label, None, None)。
    """
    items = [(lb, p) for lb, p in pairs if p is not None]
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    m = len(items)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, items[i][1] * (m - rank)))
        adj[i] = running
    out = [(lb, p, a) for (lb, p), a in zip(items, adj)]
    out += [(lb, None, None) for lb, p in pairs if p is None]
    return out


class Eval:
    def __init__(self, seeds, cache):
        self.seeds = seeds
        self.cache = cache
        self.runs = {}          # (cond, scheme, seed) -> {cid: row}
        self.flags = {}         # (cond, scheme, seed) -> {cid: [True/False/None]}
        self.rowcount = {}      # (cond, scheme, seed) -> 行数
        self.missing_pairs = {}  # (cond, scheme, seed) -> 缺判对数
        self.paths = {}
        for cond in COND_DIRS:
            for scheme, fname in SCHEMES:
                for s in seeds:
                    p = seed_dir(cond, s) / f"{fname}.jsonl"
                    if not p.exists():
                        continue
                    rows = load_jsonl(p)
                    self.paths[(cond, scheme, s)] = p
                    self.runs[(cond, scheme, s)] = rows
                    self.rowcount[(cond, scheme, s)] = len(rows)
                    fl, miss = {}, 0
                    for cid, r in rows.items():
                        f = [cache.get(key_of(r["gold"], c))
                             for c in r["top5"][:5]]
                        fl[cid] = f
                        miss += sum(1 for x in f if x is None)
                    self.flags[(cond, scheme, s)] = fl
                    self.missing_pairs[(cond, scheme, s)] = miss
        self.n_cases = len(json.loads(WORKUP_DATA.read_text()))

    def complete(self, cond, scheme, seed):
        k = (cond, scheme, seed)
        return k in self.runs and self.rowcount[k] == self.n_cases

    def ready_schemes(self, seed):
        """两个条件都跑满 364 行的方案。"""
        return [m for m, _ in SCHEMES
                if self.complete("baseline", m, seed)
                and self.complete("workup", m, seed)]

    def hit(self, cond, scheme, seed, cid, k):
        flags = self.flags.get((cond, scheme, seed), {}).get(cid)
        if not flags:
            return None
        f = flags[:k]
        if not f or any(x is None for x in f):
            return None      # 判定缺失 -> 该 case 在该 k 上剔除，不静默算错
        return any(x is True for x in f)

    def hit_consensus(self, cond, scheme, cid, k, seeds):
        vals = [self.hit(cond, scheme, s, cid, k) for s in seeds]
        if any(v is None for v in vals):
            return None
        return int(sum(vals) >= (len(seeds) + 1) // 2)

    def hit_rate(self, cond, scheme, cid, k, seeds):
        vals = [self.hit(cond, scheme, s, cid, k) for s in seeds]
        if any(v is None for v in vals):
            return None
        return sum(vals) / len(vals)

    def acc(self, cond, scheme, seed, k, ids):
        vals = [self.hit(cond, scheme, seed, c, k) for c in ids]
        num = [v for v in vals if v is not None]
        return (sum(num) / len(num) if num else None), len(vals) - len(num)


def compare_pair(ev, scheme, k, ids, seed=None, seeds=None):
    """workup vs baseline 的病例级精确 McNemar（同 seed 或同 seed 集合多数决）。

    返回独对数 / p / 配对差异 bootstrap CI。方向 a = workup，b = baseline。
    """
    wu, bl, dropped = [], [], 0
    if seed is not None:
        for c in ids:
            a = ev.hit("workup", scheme, seed, c, k)
            b = ev.hit("baseline", scheme, seed, c, k)
            if a is None or b is None:
                dropped += 1
                continue
            wu.append(a)
            bl.append(b)
    else:
        for c in ids:
            a = ev.hit_consensus("workup", scheme, c, k, seeds)
            b = ev.hit_consensus("baseline", scheme, c, k, seeds)
            if a is None or b is None:
                dropped += 1
                continue
            wu.append(a)
            bl.append(b)
    a_only = sum(1 for a, b in zip(wu, bl) if a and not b)   # workup 独对
    b_only = sum(1 for a, b in zip(wu, bl) if b and not a)   # baseline 独对
    n = len(wu)
    diff, lo, hi = boot_paired_diff(list(zip(wu, bl)))
    return {
        "n_cases": n, "n_excluded_unjudged": dropped,
        "acc_workup": (sum(wu) / n if n else None),
        "acc_baseline": (sum(bl) / n if n else None),
        "workup_only": a_only, "baseline_only": b_only,
        "delta_hits": a_only - b_only,
        "diff": diff, "ci95": [lo, hi],
        "mcnemar_p": mcnemar_exact(a_only, b_only),
    }


def case_rate_comparison(ev, scheme, k, ids, seeds):
    """病例级命中率口径的 workup vs baseline（口径同 stats_caselevel.py）。

    每病例先在参与的 seeds 上取 0–1 命中率（要求该 seed 在该方案下可判定，
    否则该病例剔除并计入 n_excluded_unjudged，不当作判错），然后跨病例做
    配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 95% CI；多数决精确
    McNemar 作为补充（与既有 §3「多数决」行同一口径）。
    """
    wu, bl = {}, {}
    dropped = 0
    for c in ids:
        a = ev.hit_rate("workup", scheme, c, k, seeds)
        b = ev.hit_rate("baseline", scheme, c, k, seeds)
        if a is None or b is None:
            dropped += 1
            continue
        wu[c], bl[c] = a, b
    common = sorted(wu)
    ra = np.array([wu[c] for c in common], dtype=float)
    rb = np.array([bl[c] for c in common], dtype=float)
    diff = ra - rb
    n = len(common)
    mean, lo, hi = boot_ci(diff, rng=RNG_NEW) if n else (None, None, None)
    p, n_zero = wilcoxon_p(diff)
    a_only = b_only = 0
    for c in common:
        ha = ev.hit_consensus("workup", scheme, c, k, seeds)
        hb = ev.hit_consensus("baseline", scheme, c, k, seeds)
        if ha is None or hb is None:
            continue
        if ha and not hb:
            a_only += 1
        elif hb and not ha:
            b_only += 1
    return {
        "n_cases": n, "n_excluded_unjudged": dropped,
        "n_seeds": len(seeds), "seeds": list(seeds),
        "mean_rate_workup": (float(ra.mean()) if n else None),
        "mean_rate_baseline": (float(rb.mean()) if n else None),
        "mean_diff": mean, "median_diff": (float(np.median(diff)) if n else None),
        "ci95": [lo, hi], "wilcoxon_p": p, "n_zero_diff": n_zero,
        "majority_mcnemar": {"workup_only": a_only, "baseline_only": b_only,
                             "p": mcnemar_exact(a_only, b_only)},
    }


def scheme_effect(ev, scheme, k, ids, seeds):
    """病例级 workup 效应 d = rate_workup − rate_baseline（缺判定则剔除该病例）。"""
    out = {}
    for c in ids:
        a = ev.hit_rate("workup", scheme, c, k, seeds)
        b = ev.hit_rate("baseline", scheme, c, k, seeds)
        if a is None or b is None:
            continue
        out[c] = a - b
    return out


def friedman_test(arrs, labels):
    """Friedman 检验（病例为区组）：三策略的病例级 workup 效应是否齐一。"""
    n = len(arrs[0]) if arrs else 0
    res = {"n_blocks": n, "k": len(arrs), "strategies": list(labels),
           "chi2": None, "p": None, "note": ""}
    if n < 2:
        res["note"] = f"可用病例数不足（{n} < 2），无法做 Friedman"
        return res
    if all(np.array_equal(arrs[0], a) for a in arrs[1:]):
        res["note"] = "三策略的病例级效应逐例完全相同，Friedman 无法计算"
        return res
    try:
        chi2, p = friedmanchisquare(*[np.asarray(a, dtype=float) for a in arrs])
    except Exception as exc:
        res["note"] = f"Friedman 无法计算：{exc}"
        return res
    res["chi2"], res["p"] = float(chi2), float(p)
    if n < 10:
        res["note"] = f"区组数 {n} < 10，Friedman 卡方近似可能不准，以后事检验为准"
    else:
        res["note"] = "（病例为区组；χ² 近似，区组数 ≥10）"
    return res


def interaction_analysis(ev, ids, seeds):
    """交互检验：workup 效应是否因策略而异（本次核心）。

    对每病例算 d_strategy = rate_workup − rate_baseline（跨参与 seeds 的命中率
    之差），然后跨病例配对检验
      d_MDT − d_A×1（= Δdiff，正值 = workup 加入后 MDT 落后 A×1 的差距缩小）、
      d_P   − d_A×1、d_MDT − d_P。
    整体：Friedman（病例为区组）检验三策略的 d 是否齐一；事后三对两两配对
    Wilcoxon + Holm 校正。seeds 为空或缺方案时给出 can_do=False 与原因。
    """
    out = {"n_seeds": len(seeds), "seeds": list(seeds),
           "definition": "d_strategy = 每病例 rate_workup − rate_baseline（跨参与"
                         " seeds 的 0–1 命中率之差）；d_X − d_Y 为跨病例配对量",
           "can_do": True, "topk": {}}
    if not seeds:
        out["can_do"] = False
        out["reason"] = ("A×1 / P / MDT 三个方案在两个条件下同时跑满的 seed 一个"
                         "都没有，无法做交互检验")
        return out
    for k in KS:
        eff = {m: scheme_effect(ev, m, k, ids, seeds) for m, _ in SCHEMES}
        common = sorted(set(eff["Ax1"]) & set(eff["P"]) & set(eff["MDT"]))
        d = {m: np.array([eff[m][c] for c in common], dtype=float)
             for m, _ in SCHEMES}
        per_seed = {}
        for s in seeds:
            row = {}
            for m, _ in SCHEMES:
                a, _na = ev.acc("workup", m, s, k, ids)
                b, _nb = ev.acc("baseline", m, s, k, ids)
                row[m] = None if (a is None or b is None) else a - b
            per_seed[s] = row
        cell = {"n_cases": len(common),
                "n_excluded_unjudged": len(ids) - len(common),
                "n_seeds": len(seeds), "seeds": list(seeds),
                "effect_mean": {m: (float(d[m].mean()) if common else None)
                                for m, _ in SCHEMES},
                "per_seed_effect": {s: per_seed[s] for s in seeds},
                "pairs": {}}
        for a, b in (("MDT", "Ax1"), ("P", "Ax1"), ("MDT", "P")):
            diff = d[a] - d[b]
            mean, lo, hi = (boot_ci(diff, rng=RNG_NEW) if len(common)
                            else (None, None, None))
            p, n_zero = wilcoxon_p(diff)
            cell["pairs"][f"{a}_vs_{b}"] = {
                "label": f"d_{SCHEME_LABEL[a]} − d_{SCHEME_LABEL[b]}",
                "n_cases": len(common), "mean": mean,
                "median": (float(np.median(diff)) if len(common) else None),
                "ci95": [lo, hi], "wilcoxon_p": p, "n_zero_diff": n_zero,
            }
        cell["friedman"] = friedman_test([d[m] for m, _ in SCHEMES],
                                        [SCHEME_LABEL[m] for m, _ in SCHEMES])
        cell["posthoc_holm"] = [
            {"pair": lb, "wilcoxon_p": p, "p_holm": a}
            for lb, p, a in holm([(f"{a}_vs_{b}",
                                   cell["pairs"][f"{a}_vs_{b}"]["wilcoxon_p"])
                                  for a, b in (("MDT", "Ax1"), ("P", "Ax1"),
                                               ("MDT", "P"))])]
        cell["single_seed_caveat"] = (
            "仅 1 个 seed：病例级 d 取值只能是 −1/0/1，检验只反映该 seed 内病例"
            "层面的交互，不含 seed 抽样随机性，结论按点估计/探索性看待。"
            if len(seeds) == 1 else "")
        out["topk"][k] = cell
    return out


def analyze_stratum(ev, ids, seeds):
    res = {"n": len(ids)}
    ready_by_seed = {s: [m for m, _ in SCHEMES if m in ev.ready_schemes(s)]
                     for s in seeds}
    # 逐 seed 逐方案准确率
    acc = {m: {c: {} for c in COND_DIRS} for m, _ in SCHEMES}
    unjudged = {m: {c: {} for c in COND_DIRS} for m, _ in SCHEMES}
    for m, _ in SCHEMES:
        for cond in COND_DIRS:
            for s in seeds:
                if not ev.complete(cond, m, s):
                    continue
                acc[m][cond][s] = {}
                unjudged[m][cond][s] = {}
                for k in KS:
                    a, nj = ev.acc(cond, m, s, k, ids)
                    acc[m][cond][s][k] = a
                    unjudged[m][cond][s][k] = nj
    res["accuracy"] = acc
    res["unjudged"] = unjudged

    # 多 seed：均值±SD（只统计两条件都完整、且该 case 有判定的部分）
    res["mean_sd"] = {}
    for m, _ in SCHEMES:
        res["mean_sd"][m] = {}
        for cond in COND_DIRS:
            per = {k: [acc[m][cond][s][k] for s in seeds
                       if s in acc[m][cond] and acc[m][cond][s][k] is not None]
                   for k in KS}
            res["mean_sd"][m][cond] = {
                k: {"mean": (float(np.mean(per[k])) if per[k] else None),
                    "sd": (float(np.std(per[k], ddof=1)) if len(per[k]) > 1
                           else (0.0 if per[k] else None)),
                    "n_seeds": len(per[k]), "seeds": [
                        s for s in seeds
                        if s in acc[m][cond] and acc[m][cond][s][k] is not None]}
                for k in KS}

    # 核心对比：同 seed 配对精确 McNemar
    comp = {}
    for m, _ in SCHEMES:
        entry = {"per_seed": {}}
        for s in seeds:
            if m not in ready_by_seed[s]:
                continue
            entry["per_seed"][s] = {k: compare_pair(ev, m, k, ids, seed=s)
                                    for k in KS}
        if len(seeds) > 1:
            usable = [s for s in seeds if m in ready_by_seed[s]]
            if usable:
                entry["consensus"] = {
                    k: compare_pair(ev, m, k, ids, seeds=usable) for k in KS}
                # 敏感分析：seed×case 视作独立（会低估方差，仅作参考）
                entry["pooled_seedcase"] = {}
                for k in KS:
                    wu, bl = [], []
                    for s in usable:
                        for c in ids:
                            a = ev.hit("workup", m, s, c, k)
                            b = ev.hit("baseline", m, s, c, k)
                            if a is None or b is None:
                                continue
                            wu.append(a)
                            bl.append(b)
                    ao = sum(1 for a, b in zip(wu, bl) if a and not b)
                    bo = sum(1 for a, b in zip(wu, bl) if b and not a)
                    entry["pooled_seedcase"][k] = {
                        "n_obs": len(wu), "workup_only": ao, "baseline_only": bo,
                        "mcnemar_p": mcnemar_exact(ao, bo)}
        comp[m] = entry
    res["comparisons"] = comp

    # 病例级命中率口径：每病例跨 seed 命中率（口径同 stats_caselevel.py）
    res["case_level"] = {}
    for m, _ in SCHEMES:
        usable = [s for s in seeds if m in ready_by_seed[s]]
        res["case_level"][m] = {
            "usable_seeds": usable, "n_seeds": len(usable),
            "per_k": ({k: case_rate_comparison(ev, m, k, ids, usable)
                       for k in KS} if usable else {})}

    # 交互检验：只有三方案在两条件下都跑满的 seed 才参与（保证同一病例集合
    # 上三个策略的 d 可比、Friedman 区组对齐）
    all_schemes = {m for m, _ in SCHEMES}
    inter_seeds = [s for s in seeds if all_schemes <= set(ev.ready_schemes(s))]
    res["interaction"] = interaction_analysis(ev, ids, inter_seeds)
    res["interaction"]["seeds_skipped"] = [s for s in seeds
                                          if s not in inter_seeds]

    # 关键问题：MDT 相对 A×1 的差距（Δ = MDT − A×1）及其变化
    res["gap_mdt_ax1"] = gap_analysis(ev, ids, seeds)
    return res


def gap_analysis(ev, ids, seeds):
    """Δ_baseline = MDT−A×1（baseline），Δ_workup 同理；看 Δ 是否缩小。"""
    out = {"per_seed": {}, "note": "Δ = MDT top-k 命中率 − A×1 top-k 命中率"}
    common = [s for s in seeds
              if {"MDT", "Ax1"} <= set(ev.ready_schemes(s))]
    for s in common:
        out["per_seed"][s] = {}
        for k in KS:
            cell = {}
            for cond in COND_DIRS:
                a, _ = ev.acc(cond, "MDT", s, k, ids)
                b, _ = ev.acc(cond, "Ax1", s, k, ids)
                cell[cond] = None if (a is None or b is None) else a - b
            out["per_seed"][s][k] = cell
            d_b, d_w = cell["baseline"], cell["workup"]
            cell["diff_of_diff"] = (None if d_b is None or d_w is None
                                    else d_w - d_b)
    if len(common) > 1:
        out["multi_seed"] = {}
        out["multi_seed_seeds"] = common
        # 统一在同一病例集合上算 Δ 与 Δdiff：每病例先跨 seed 取平均命中率，
        # 再在病例集合上算 Δ = MDT − A×1（病例级，避免不同 k 分母不同导致
        # 「Δ 的均值」与「Δdiff 的均值」对不上）。
        for k in KS:
            gaps = {"baseline": [], "workup": [], "diff_of_diff": []}
            cell = {}
            for c in ids:
                g = {}
                ok = True
                for cond in COND_DIRS:
                    mdt = ev.hit_rate(cond, "MDT", c, k, common)
                    ax1 = ev.hit_rate(cond, "Ax1", c, k, common)
                    if mdt is None or ax1 is None:
                        ok = False
                        break
                    g[cond] = mdt - ax1
                if not ok:
                    continue
                for cond in COND_DIRS:
                    gaps[cond].append(g[cond])
                gaps["diff_of_diff"].append(g["workup"] - g["baseline"])
            n_case = len(gaps["diff_of_diff"])
            for cond in COND_DIRS:
                v = gaps[cond]
                cell[cond] = ({"mean": float(np.mean(v)),
                               "sd": (float(np.std(v, ddof=1))
                                      if len(v) > 1 else 0.0),
                               "n_cases": len(v)} if v else None)
            did = gaps["diff_of_diff"]
            cell["diff_of_diff"] = {
                "n_cases": n_case,
                "mean": float(np.mean(did)) if did else None,
                "ci95": (list(boot_ci(did)[1:]) if did else None),
                "p_wilcoxon": (float(wilcoxon(did, zero_method="wilcox").pvalue)
                               if did and not np.all(np.array(did) == 0)
                               else (1.0 if did else None))}
            out["multi_seed"][k] = cell
    return out


def build_strata(ev, seeds):
    sub = {c["case_id"]: c for c in json.loads(WORKUP_DATA.read_text())}
    # 只保留在所有「已跑满 364 行」的文件里都出现的病例（跑满的文件本就是全集，
    # 该过滤是防止某个文件残缺时静默少算）
    ids = [c for c in sub
           if all(c in ev.runs[k] for k in ev.runs
                  if ev.rowcount[k] == ev.n_cases)]
    sym = [i for i in ids if SYMPTOM.search(sub[i]["gold"])]
    dis = [i for i in ids if not SYMPTOM.search(sub[i]["gold"])]
    return {
        "all": ("全部", ids),
        "objective_has_gold": ("客观段含金标签（objective_contains_gold=True）",
                               [i for i in ids if sub[i]["objective_contains_gold"]]),
        "objective_no_gold": ("客观段不含金标签（False）",
                              [i for i in ids
                               if not sub[i]["objective_contains_gold"]]),
        "objective_len_ge200": ("客观段 ≥200 字符",
                                [i for i in ids if len(sub[i]["objective"]) >= 200]),
        "objective_len_lt200": ("客观段 <200 字符（含 0 字符的无客观段病例）",
                                [i for i in ids if len(sub[i]["objective"]) < 200]),
        "symptom": ("症状级金标签（沿用 erreason_5seeds 定义）", sym),
        "disease": ("疾病级金标签", dis),
        # 敏感性分析：剔除客观数据段为空（两条件输入按构造完全相同）的零信息
        # 病例。必须追加在最后：analyze 按插入顺序消费全局 RNG_NEW，插在中间会
        # 改变后续分层的 bootstrap 抽样序列，动到既有数字。
        "objective_nonzero": ("客观数据段非空（剔除 objective 为空的 34 例，"
                              "敏感性分析）",
                              [i for i in ids if len(sub[i]["objective"]) > 0]),
    }


# ---------- 报告 ----------

def availability_rows(ev, seeds, n_cases):
    rows = []
    for cond in COND_DIRS:
        for m, fname in SCHEMES:
            for s in seeds:
                k = (cond, m, s)
                if k not in ev.paths:
                    rows.append((cond, m, s, None, None, "文件缺失"))
                    continue
                n = ev.rowcount[k]
                miss = ev.missing_pairs[k]
                if n != n_cases:
                    status = f"未跑完（{n}/{n_cases}）"
                elif miss:
                    status = f"判定缺 {miss} 对"
                else:
                    status = "完整"
                rows.append((cond, m, s, n, miss, status))
    return rows


def ready_notes(ev, seeds):
    """哪些 (方案, seed) 还不能做完整对比，及原因。"""
    bad = []
    for m, _ in SCHEMES:
        for s in seeds:
            why = []
            for cond in COND_DIRS:
                k = (cond, m, s)
                if k not in ev.paths:
                    why.append(f"{COND_LABEL[cond]} 文件缺失")
                elif ev.rowcount[k] != ev.n_cases:
                    why.append(f"{COND_LABEL[cond]} 行数 {ev.rowcount[k]}/"
                               f"{ev.n_cases}")
            if why:
                bad.append((m, s, why))
    return bad


def write_case_level(A, results, seeds):
    """第 7 节：多 seed 病例级命中率口径（与 stats_caselevel.py 对齐）。"""
    A("## 7. 多 seed 病例级聚合（口径同 stats_caselevel.py / erreason_5seeds.py）\n")
    A("口径：每病例 × (条件, 方案, top-k) 先在参与 seeds 上取 0–1 命中率，再在病例"
      "集合内做跨病例配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次 "
      "95% CI；不把 seed 当独立样本合并（seed×病例合并的 McNemar 只在第 3 节作为"
      "敏感性分析）。参与 seed 数 = 该方案在两条件都跑满 364 行的 seed 数；每格"
      "标出参与 seed 数与分母 n 例；任一参与 seed 上判定缺失的病例从该格剔除并"
      "计入剔除列，不当作判错。\n")
    A("与 `stats_caselevel.py` / `erreason_5seeds.py` 的唯一差别：这两个脚本对"
      "「前 k 个候选里部分未判定」按未命中处理（把未判定当判错），本脚本一律把该"
      "病例从该格剔除；判定缓存无缺失对时两者逐位相同（当前数据即如此）。\n")
    for skey, res in results["strata"].items():
        A(f"### {res['name']}（n={res['n']}）\n")
        A("每病例跨参与 seeds 命中率的均值（%），括号内为该格分母：\n")
        hdr = ["方案", "条件", "参与 seed"] + [f"top-{k}" for k in KS]
        A("| " + " | ".join(hdr) + " |")
        A("|---" * len(hdr) + "|")
        for m, _ in SCHEMES:
            cl = res["case_level"][m]
            for cond in COND_DIRS:
                cells = []
                for k in KS:
                    c = cl["per_k"].get(k)
                    if c is None:
                        cells.append("NA")
                        continue
                    v = (c["mean_rate_workup"] if cond == "workup"
                         else c["mean_rate_baseline"])
                    cells.append("NA" if v is None
                                 else f"{pc(v)}% (n={c['n_cases']})")
                A(f"| {SCHEME_LABEL[m]} | {COND_LABEL[cond]} | "
                  f"{cl['n_seeds']} | " + " | ".join(cells) + " |")
        A("")
        for m, _ in SCHEMES:
            cl = res["case_level"][m]
            if not cl["per_k"]:
                A(f"**{SCHEME_LABEL[m]}**：待补（该方案在两个条件下同时跑满的 "
                  "seed 还没有）\n")
                continue
            A(f"**{SCHEME_LABEL[m]}：workup vs baseline**（参与 seeds "
              f"{cl['usable_seeds']}，{cl['n_seeds']} 个）\n")
            A("| top-k | n 例（分母） | 剔除未判定 | workup 命中率 | baseline 命中率 "
              "| 差值 [95% CI] | 配对 Wilcoxon p | 零差病例 | 多数决 McNemar (w:b) p |")
            A("|---|---|---|---|---|---|---|---|---|")
            for k in KS:
                c = cl["per_k"][k]
                lo, hi = c["ci95"]
                mj = c["majority_mcnemar"]
                A(f"| top-{k} | {c['n_cases']} | {c['n_excluded_unjudged']} | "
                  f"{pc(c['mean_rate_workup'])}% | "
                  f"{pc(c['mean_rate_baseline'])}% | "
                  f"{pp(c['mean_diff'])}pp [{pp(lo)}, {pp(hi)}] | "
                  f"{fmt_p(c['wilcoxon_p'])}{sig(c['wilcoxon_p'])} | "
                  f"{c['n_zero_diff']} | "
                  f"{mj['workup_only']}:{mj['baseline_only']} "
                  f"p={fmt_p(mj['p'])}{sig(mj['p'])} |")
            A("")
        if len(seeds) == 1:
            A("注意：本次参与比较的 seed 只有 1 个，每病例的「跨 seed 命中率」退化为 "
              "0/1 指示值，配对 Wilcoxon 与第 3 节的 McNemar 检验的是同一个病例级"
              "差异（统计量口径不同）；多 seed 结果在 seeds 2–5 跑完后重跑本脚本"
              "即自动更新。\n")


def write_report(ev, results, seeds, added, cache_size, missing_after):
    L = []
    A = L.append
    n_cases = ev.n_cases
    A(f"# workup-informed vs presentation-only（ER-Reason {n_cases} 例）\n")
    A(f"- 条件 A（baseline，presentation-only）：`routing_study/results/topn_erreason/`")
    A(f"- 条件 B（workup，临床表现 + 本次就诊客观检查）："
      f"`routing_study/results/topn_erreason_workup/`")
    A(f"- 输入：`data/er_reason_workup_subset.json`（{n_cases} 例，两条件 case_id"
      f" 与 gold 完全一致，只有输入不同）")
    A(f"- 判官：GLM-5.3-flash × v3 提示词（`judge_v3.V3_PROMPT` 原样复用），缓存"
      f" `judge_cache_glm_v3.json`（{cache_size} 对，本次新增 {added}）")
    A(f"- 参与比较的 seed：{seeds}；seed 1 = 顶层目录，seeds 2–5 = `s{{2..5}}/`")
    req = results.get("seeds_requested", seeds)
    if sorted(req) != sorted(seeds):
        A(f"- 请求的 seed {req} 中 {sorted(set(req) - set(seeds))} 在两个条件下"
          f"都还没有目录，未参与比较")
    A(f"- 判定缺失对（只统计已跑满 {n_cases} 行的文件）：{missing_after}\n")
    A("命中率为判定口径：某一 (条件, 方案, seed, top-k) 里若有病例的前 k 个候选"
      "存在未判定项，该病例从该格剔除（不当作判错），格子里以 `命中率*(实际分母)` "
      "标出；未跑满 364 行的文件不参与命中率与检验。\n")
    A("- 统计口径分层：第 3 节为逐 seed 病例级精确 McNemar（同一 seed 内两条件"
      "配对）；第 4 节末为策略 × 条件的交互检验（每病例跨 seed 的 workup 效应之差"
      " → 配对 Wilcoxon + Friedman）；第 7 节为多 seed 病例级命中率口径"
      "（与 `stats_caselevel.py` 一致）。多 seed 结果一律不把 seed 当独立样本。\n")

    # 1. 数据状态
    A("## 1. 数据与判定状态\n")
    A("| 条件 | 方案 | seed | 行数 | 判定缺对 | 状态 |")
    A("|---|---|---|---|---|---|")
    for cond, m, s, n, miss, status in availability_rows(ev, seeds, n_cases):
        A(f"| {COND_LABEL[cond]} | {SCHEME_LABEL[m]} | {s} | "
          f"{'—' if n is None else n} | {'—' if miss is None else miss} | "
          f"{status} |")
    bad = ready_notes(ev, seeds)
    if bad:
        A("\n**尚不能对比的组合（待补）**：\n")
        for m, s, why in bad:
            A(f"- {SCHEME_LABEL[m]} seed {s}：{'；'.join(why)}")
    A("")

    # 2. 逐条件逐方案 top-1/3/5
    A("## 2. 逐条件逐方案 top-1/3/5\n")
    for skey, res in results["strata"].items():
        if skey != "all":
            continue
        A(f"### {res['name']}（n={res['n']}）\n")
        def cell_acc(m, cond, s, k):
            """准确率（%）；该格有未判定病例时标出实际分母。"""
            a = res["accuracy"][m][cond].get(s, {}).get(k)
            if a is None:
                return "NA"
            nj = res["unjudged"][m][cond].get(s, {}).get(k, 0)
            return (f"{pc(a)}%*(n={res['n'] - nj})" if nj else f"{pc(a)}%")

        if len(seeds) == 1:
            s = seeds[0]
            A("| 方案 | 条件 | top-1 | top-3 | top-5 |")
            A("|---|---|---|---|---|")
            for m, _ in SCHEMES:
                for cond in COND_DIRS:
                    cells = [cell_acc(m, cond, s, k) for k in KS]
                    A(f"| {SCHEME_LABEL[m]} | {COND_LABEL[cond]} | "
                      f"{' | '.join(cells)} |")
        else:
            A("| 方案 | 条件 | 指标 | " + " | ".join(f"s{s}" for s in seeds)
              + " | 均值±SD（参与 seed 数） |")
            A("|---|---|---|" + "---|" * len(seeds) + "---|")
            for m, _ in SCHEMES:
                for cond in COND_DIRS:
                    for k in KS:
                        ms = res["mean_sd"][m][cond][k]
                        cells = [cell_acc(m, cond, s, k) for s in seeds]
                        if ms["mean"] is None:
                            agg = "NA（0 个 seed 可用）"
                        elif ms["n_seeds"] <= 1:
                            agg = (f"{pc(ms['mean'])}%（仅 {ms['n_seeds']} 个 "
                                   f"seed 可用，无 SD）")
                        else:
                            agg = (f"{pc(ms['mean'])} ± {num(ms['sd'])}%"
                                   f"（{ms['n_seeds']} 个 seed）")
                        A(f"| {SCHEME_LABEL[m]} | {COND_LABEL[cond]} | top-{k} | "
                          + " | ".join(cells) + " | " + agg + " |")
        A("")

    # 3. 核心对比
    A("## 3. 核心对比：workup vs baseline（病例级精确 McNemar，双侧）\n")
    A("独对数 = 该策略下只有一侧命中的病例数；方向 = workup − baseline；"
      "CI 为病例级配对 bootstrap 10,000 次 95%。\n")
    res_all = results["strata"]["all"]
    for m, _ in SCHEMES:
        entry = res_all["comparisons"][m]
        A(f"### {SCHEME_LABEL[m]}\n")
        if not entry["per_seed"] and "consensus" not in entry:
            A("（两条件尚未同时跑完/判定，本方案待补）\n")
            continue
        A("| seed | top-k | workup | baseline | 独对 (workup:baseline) | "
          "差值 [95% CI] | McNemar p |")
        A("|---|---|---|---|---|---|---|")
        for s, per_k in entry["per_seed"].items():
            for k in KS:
                c = per_k[k]
                if c["n_cases"] == 0:
                    A(f"| s{s} | top-{k} | — | — | — | — | "
                      f"（无两侧都可判定的病例） |")
                    continue
                lo, hi = c["ci95"]
                A(f"| s{s} | top-{k} | {pc(c['acc_workup'])}% | "
                  f"{pc(c['acc_baseline'])}% | "
                  f"{c['workup_only']}:{c['baseline_only']} | "
                  f"{pp(c['diff'])}pp [{pp(lo)}, {pp(hi)}] | "
                  f"{fmt_p(c['mcnemar_p'])}{sig(c['mcnemar_p'])} |")
        if "consensus" in entry:
            for k in KS:
                c = entry["consensus"][k]
                lo, hi = c["ci95"]
                A(f"| 多数决 | top-{k} | {pc(c['acc_workup'])}% | "
                  f"{pc(c['acc_baseline'])}% | "
                  f"{c['workup_only']}:{c['baseline_only']} | "
                  f"{pp(c['diff'])}pp [{pp(lo)}, {pp(hi)}] | "
                  f"{fmt_p(c['mcnemar_p'])}{sig(c['mcnemar_p'])} |")
            # 剔除数（未判定）逐个写明，不静默丢弃
            excl = []
            for s, per_k in entry["per_seed"].items():
                for k in KS:
                    if per_k[k]["n_excluded_unjudged"]:
                        excl.append(f"s{s} top-{k} {per_k[k]['n_excluded_unjudged']} 例")
            for k in KS:
                if entry["consensus"][k]["n_excluded_unjudged"]:
                    excl.append(f"多数决 top-{k} "
                                f"{entry['consensus'][k]['n_excluded_unjudged']} 例")
            if excl:
                A(f"\n（因未判定而从配对中剔除：{'；'.join(excl)}）")
            pooled = entry.get("pooled_seedcase", {})
            if pooled:
                A("\n（敏感分析：把 seed×病例 视作独立观测的合并 McNemar，"
                  "方差被低估，仅作参考；多 seed 的主口径是第 7 节的病例级命中率"
                  "配对检验与本节末的交互检验）\n")
                A("| top-k | 观测数 | 独对 (workup:baseline) | p |")
                A("|---|---|---|---|")
                for k in KS:
                    c = pooled[k]
                    A(f"| top-{k} | {c['n_obs']} | "
                      f"{c['workup_only']}:{c['baseline_only']} | "
                      f"{fmt_p(c['mcnemar_p'])}{sig(c['mcnemar_p'])} |")
        A("")

    # 4. 关键问题
    A("## 4. 关键问题：workup 数据加入后，MDT 相对 A×1 的差距是否缩小\n")
    gap = res_all["gap_mdt_ax1"]
    ready_mdt = [s for s in seeds if s in gap["per_seed"]]
    if not ready_mdt:
        A(f"**待补**：MDT 与 A×1 在两个条件下都跑满 {n_cases} 行后才能计算。"
          "当前缺失明细见第 1 节。\n")
    else:
        A("Δ 定义 = MDT 命中率 − A×1 命中率（负数 = MDT 落后于 A×1）；"
          "Δdiff = Δ_workup − Δ_baseline（>0 = workup 数据加入后差距缩小）。\n")
        A("逐 seed（Δ 为两方案该 seed 命中率之差）：\n")
        A("| seed | top-k | Δ_baseline | Δ_workup | Δdiff |")
        A("|---|---|---|---|---|")
        for s in ready_mdt:
            for k in KS:
                cell = gap["per_seed"][s][k]
                A(f"| s{s} | top-{k} | {pp(cell['baseline'])}pp | "
                  f"{pp(cell['workup'])}pp | {pp(cell['diff_of_diff'])}pp |")
        if "multi_seed" in gap:
            A(f"\n跨 seed（病例级：每病例先跨 seeds {gap['multi_seed_seeds']} "
              "取平均命中率，再在病例集合内计算 Δ；三者用同一病例集合）：\n")
            A("| top-k | n 例 | Δ_baseline（均值±SD） | Δ_workup（均值±SD） | "
              "Δdiff 均值 [95% CI] | Wilcoxon p |")
            A("|---|---|---|---|---|---|")
            for k in KS:
                cell = gap["multi_seed"][k]
                db, dw = cell["baseline"], cell["workup"]
                did = cell["diff_of_diff"]
                if db is None or dw is None or did["mean"] is None:
                    continue
                lo, hi = (did["ci95"] if did["ci95"] else (None, None))
                A(f"| top-{k} | {did['n_cases']} | "
                  f"{pp(db['mean'])} ± {num(db['sd'])}pp | "
                  f"{pp(dw['mean'])} ± {num(dw['sd'])}pp | "
                  f"{pp(did['mean'])}pp [{pp(lo)}, {pp(hi)}] | "
                  f"{fmt_p(did['p_wilcoxon'])}{sig(did['p_wilcoxon'])} |")
        # 结论
        A("")
        concl = []
        for k in KS:
            if "multi_seed" in gap:
                did = gap["multi_seed"][k]["diff_of_diff"]
                mean, p = did["mean"], did["p_wilcoxon"]
            else:
                s = ready_mdt[0]
                mean = gap["per_seed"][s][k]["diff_of_diff"]
                p = None
            if mean is None:
                concl.append(f"- top-{k}：无法计算")
            elif p is None:
                concl.append(f"- top-{k}：Δdiff = {pp(mean)}pp（单 seed 点估计，"
                             f"病例级配对检验见本节末尾「交互检验」）")
            elif p < 0.05:
                concl.append(f"- top-{k}：Δdiff = {pp(mean)}pp，p={fmt_p(p)}"
                             f"{sig(p)} → 差距**{'缩小' if mean > 0 else '扩大'}**（显著）")
            else:
                concl.append(f"- top-{k}：Δdiff = {pp(mean)}pp，p={fmt_p(p)} → "
                             + ("差距变化不大（不显著）" if abs(mean) < 0.02
                                else "差距有变化但未达显著"))
        A("\n".join(concl) + "\n")

        # --- 交互检验：workup 效应是否因策略而异（病例级配对） ---
        A("### 交互检验：workup 效应是否因策略而异\n")
        inter = res_all["interaction"]
        if not inter.get("can_do"):
            A(f"**无法计算**：{inter.get('reason', '')}。当前可用情况见第 1 节。\n")
        else:
            A("d_strategy = 每病例跨参与 seeds 的 (rate_workup − rate_baseline)；"
              "策略间的 d 之差即交互量，跨病例做配对 Wilcoxon 双侧，CI 为病例级"
              " cluster bootstrap 10,000 次 95%。参与 seeds = "
              f"{inter['seeds']}（{inter['n_seeds']} 个）"
              + (f"；因三方案未同时跑满而排除的 seed：{inter['seeds_skipped']}"
                 if inter.get("seeds_skipped") else "")
              + "。正值 = 该策略在 workup 数据上获益更多。\n")
            A("| top-k | 参与 seed | n 例（分母） | d(A×1) | d(P) | d(MDT) | "
              "d_MDT−d_A×1 [95% CI] | p | d_P−d_A×1 [95% CI] | p | "
              "d_MDT−d_P [95% CI] | p |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|")

            def cell_pair(k, key):
                pr = inter["topk"][k]["pairs"][key]
                lo2, hi2 = pr["ci95"]
                return (f"{pp(pr['mean'])}pp [{pp(lo2)}, {pp(hi2)}] | "
                        f"{fmt_p(pr['wilcoxon_p'])}{sig(pr['wilcoxon_p'])}")

            for k in KS:
                c = inter["topk"][k]
                em = c["effect_mean"]
                denom = f"{c['n_cases']}"
                if c["n_excluded_unjudged"]:
                    denom += f"（剔除 {c['n_excluded_unjudged']} 例未判定）"
                A(f"| top-{k} | {c['n_seeds']} | {denom} | {pp(em['Ax1'])}pp | "
                  f"{pp(em['P'])}pp | {pp(em['MDT'])}pp | "
                  + cell_pair(k, "MDT_vs_Ax1") + " | "
                  + cell_pair(k, "P_vs_Ax1") + " | "
                  + cell_pair(k, "MDT_vs_P") + " |")
            A("")
            if inter["n_seeds"] > 1:
                A("逐 seed 的策略级 workup 效应 d（单 seed 时 = 该 seed 的命中率差，"
                  "即 Δ 列）：\n")
                A("| seed | top-k | d(A×1) | d(P) | d(MDT) | d_MDT−d_A×1 |")
                A("|---|---|---|---|---|---|")
                for s in inter["seeds"]:
                    for k in KS:
                        row = inter["topk"][k]["per_seed_effect"][s]
                        did = (None if row["MDT"] is None or row["Ax1"] is None
                               else row["MDT"] - row["Ax1"])
                        A(f"| s{s} | top-{k} | {pp(row['Ax1'])}pp | "
                          f"{pp(row['P'])}pp | {pp(row['MDT'])}pp | "
                          f"{pp(did)}pp |")
                A("")
            if inter["n_seeds"] == 1:
                A("（" + inter["topk"][KS[0]]["single_seed_caveat"] + "）\n")
            A("整体交互（Friedman 检验，病例为区组，检验三个策略的 workup 效应"
              "是否齐一）：\n")
            A("| top-k | n 区组 | Friedman χ² | p | 说明 |")
            A("|---|---|---|---|---|")
            for k in KS:
                f = inter["topk"][k]["friedman"]
                chi2 = "NA" if f["chi2"] is None else f"{f['chi2']:.3f}"
                A(f"| top-{k} | {f['n_blocks']} | {chi2} | "
                  f"{fmt_p(f['p'])}{sig(f['p'])} | {f['note']} |")
            A("")
            A("Friedman 显著时需事后两两比较；下表为三对配对 Wilcoxon 双侧 + Holm "
              "逐步法校正（在每个 top-k 内族大小 k=3，控制族错误率）：\n")
            A("| top-k | 对比 | 原始 p | Holm p |")
            A("|---|---|---|---|")
            for k in KS:
                for ph in inter["topk"][k]["posthoc_holm"]:
                    A(f"| top-{k} | {ph['pair']} | "
                      f"{fmt_p(ph['wilcoxon_p'])}{sig(ph['wilcoxon_p'])} | "
                      f"{fmt_p(ph['p_holm'])}{sig(ph['p_holm'])} |")
            A("")
            A("**结论（加入客观数据是否缩小 MDT 相对 A×1 的差距）**：\n")
            for k in KS:
                pr = inter["topk"][k]["pairs"]["MDT_vs_Ax1"]
                mean, p = pr["mean"], pr["wilcoxon_p"]
                if mean is None:
                    A(f"- top-{k}：无法计算")
                elif p is None:
                    A(f"- top-{k}：d_MDT−d_A×1 = {pp(mean)}pp（检验未得出 p 值）")
                else:
                    lo2, hi2 = pr["ci95"]
                    line = (f"- top-{k}：d_MDT−d_A×1 = {pp(mean)}pp "
                            f"[{pp(lo2)}, {pp(hi2)}]，p={fmt_p(p)}{sig(p)} → ")
                    if p < 0.05:
                        line += (f"差距**{'缩小' if mean > 0 else '扩大'}**"
                                 "（显著，不能归因于偶然）")
                    else:
                        line += ("差距变化不显著：不能认为加入客观数据改变了"
                                 "MDT 相对 A×1 的差距")
                    A(line)
            A("")
            f1 = inter["topk"][KS[0]]["friedman"]
            if f1["p"] is not None:
                A("三策略整体：Friedman p = "
                  + "、".join(f"top-{k} {fmt_p(inter['topk'][k]['friedman']['p'])}"
                              f"{sig(inter['topk'][k]['friedman']['p'])}"
                              for k in KS)
                  + " → "
                  + ("三策略的 workup 效应不齐一（策略间存在显著交互）"
                     if any(inter["topk"][k]["friedman"]["p"] < 0.05 for k in KS)
                     else "没有证据表明三个策略的 workup 效应不同（未见显著交互）")
                  + "。\n")
            else:
                A(f"三策略整体：Friedman 不可用（{f1['note']}）。\n")

    # 5. 分层
    A("## 5. 分层分析\n")
    A("每层给出两条件的 top-1/3/5 命中率"
      + ("（跨 seed 均值）" if len(seeds) > 1 else f"（s{seeds[0]}）")
      + f"+ workup vs baseline 病例级精确 McNemar（表中独对/p 列为 s{seeds[0]}"
      + ("；另附多数决汇总）。" if len(seeds) > 1 else "）。") + "\n")
    for skey, res in results["strata"].items():
        if skey == "all":
            continue
        A(f"### {res['name']}（n={res['n']}）\n")
        if res["n"] == 0:
            A("（本层为空）\n")
            continue
        hdr = ["方案", "top-k", "workup", "baseline"]
        if len(seeds) > 1:
            hdr = ["方案", "top-k", "workup（均值）", "baseline（均值）"]
        hdr += ["独对 (w:b)", "McNemar p"]
        A("| " + " | ".join(hdr) + " |")
        A("|---" * len(hdr) + "|")
        s0 = seeds[0]
        for m, _ in SCHEMES:
            for k in KS:
                if len(seeds) > 1:
                    aw = res["mean_sd"][m]["workup"][k]["mean"]
                    ab = res["mean_sd"][m]["baseline"][k]["mean"]
                else:
                    aw = res["accuracy"][m]["workup"].get(s0, {}).get(k)
                    ab = res["accuracy"][m]["baseline"].get(s0, {}).get(k)
                aws = "NA" if aw is None else f"{pc(aw)}%"
                abs_ = "NA" if ab is None else f"{pc(ab)}%"
                c = res["comparisons"][m]["per_seed"].get(s0, {}).get(k)
                if c is None or c["n_cases"] == 0:
                    A(f"| {SCHEME_LABEL[m]} | top-{k} | {aws} | {abs_} | — | — |")
                    continue
                if c["n_excluded_unjudged"]:
                    A(f"| {SCHEME_LABEL[m]} | top-{k} | {aws} | {abs_} | "
                      f"{c['workup_only']}:{c['baseline_only']}"
                      f"（+{c['n_excluded_unjudged']} 例未判定剔除） | "
                      f"{fmt_p(c['mcnemar_p'])}{sig(c['mcnemar_p'])} |")
                    continue
                A(f"| {SCHEME_LABEL[m]} | top-{k} | {aws} | {abs_} | "
                  f"{c['workup_only']}:{c['baseline_only']} | "
                  f"{fmt_p(c['mcnemar_p'])}{sig(c['mcnemar_p'])} |")
        if len(seeds) > 1:
            A("")
            for m, _ in SCHEMES:
                cons = res["comparisons"][m].get("consensus")
                if not cons:
                    continue
                A(f"- {SCHEME_LABEL[m]} 多数决（{len(seeds)} seeds）："
                  + "；".join(
                      f"top-{k} {pc(cons[k]['acc_workup'])}% vs "
                      f"{pc(cons[k]['acc_baseline'])}%，独对 "
                      f"{cons[k]['workup_only']}:{cons[k]['baseline_only']}，"
                      f"p={fmt_p(cons[k]['mcnemar_p'])}{sig(cons[k]['mcnemar_p'])}"
                      for k in KS))
        it = res["interaction"]
        if not it.get("can_do"):
            A(f"- 交互检验（d_MDT−d_A×1）：{it.get('reason', '无法计算')}")
        else:
            A(f"- 交互检验（病例级 d_MDT−d_A×1，参与 seed {it['seeds']}）："
              + "；".join(
                  f"top-{k} n={it['topk'][k]['n_cases']}，"
                  f"{pp(it['topk'][k]['pairs']['MDT_vs_Ax1']['mean'])}pp"
                  f"（p={fmt_p(it['topk'][k]['pairs']['MDT_vs_Ax1']['wilcoxon_p'])}"
                  f"{sig(it['topk'][k]['pairs']['MDT_vs_Ax1']['wilcoxon_p'])}）"
                  for k in KS)
              + "；Friedman（三策略齐一性）p="
              + "/".join(f"{fmt_p(it['topk'][k]['friedman']['p'])}"
                         f"{sig(it['topk'][k]['friedman']['p'])}" for k in KS))
        A("")

    # 6. 缺失
    A("## 6. 缺失与待补\n")
    problems = [r for r in availability_rows(ev, seeds, n_cases)
                if r[5] != "完整"]
    if not problems:
        A(f"无：请求范围内 {len(seeds)} 个 seed × 3 方案 × 2 条件的文件都跑满"
          f" {n_cases} 行，判定缓存无缺失对。\n")
    else:
        A("| 条件 | 方案 | seed | 行数 | 缺判对数 | 状态 |")
        A("|---|---|---|---|---|---|")
        for cond, m, s, n, miss, status in problems:
            A(f"| {COND_LABEL[cond]} | {SCHEME_LABEL[m]} | {s} | "
              f"{'—' if n is None else n} | {'—' if miss is None else miss} | "
              f"{status} |")
        A("")
    A("判定缺失只会让对应病例在该 (方案, k) 上被剔除并计入 `n_excluded_unjudged`，"
      "不会静默当成判错。\n")

    # 7. 多 seed 病例级聚合（与 stats_caselevel.py 对齐的口径）
    write_case_level(A, results, seeds)

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写出 {OUT_MD}", flush=True)


# ---------- 主流程 ----------

def main():
    phase = os.environ.get("PHASE", "all").lower()
    seeds, seeds_requested = selected_seeds()
    ev = Eval(seeds, {})
    n_cases = ev.n_cases
    print(f"[检查] 参与比较的 seed：{seeds} | 病例数 {n_cases}", flush=True)

    # --- 判定 ---
    added = 0
    if phase in ("all", "judge"):
        rows = []
        skipped = []
        for (cond, m, s), p in ev.paths.items():
            if ev.rowcount[(cond, m, s)] != n_cases and not JUDGE_INCOMPLETE:
                skipped.append((cond, m, s, ev.rowcount[(cond, m, s)]))
                continue
            rows.extend(ev.runs[(cond, m, s)].values())
        for cond, m, s, n in skipped:
            print(f"[GLM 判定] 跳过未跑完的文件 {COND_LABEL[cond]} "
                  f"{SCHEME_LABEL[m]} s{s}（{n}/{n_cases} 行）；"
                  f"JUDGE_INCOMPLETE=1 可强制判定", flush=True)
        cache, added = judge_missing(rows)
        ev = Eval(seeds, cache)
    if phase == "judge":
        return

    cache = json.loads(GLM_CACHE.read_text()) if GLM_CACHE.exists() else {}
    if phase == "analyze":
        ev = Eval(seeds, cache)
    missing_after = sum(m for k, m in ev.missing_pairs.items()
                        if ev.rowcount[k] == n_cases)
    incomplete = [(k, ev.rowcount[k]) for k in ev.paths
                  if ev.rowcount[k] != n_cases]
    print(f"[检查] 已跑满文件的判定缺失对合计 {missing_after}；"
          f"未跑完文件 {len(incomplete)} 个", flush=True)

    # --- 分析 ---
    strata = build_strata(ev, seeds)
    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge": {"model": GLM_MODEL, "prompt": "v3 (judge_v3.V3_PROMPT)",
                  "cache": GLM_CACHE.name, "cache_size": len(cache),
                  "pairs_added_this_run": added,
                  "pairs_missing_after_judge": missing_after},
        "n_cases": n_cases, "seeds": seeds,
        "seeds_requested": seeds_requested,
        "key_note": "JSON 中所有 seed / top-k 字典的键在为可读性已序列化为字符串"
                    "（如 \"1\"/\"3\"/\"5\"），与 JSON 标准一致。",
        "stats_note": "多 seed 结果一律基于每病例跨 seed 的 0–1 命中率（不把 seed "
                      "当独立样本合并）。strata[*].case_level = 病例级 workup vs "
                      "baseline（配对 Wilcoxon 双侧 + 病例级 cluster bootstrap "
                      "95% CI + 多数决 McNemar）；strata[*].interaction = "
                      "策略 × 条件交互（d_strategy = rate_workup − rate_baseline "
                      "的跨策略配对 Wilcoxon + Friedman + Holm 事后）；每格都带 "
                      "n_seeds / n_cases（分母）/ n_excluded_unjudged，未判定病例"
                      "一律剔除、不当作判错。",
        "conditions": {c: str(COND_DIRS[c]) for c in COND_DIRS},
        "availability": [
            {"condition": cond, "scheme": m, "seed": s, "rows": n,
             "missing_judgements": miss, "status": status}
            for cond, m, s, n, miss, status in availability_rows(ev, seeds, n_cases)],
        "strata": {},
    }
    for skey, (sname, sids) in strata.items():
        print(f"[分析] {skey}（n={len(sids)}）", flush=True)
        res = analyze_stratum(ev, sids, seeds)
        res["name"] = sname
        results["strata"][skey] = res
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"写出 {OUT_JSON}", flush=True)
    write_report(ev, results, seeds, added, len(cache), missing_after)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
