#!/usr/bin/env python3
"""ax5_full87.py 与 seeds_87_dsflash.py 共用的判分 / 病例级统计库。

口径严格复刻 stats_caselevel.py（该脚本模块级有副作用、import 会重写结果文件，
故此处复刻其函数而不 import）：
- 命中 flags：top5 候选逐个在 judge_cache_glm_v3.json 中查表，缺失记 None。
- topk：前 k 个候选中任一 True 即命中；全部缺失判官返回 None。
- 新口径 1：每病例 × 方案 × 指标，5 seeds 命中率（0-1），跨病例配对
  Wilcoxon 符号秩检验（zero_method="wilcox"，双侧）。
- 新口径 2：病例级 cluster bootstrap（按病例重抽样 10,000 次）均值差 95% CI。
- 新口径 3：多数决（>=3/5 seeds 命中记为对）精确 McNemar。
- 旧口径：跨 seed 合并配对 McNemar（作对照）。

判分：GLM-5.3-flash × judge_v3.V3_PROMPT，共享缓存 judge_cache_glm_v3.json
（键 gold[:150]+"||"+cand[:150]），只补缺失对；6 并发 / 150s 超时 / 最多 3 轮。
"""
import json
import math
import os
import time
import fcntl
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

import numpy as np
import requests
from scipy.stats import wilcoxon

from judge_v3 import V3_PROMPT  # 调用方需先设好 env 并注入 sys.path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "routing_study" / "results"
GLM_CACHE = Path(os.environ.get("GLM_CACHE",
                                RESULTS / "judge_cache_glm_v3.json"))
GLM_MODEL = "glm-5.3-flash"
# 判定通道：GLM 官网（Zhipu 开放平台）；可用 GLM_JUDGE_URL/KEY/MODEL 覆盖
GLM_URL = os.environ.get(
    "GLM_JUDGE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
GLM_JUDGE_MODEL = os.environ.get("GLM_JUDGE_MODEL", "GLM-5.3-Flash")
GLM_JUDGE_KEY = os.environ.get("ZHIPU_API_KEY", "")
GLM_WORKERS = int(os.environ.get("GLM_WORKERS", "2"))  # 限流期低并发（6 会持续刺激限流器）
GLM_TIMEOUT = 150

# ---- 判官侧限流熔断（与 run_inference 的全局熔断同理；此文件刻意独立不导入它）----
_JUDGE_LOCK = __import__("threading").Lock()
_JUDGE = {"pause_until": 0.0}


def _judge_gate():
    while True:
        with _JUDGE_LOCK:
            wait = _JUDGE["pause_until"] - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 5.0))


def _judge_event():
    with _JUDGE_LOCK:
        _JUDGE["pause_until"] = time.time() + 60.0
ROUND_BUDGET = 20 * 60
N_BOOT = 10000
BOOT_SEED = 20260917
METHOD_KEYS = [1, 3, 5]

HELDOUT_FILE = ROOT / "data" / "mgh_qa_dataset_new_cases.json"
DEV_AX1 = RESULTS / "topn_ablation" / "ax1.jsonl"
FULL_MERGED = ROOT / "data" / "mgh_qa_dataset_merged.json"


def _read_zhipu_key():
    if os.environ.get("ZHIPU_API_KEY"):
        return os.environ["ZHIPU_API_KEY"].strip()
    for line in open(ROOT / "scripts" / ".env"):
        if line.startswith("ZHIPU_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ZHIPU_API_KEY 缺失（scripts/.env）")


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


# ---------- 判分（GLM v3，只补缺失对） ----------

_HARD_LEAKS = {"n": 0}


def _post_hard_deadline(url, total_s, **kw):
    """requests.post 带强制总时限（防滴水式缓速卡死，见 run_inference.post_with_deadline）。"""
    import threading
    box = {}

    def _run():
        try:
            box["resp"] = requests.post(url, **kw)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(total_s)
    if th.is_alive():
        _HARD_LEAKS["n"] += 1
        if _HARD_LEAKS["n"] > 128:
            raise RuntimeError("hard-deadline 泄漏线程超上限，疑似持续限流")
        raise requests.exceptions.Timeout(f"hard deadline {total_s}s exceeded")
    if "err" in box:
        raise box["err"]
    return box["resp"]

def glm_verdict(gold, cand, key=None):
    body = {"model": GLM_MODEL,
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    _judge_gate()
    body["model"] = GLM_JUDGE_MODEL
    auth = key if (key and key.startswith("sk-")) else GLM_JUDGE_KEY
    # 裁判流量只含参考标签与候选诊断字符串（无病例文本），按策略直连官方端点；
    # OpenRouter 零保留路由仅适用于携带 ER 病例文本的生成类调用（见 run_inference）。
    r_url = GLM_URL
    r = _post_hard_deadline(r_url, int(os.environ.get("GLM_HARD_DEADLINE", "600")),
                            headers={"Authorization": f"Bearer {auth}"},
                            json=body, timeout=GLM_TIMEOUT)
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def missing_pairs(rows, cache):
    """rows: 可迭代的 row dict（用 top5 字段）；返回 {key: (gold, cand)}。"""
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            k = key_of(row["gold"], cand)
            if k not in cache:
                jobs[k] = (row["gold"], cand)
    return jobs


def judge_missing(rows, cache_path=None):
    """对 rows 中所有 (gold, cand) 只补缺失判定，写回共享缓存。返回缓存 dict。

    全程持有 <cache>.lock 独占锁（2026-09-21 加装）：并发判分进程在此串行，
    获锁后重新读缓存——后到者只补先到者仍未覆盖的缺失对，杜绝
    "读旧-判-整写回"互相覆盖导致判定丢失（ax5_mod_mcr 无效版事故的根治）。
    """
    path = Path(cache_path or GLM_CACHE)
    _lock = open(str(path) + ".lock", "w")
    fcntl.flock(_lock, fcntl.LOCK_EX)
    try:
        return _judge_missing_locked(rows, path)
    finally:
        fcntl.flock(_lock, fcntl.LOCK_UN)
        _lock.close()


def _judge_missing_locked(rows, path):
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = _read_zhipu_key()
    jobs = missing_pairs(rows, cache)
    print(f"[GLM 判定] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
    todo = list(jobs.items())
    for round_no in (1, 2, 3):
        if not todo:
            break
        errs = []

        _call_interval = float(os.environ.get("GLM_CALL_INTERVAL", "0"))

        def work(item):
            try:
                if _call_interval:
                    time.sleep(_call_interval)
                v = glm_verdict(item[1][0], item[1][1])
                return item[0], v
            except Exception:
                _judge_event()
                return item[0], None

        # as_completed + 轮次时间预算：个别挂死的连接不阻塞整轮
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
                    path.write_text(json.dumps(cache, ensure_ascii=False))
                    print(f"  轮{round_no} {n}/{len(todo)} | 失败 {len(errs)}",
                          flush=True)
        except TimeoutError:
            stuck = [it for f, it in futs.items() if it[0] not in done_keys]
            print(f"  轮{round_no} 超时（{ROUND_BUDGET}s），{len(stuck)} 个连接"
                  f"挂死，放弃本轮重试之", flush=True)
            errs.extend(stuck)
        ex.shutdown(wait=False, cancel_futures=True)
        path.write_text(json.dumps(cache, ensure_ascii=False))
        todo = errs
        print(f"[GLM 判定] 轮{round_no} 结束：失败 {len(errs)}", flush=True)
        if todo and round_no < 3:
            cool = min(60 * round_no, 300)
            print(f"[GLM 判定] 有 {len(todo)} 对未决，冷却 {cool}s 后进下一轮",
                  flush=True)
            time.sleep(cool)
    if todo:
        print(f"[GLM 判定] 警告：{len(todo)} 对仍未解析", flush=True)
    return cache


# ---------- 病例级统计口径（复刻 stats_caselevel.py） ----------

def load(path):
    """jsonl → {case_id: row}。"""
    path = Path(path)
    out = {}
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["case_id"]] = row
    return out


def hit_flags(row, cache, top5_key="top5", missing=None):
    """top5 候选的判官 flags（True/False/None）。missing: 可选计数器 list。"""
    flags = []
    for c in row[top5_key][:5]:
        k = key_of(row["gold"], c)
        if k in cache:
            flags.append(bool(cache[k]))
        else:
            flags.append(None)
            if missing is not None:
                missing[0] += 1
    return flags


def topk(flags, k):
    f = flags[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def hit_rank(row, cache, top5_key="top5"):
    flags = hit_flags(row, cache, top5_key)
    return next((r for r, v in enumerate(flags, start=1) if v), None)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n, 1.0)


def case_rates(runs, ids, k, top5_keys):
    """runs: {seed: {cid: row}}；top5_keys: {seed: 字段名}。
    返回 cid -> 命中率（仅全部 seeds 可判定的病例）。"""
    rates = {}
    for cid in ids:
        vals = []
        for s, rows in runs.items():
            row = rows.get(cid)
            if row is None:
                vals = None
                break
            vals.append(topk(hit_flags(row, CACHE_HOLDER["cache"],
                                       top5_keys.get(s, "top5")), k))
        if not vals or any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / float(len(vals))
    return rates


def case_majority(runs, ids, k, top5_keys, thresh=0.6):
    out = {}
    for cid in ids:
        vals = []
        for s, rows in runs.items():
            row = rows.get(cid)
            if row is None:
                vals = None
                break
            vals.append(topk(hit_flags(row, CACHE_HOLDER["cache"],
                                       top5_keys.get(s, "top5")), k))
        if not vals or any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) / len(vals) >= thresh)
    return out


# 模块级缓存句柄：case_rates/case_majority 通过它取判官缓存，
# 避免把 cache 透传进每个调用点。
CACHE_HOLDER = {"cache": {}}


def set_cache(cache):
    CACHE_HOLDER["cache"] = cache


def boot_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    """病例级 cluster bootstrap：按病例重抽样，均值差 95% 百分位 CI。

    每次调用用固定种子（与 stats_caselevel.py 的 RNG 同种子），保证可复现。"""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def wilcoxon_p(ra, rb):
    ra = np.asarray(ra, dtype=float)
    rb = np.asarray(rb, dtype=float)
    if len(ra) == 0:
        return None
    if np.all(ra - rb == 0):
        return 1.0
    return float(wilcoxon(ra, rb, zero_method="wilcox").pvalue)


def paired_caselevel(runs_a, runs_b, ids, k, keys_a, keys_b):
    """stats_caselevel.py 的三口径配对比较（A vs B，top-k）。"""
    rates_a = case_rates(runs_a, ids, k, keys_a)
    rates_b = case_rates(runs_b, ids, k, keys_b)
    common = sorted(set(rates_a) & set(rates_b))
    ra = [rates_a[c] for c in common]
    rb = [rates_b[c] for c in common]
    md, lo, hi = boot_ci([a - b for a, b in zip(ra, rb)])
    wp = wilcoxon_p(ra, rb)

    maj_a = case_majority(runs_a, ids, k, keys_a)
    maj_b = case_majority(runs_b, ids, k, keys_b)
    cm = sorted(set(maj_a) & set(maj_b))
    ao = sum(1 for c in cm if maj_a[c] and not maj_b[c])
    bo = sum(1 for c in cm if maj_b[c] and not maj_a[c])

    # 旧口径：跨 seed 合并 McNemar（违反独立性，仅作对照）
    pao = pbo = 0
    for s in runs_a:
        rows_b_seed = runs_b.get(s, {})
        for cid in ids:
            row_a = runs_a[s].get(cid)
            row_b = rows_b_seed.get(cid)
            if row_a is None or row_b is None:
                continue
            ha = topk(hit_flags(row_a, CACHE_HOLDER["cache"],
                               keys_a.get(s, "top5")), k)
            hb = topk(hit_flags(row_b, CACHE_HOLDER["cache"],
                               keys_b.get(s, "top5")), k)
            if ha and not hb:
                pao += 1
            elif hb and not ha:
                pbo += 1

    return {
        "n_cases": len(common),
        "mean_rate_a": float(st.mean(ra)) if ra else None,
        "mean_rate_b": float(st.mean(rb)) if rb else None,
        "mean_diff": md,
        "boot95_ci": [lo, hi],
        "wilcoxon_p": wp,
        "majority": {"a_only": ao, "b_only": bo,
                     "mcnemar_p": mcnemar_exact(ao, bo)},
        "pooled_mcnemar": {"a_only": pao, "b_only": pbo,
                           "p": mcnemar_exact(pao, pbo)},
    }


def per_seed_metrics(runs, ids, keys, missing=None):
    """每 seed 的 top-1/3/5 命中数 + 病例级均值。"""
    out = []
    for s in sorted(runs):
        t = {k: 0 for k in METHOD_KEYS}
        n = 0
        for cid in ids:
            row = runs[s].get(cid)
            if row is None:
                continue
            n += 1
            flags = hit_flags(row, CACHE_HOLDER["cache"],
                              keys.get(s, "top5"), missing)
            hit = next((r for r, v in enumerate(flags, start=1) if v), None)
            if hit:
                for k in METHOD_KEYS:
                    if hit <= k:
                        t[k] += 1
        out.append({"seed": s, "n": n,
                    **{f"top{k}": t[k] for k in METHOD_KEYS},
                    **{f"top{k}_acc": (t[k] / n if n else None)
                       for k in METHOD_KEYS}})
    return out


def mean_sd(per_seed):
    """每 seed 的 top-k 准确率 → mean±SD。"""
    out = {}
    for k in METHOD_KEYS:
        vals = [p[f"top{k}_acc"] for p in per_seed if p[f"top{k}_acc"] is not None]
        if not vals:
            out[f"top{k}"] = {"mean": None, "sd": None, "n_seeds": 0}
        else:
            out[f"top{k}"] = {
                "mean": round(st.mean(vals), 4),
                "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals)}
    return out


def with_variant(rows_by_seed, field):
    """把 {seed: {cid: row}} 的 top5 字段换成 field 指向的变体。"""
    out = {}
    for s, rows in rows_by_seed.items():
        out[s] = {cid: {**row, "top5": row[field]} for cid, row in rows.items()}
    return out


def split_ids():
    """heldout46 / dev41 / full87 的 case_id 集合（口径同 holdout_primary.py）。"""
    heldout = [c["case_id"] for c in json.loads(HELDOUT_FILE.read_text())]
    dev = list(load(DEV_AX1))
    full = [c["case_id"] for c in json.loads(FULL_MERGED.read_text())]
    return {"full87": full, "heldout46": heldout, "dev41": dev}


def arm_stats(runs, ids, missing=None):
    """一条臂（{seed: {cid: row}}，row 已归一为 top5 字段）在给定病例集上的指标。"""
    keys = {s: "top5" for s in runs}
    per = per_seed_metrics(runs, ids, keys, missing)
    rate_mean = {}
    for k in METHOD_KEYS:
        rates = case_rates(runs, ids, k, keys)
        rate_mean[f"top{k}"] = float(st.mean(rates.values())) if rates else None
    return {"per_seed": per, "mean_sd": mean_sd(per),
            "case_rate_mean": rate_mean}


def compare_all(arms, ids, pairs):
    """pairs: [(a, b)] → {f"{a}_vs_{b}": {top-k: paired_caselevel(...)}}。"""
    out = {}
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        keys = {s: "top5" for s in arms[a]}
        out[f"{a}_vs_{b}"] = {
            f"top{k}": paired_caselevel(arms[a], arms[b], ids, k, keys, keys)
            for k in METHOD_KEYS}
    return out


def split_stats(arms, splits, pairs, missing=None):
    """对每个病例集算各臂指标 + 配对比较。"""
    out = {}
    for name, ids in splits.items():
        m = [0]
        out[name] = {
            "n_cases": len(ids),
            "arms": {a: arm_stats(runs, ids, m) for a, runs in arms.items()},
            "judge_missing_pairs": m[0],
            "comparisons": compare_all(arms, ids, pairs),
        }
    return out


def fmt_p(p):
    if p is None:
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def acc_cell(ms, k):
    m = ms[f"top{k}"]["mean"]
    s = ms[f"top{k}"]["sd"]
    if m is None:
        return "NA"
    return f"{m*100:.1f} ± {s*100:.1f}"
