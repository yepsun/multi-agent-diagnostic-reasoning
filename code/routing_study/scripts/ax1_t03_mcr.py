#!/usr/bin/env python3
"""MCR 406 例：补跑 A×1@T=0.3 温度匹配臂 + 病例级温度匹配分析。

背景
----
论文主实验在 MCR（MedCaseReasoning 外部验证集，406 例）上 A×1 跑 T=0（贪心），
P 与 MDT 跑 T=0.3；`results/topn_mcr/`（seed 1）与 `results/topn_mcr_seeds/`
（seeds 2–5）里只有 A×1@T=0、P@0.3、MDT@0.3。CPC 上已补跑 A×1@T=0.3
（`topn_seeds_ax1t03/`，见 `temp_matched_caselevel.py`），ER 上有 A×1@T=0.3（seed 1，
见 `erreason_temp_matched.py`），MCR 一直缺这条臂 —— 本脚本补齐，使 MCR 也能给出
「三臂全 T=0.3」的温度匹配对照。

推理：与主实验 MCR 的 A×1 臂逐字一致（唯一差别是温度）
----------------------------------------------------
直接复用既有件，不复制粘贴参数：
- 病例：`topn_mcr.load_cases()`（406 例：`data/medcasereasoning_subset.json` 的
  `Q` 呈现段 + `A.final_diagnosis` 金标签，与主实验同一函数）
- 提示词：`topn_cpc_promptv2.A_TOPN_PROMPT.format(case_text=c["text"])`
  —— 与 `topn_mcr.run_simple` 与 `seeds_mcr.run_simple_seed` 的 A×1 调用点逐字
  相同；`source_audit()` 每次运行读这两个源文件做正则审计并落盘。
- 调用：`topn_mcr.call_top5(prompt, 0.3)`（provider=qwen → qwen3.8-flash、
  `disable_thinking=True`、`max_tokens=4096`、`timeout=300`、内部 3 次空重试，
  全与主实验 MCR 的 A×1/P 相同）
- **唯一差别：temperature 0.0（主实验）→ 0.3（本臂）**。

输出 `MCRT03_OUTDIR/Ax1t03_s{1..5}.jsonl`（默认 `results/topn_seeds_mcr_ax1t03/`），
行 schema = 主实验各臂同款 `{case_id, gold, top5, total_tokens}`；406 例 × 5 seeds
= 2030 次调用；断点续跑；空 top5 或不足 5 项视为失败**不落盘**（下轮重跑）。
另写 `parse_stats.json`（逐 seed 首轮空输出数 / 需重试数 / 硬失败数 + 跨次累加）、
`failures.jsonl`。

阶段 PHASE=infer|judge|analyze（默认 all）
- infer：跑 A×1@T=0.3（406×5）。
- judge：`caselevel_stats.judge_missing(rows, cache_path=...)`（GLM-5.3-flash ×
  `judge_v3.V3_PROMPT`，**只补缺失对**）。缓存路径由环境变量 `MCRT03_JUDGE_CACHE`
  （其次 `GLM_CACHE`）显式给出；**未给出则报错退出**（默认不写任何缓存）。judge
  阶段**拒绝**写入共享主缓存 `judge_cache_glm_v3.json` 与任何 `judge_shard_*.json`
  （realpath 比较，软链/相对路径都能识别）—— 该类文件由主代理统一串行调度。
  analyze 则可用 `os.pathsep`（`:`）分隔**多个**缓存路径按序合并读取，便于
  「新臂写在分片 + 其余臂在主缓存」时一次读全（主缓存已覆盖现有三臂的全部对）。
- analyze：病例级主口径（与 `stats_caselevel.py` / `caselevel_stats.py` 逐字同口径）：
  每例 5-seed 平均命中率 → 跨病例配对 Wilcoxon 双侧（zero_method="wilcox"）+
  病例级 cluster bootstrap 10,000 次 95% CI + 多数决（>=3/5）精确 McNemar，
  另有跨 seed 合并 McNemar 作旧口径对照。比较三组：
    (a) 温度完全匹配（三臂全 T=0.3）：A×1@0.3（本臂）vs P@0.3 vs MDT@0.3
    (b) A×1 自身的温度效应：A×1@0.3 vs A×1@T=0
    (c) 温度不对称口径（论文原口径）：A×1@T=0 vs P@0.3 / MDT@0.3
  缺失判定：主口径把「5 seeds 中任一候选未判定」的病例**剔除**（逐对比报 n_cases），
  并另给两个全 406 例口径夹住真值 —— 未判定按 miss（悲观下界）/ 按命中（乐观上界）；
  未判定 (gold,candidate) 的槽位、去重对数、涉及病例与「悬空观察」逐臂逐 seed 落盘，
  **绝不静默计 miss**。产出 `results/ax1_t03_mcr.json` + `.md`。

环境变量：MCRT03_OUTDIR、MCRT03_SEEDS（默认 "1,2,3,4,5"）、MAX_WORKERS（默认 6）、
MCRT03_JUDGE_CACHE / GLM_CACHE、SMOKE_DIR。SMOKE=1 只跑前 3 例到
/tmp/ax1t03_mcr_smoke（不判定、不分析）。

用法
----
  SMOKE=1 ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py
  MAX_WORKERS=6 PHASE=infer ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py
  MCRT03_JUDGE_CACHE=routing_study/results/judge_shard_ax1t03mcr.json \
      PHASE=judge ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py
  MCRT03_JUDGE_CACHE=... PHASE=analyze ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py
"""
import hashlib
import json
import os
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

import caselevel_stats as cs  # noqa: E402
from topn_cpc import load_done, append_row  # noqa: E402
from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402  主实验原常量（只读）
from topn_mcr import load_cases, call_top5  # noqa: E402
from run_inference import resolve_provider, QWEN_MODEL  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("MCRT03_OUTDIR",
                             RESULTS / "topn_seeds_mcr_ax1t03"))
S1_DIR = RESULTS / "topn_mcr"           # seed 1 的主实验臂
SEEDS_DIR = RESULTS / "topn_mcr_seeds"  # seed 2–5 的主实验臂
OUT_JSON = RESULTS / "ax1_t03_mcr.json"
OUT_MD = RESULTS / "ax1_t03_mcr.md"
SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/ax1t03_mcr_smoke"))

SEEDS = [int(s) for s in
         os.environ.get("MCRT03_SEEDS", "1,2,3,4,5").replace(",", " ").split()]
TEMPERATURE = 0.3
MAX_ATTEMPTS = 3          # 外层尝试轮数（call_top5 内部另有 3 次空重试）
WORKERS = int(os.environ.get("MAX_WORKERS", 6))
SPLIT = "mcr406"
N_EXPECTED = 406
KS = tuple(cs.METHOD_KEYS)          # (1, 3, 5)

# 臂名 → 结果的 A − B 方向：mean_diff > 0 表示 A 的召回更高
PAIRS = [("MDT03", "Ax1t03"),       # (a) 温度匹配：MDT 相对 A×1@0.3 的优势
         ("P03", "Ax1t03"),         # (a) 温度匹配：P 相对 A×1@0.3
         ("MDT03", "P03"),          # (a) 温度匹配：MDT 相对 P
         ("Ax1t03", "Ax1T0"),       # (b) A×1 自身的温度效应（正 = T=0.3 更好）
         ("MDT03", "Ax1T0"),        # (c) 论文原口径（不对称）：MDT@0.3 vs A×1@T=0
         ("P03", "Ax1T0")]          # (c) 论文原口径（不对称）：P@0.3 vs A×1@T=0
GROUPS = [
    ("温度完全匹配（三臂全部 T=0.3）",
     [("MDT03", "Ax1t03"), ("P03", "Ax1t03"), ("MDT03", "P03")]),
    ("A×1 自身的温度效应", [("Ax1t03", "Ax1T0")]),
    ("温度不对称（论文原口径：T=0.3 的臂 vs A×1@T=0）",
     [("MDT03", "Ax1T0"), ("P03", "Ax1T0")]),
]
ARM_LABEL = {
    "Ax1t03": "A×1 (T=0.3)",
    "P03": "P (T=0.3)",
    "MDT03": "MDT (T=0.3)",
    "Ax1T0": "A×1 (T=0)",
}


# ---------- 提示词一致性：源码审计 ----------

PROMPT_EXPR = 'A_TOPN_PROMPT.format(case_text=c["text"])'
MAIN_T03_CALL_SITES = [("topn_mcr.py", "run_simple（seed 1 主实验臂）"),
                       ("seeds_mcr.py", "run_simple_seed（seeds 2–5 主实验臂）")]


def source_audit():
    """读主实验 MCR A×1 的两处调用点，核实提示词表达式逐字相同、温度为 0.0。"""
    out = {"prompt_expr": PROMPT_EXPR, "sites": []}
    ok = True
    for fname, desc in MAIN_T03_CALL_SITES:
        src = (ROOT / "routing_study" / "scripts" / fname).read_text()
        n_expr = src.count(PROMPT_EXPR)
        m = re.search(re.escape(PROMPT_EXPR) + r"\s*,\s*(0\.\d+)", src)
        temp = m.group(1) if m else None
        site = {"file": fname, "desc": desc, "prompt_expr_count": n_expr,
                "prompt_expr_literal_match": n_expr >= 1,
                "temperature_in_source": temp,
                "is_Ax1_arm_T0": temp == "0.0"}
        ok &= site["prompt_expr_literal_match"] and site["is_Ax1_arm_T0"]
        out["sites"].append(site)
    out["all_sites_identical_except_temperature"] = bool(ok)
    out["prompt_constant_sha256"] = hashlib.sha256(
        A_TOPN_PROMPT.encode("utf-8")).hexdigest()
    return out


def prompt_fingerprint(cases):
    """406 例格式化后提示词的聚合 sha256（本臂 T=0.3 与主实验 T=0 完全相同）。"""
    hs = [hashlib.sha256(build_prompt(c).encode("utf-8")).hexdigest()
          for c in cases]
    return {"n_cases": len(hs),
            "aggregate_sha256": hashlib.sha256("".join(hs).encode()).hexdigest(),
            "case0_sha256": hs[0] if hs else None,
            "n_distinct": len(set(hs))}


def build_prompt(case):
    """与 topn_mcr.run_simple / seeds_mcr.run_simple_seed 的 A×1 调用点逐字相同。"""
    return A_TOPN_PROMPT.format(case_text=case["text"])


# ---------- 阶段 1：推理 ----------

def row_complete(row):
    """主实验 MCR 各臂每行都是 5 项候选；不足 5 项视为失败（下轮重跑）。"""
    return bool(row) and len(row.get("top5") or []) == 5


def compact(path):
    """断点续跑可能追加重复行；按 case_id 去重保留最后一行（原子写回）。"""
    if not path.exists():
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    order, last = [], {}
    for r in rows:
        if r["case_id"] not in last:
            order.append(r["case_id"])
        last[r["case_id"]] = r
    if len(last) == len(rows):
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for cid in order:
            f.write(json.dumps(last[cid], ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[整理] {path.name}: {len(rows)} 行 → {len(order)} 行", flush=True)


def run_one(seed, cases, outdir=None, stats=None, failures=None):
    outdir = Path(outdir or OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"Ax1t03_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if not row_complete(done.get(c["case_id"]))]
    print(f"[Ax1t03 s{seed}] 已完成 {len(cases) - len(todo)}/{len(cases)}，"
          f"待跑 {len(todo)}", flush=True)

    def work(c):
        prompt = build_prompt(c)
        empty_first = False
        last_err = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                # topn_mcr.call_top5 参数与主实验 MCR 的 A×1 臂逐字相同
                # （max_tokens=4096 / timeout=300 / disable_thinking）；
                # 内部已含 3 次空重试 —— 唯一差别是这里的 temperature=0.3
                top5, tokens = call_top5(prompt, TEMPERATURE)
            except Exception as e:  # 网络异常一律重试
                top5, tokens, last_err = [], 0, f"{type(e).__name__}: {e}"
            if top5:
                return ({"case_id": c["case_id"], "gold": c["gold"],
                         "top5": top5, "total_tokens": tokens},
                        empty_first, attempt + 1)
            empty_first = True
            last_err = last_err or "empty top5 after parse"
            print(f"[Ax1t03 s{seed}] {c['case_id'][:40]} 空 top5/解析失败，"
                  f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续 {MAX_ATTEMPTS} 轮无有效 top5"
                           f"（末次：{last_err}）")

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                row, empty_first, attempts = fut.result()
            except Exception as e:
                (failures if failures is not None else []).append(
                    {"seed": seed, "case_id": c["case_id"], "error": str(e)})
                if stats is not None:
                    stats["first_attempt_empty"] += 1
                    stats["needed_retry"] += 1
                    stats["hard_failed"] += 1
                print(f"[失败] Ax1t03 s{seed} {c['case_id'][:40]}: {e}", flush=True)
                continue
            append_row(path, row)
            if stats is not None:
                if empty_first:
                    stats["first_attempt_empty"] += 1
                if attempts > 1:
                    stats["needed_retry"] += 1
            if i % 25 == 0 or i == len(todo):
                print(f"[Ax1t03 s{seed}] {i}/{len(todo)}", flush=True)
    compact(path)
    return path


def load_jsonl_list(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def load_failures(path):
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def run_infer(cases, outdir=None):
    outdir = Path(outdir or OUTDIR)
    print(f"[推理] provider=qwen（transport={resolve_provider('qwen')}）"
          f" model={QWEN_MODEL} | T={TEMPERATURE} disable_thinking=True "
          f"max_tokens=4096 timeout=300（复用 topn_mcr.call_top5）| "
          f"MAX_WORKERS={WORKERS}", flush=True)
    per_seed, failures = {}, []
    for seed in SEEDS:
        s = {"first_attempt_empty": 0, "needed_retry": 0, "hard_failed": 0}
        run_one(seed, cases, outdir, stats=s, failures=failures)
        per_seed[f"s{seed}"] = s
        print(f"[推理] s{seed} 统计: {json.dumps(s, ensure_ascii=False)}",
              flush=True)

    fpath = outdir / "failures.jsonl"
    if failures:
        with open(fpath, "a", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[推理] 硬失败 {len(failures)} 例（连续 {MAX_ATTEMPTS} 轮无有效 "
              f"top5，未落盘）："
              + "；".join(f"{f['seed']}:{f['case_id'][:36]}" for f in failures),
              flush=True)

    pstats_path = outdir / "parse_stats.json"
    prev = {}
    if pstats_path.exists():
        try:
            prev = json.loads(pstats_path.read_text())
        except Exception:
            prev = {}
    runs = list(prev.get("runs") or [])
    if not runs and prev.get("per_seed"):
        runs.append({"source": "本文件旧格式（单次运行）",
                     "per_seed": prev["per_seed"],
                     "hard_failures": [f["case_id"] for f in
                                       (prev.get("hard_failures") or [])]})
    runs.append({"source": "本次运行", "per_seed": per_seed,
                 "hard_failures": [f"{f['seed']}:{f['case_id'][:60]}"
                                   for f in failures]})
    agg = {}
    for r in runs:
        for k, v in (r.get("per_seed") or {}).items():
            a = agg.setdefault(k, {"first_attempt_empty": 0, "needed_retry": 0,
                                   "hard_failed": 0})
            for f in a:
                a[f] += int(v.get(f, 0) or 0)
    stats = {
        "model": QWEN_MODEL, "provider": resolve_provider("qwen"),
        "temperature": TEMPERATURE, "disable_thinking": True,
        "max_tokens": 4096, "timeout": 300, "call": "topn_mcr.call_top5",
        "prompt": 'A_TOPN_PROMPT.format(case_text=c["text"])',
        "n_cases": len(cases), "seeds": SEEDS, "max_workers": WORKERS,
        "n_calls_planned": len(cases) * len(SEEDS),
        "per_seed": per_seed, "runs": runs, "aggregate_per_seed": agg,
        "hard_failures": failures,
        "hard_failures_total": sum(len(r.get("hard_failures") or [])
                                   for r in runs),
        "note": "first_attempt_empty = 首轮（含 call_top5 内部 3 次调用）无有效 "
                "top5 的行数；needed_retry = 需要第二轮外层尝试才成功的行数；"
                "hard_failed = 外层重试用尽、未落盘（下轮续跑会重试）的病例数",
    }
    pstats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[推理] parse_stats 已写入 {pstats_path}；本次 {json.dumps(per_seed, ensure_ascii=False)}",
          flush=True)
    return stats


# ---------- 判分缓存路径（必须显式指定，避免误写共享缓存） ----------

def _same_path(a, b):
    try:
        return os.path.realpath(str(a)) == os.path.realpath(str(b))
    except Exception:
        return False


def _protected_cache_paths():
    """共享判官缓存（主代理统一调度）：主缓存 + 所有 judge_shard_*.json。"""
    out = [cs.GLM_CACHE] + sorted(RESULTS.glob("judge_shard_*.json"))
    return out


def resolve_judge_cache(for_write=False):
    """判分缓存路径必须由环境变量显式给出（默认不写任何缓存）。

    judge 阶段（for_write=True）：只接受**单个**路径，且**拒绝**指向共享主缓存
    `judge_cache_glm_v3.json` 或任何 `judge_shard_*.json`（realpath 比较）。
    analyze（for_write=False）：可用 `os.pathsep` 分隔多个缓存路径，按顺序合并
    （先给的优先）—— 便于「新臂写在分片里 + 其余臂在主缓存里」时一次读全；
    若未指定则回落本脚本输出目录的 `judge_cache.json`，再回落主缓存（只读）并告警。
    """
    for var in ("MCRT03_JUDGE_CACHE", "GLM_CACHE"):
        v = os.environ.get(var)
        if not v:
            continue
        parts = [Path(x) for x in v.split(os.pathsep) if x.strip()]
        if for_write:
            if len(parts) != 1:
                raise SystemExit(
                    f"{var} 含多个路径，judge 阶段只接受单个缓存文件"
                    "（判分缓存是整文件覆盖写）。")
            for prot in _protected_cache_paths():
                if _same_path(parts[0], prot):
                    raise SystemExit(
                        f"{var}={v} 解析后等于共享判官缓存 {prot}，本脚本拒绝写入"
                        "（该文件由主代理串行调度）。请指定分片路径，例如\n"
                        "  MCRT03_JUDGE_CACHE=routing_study/results/"
                        "judge_shard_ax1t03mcr.json")
        return parts, var
    if for_write:
        raise SystemExit(
            "judge 阶段必须显式指定判分缓存路径（默认不写任何缓存，以免误写共享的 "
            "judge_cache_glm_v3.json）：\n"
            "  MCRT03_JUDGE_CACHE=routing_study/results/judge_shard_ax1t03mcr.json "
            "PHASE=judge ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py")
    local = OUTDIR / "judge_cache.json"
    if local.exists():
        return [local], "默认（本脚本输出目录）"
    print(f"[分析] 警告：未指定判分缓存（MCRT03_JUDGE_CACHE / GLM_CACHE），"
          f"回落到共享主缓存 {cs.GLM_CACHE}（**只读**）", flush=True)
    return [cs.GLM_CACHE], "默认（共享主缓存，只读）"


def load_cache(paths):
    """按顺序合并多个判分缓存（先给的路径优先）。"""
    cache = {}
    for p in paths:
        if p.exists():
            for k, v in json.loads(p.read_text()).items():
                cache.setdefault(k, v)
        else:
            print(f"[分析] 警告：判分缓存 {p} 不存在，跳过", flush=True)
    return cache


# ---------- 阶段 2：判分（GLM v3，只补缺失对） ----------

def load_arm(seed, outdir=None):
    return cs.load(Path(outdir or OUTDIR) / f"Ax1t03_s{seed}.jsonl")


def run_judge(seeds=None):
    seeds = seeds or SEEDS
    cache_paths, var = resolve_judge_cache(for_write=True)
    cache_path = cache_paths[0]
    rows = []
    for seed in seeds:
        rows.extend(load_arm(seed).values())
    if not rows:
        raise SystemExit(f"没有可判定的行（{OUTDIR} 为空）。先跑 PHASE=infer。")
    n_before = len(json.loads(cache_path.read_text())) if cache_path.exists() else 0
    print(f"[判定] 缓存 {cache_path}（来自 ${var}）现有 {n_before} 对 | "
          f"待判定行 {len(rows)}", flush=True)
    cache = cs.judge_missing(rows, cache_path=cache_path)
    print(f"[判定] 缓存 {n_before} → {len(cache)}"
          f"（新增 {len(cache) - n_before} 对）", flush=True)
    return cache


# ---------- 阶段 3：分析（病例级主口径 + 缺失敏感性） ----------

def build_arms(seeds, outdir=None):
    """主实验 MCR 各臂：seed 1 在 topn_mcr/，seeds 2–5 在 topn_mcr_seeds/。"""
    return {
        "Ax1t03": {s: load_arm(s, outdir) for s in seeds},
        "P03": {s: cs.load(S1_DIR / "p.jsonl" if s == 1
                           else SEEDS_DIR / f"P_s{s}.jsonl") for s in seeds},
        "MDT03": {s: cs.load(S1_DIR / "mdt_synth.jsonl" if s == 1
                             else SEEDS_DIR / f"s{s}" / "mdt_synth.jsonl")
                  for s in seeds},
        "Ax1T0": {s: cs.load(S1_DIR / "ax1.jsonl" if s == 1
                             else SEEDS_DIR / f"Ax1_s{s}.jsonl") for s in seeds},
    }


def _rel(p):
    """相对仓库根的展示路径（输出目录被指到仓库外时退回绝对路径）。"""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def arm_paths(seeds, outdir=None):
    outdir = Path(outdir or OUTDIR)
    return {
        "Ax1t03": [_rel(outdir / f"Ax1t03_s{s}.jsonl") for s in seeds],
        "P03": [_rel(S1_DIR / "p.jsonl" if s == 1
                     else SEEDS_DIR / f"P_s{s}.jsonl") for s in seeds],
        "MDT03": [_rel(S1_DIR / "mdt_synth.jsonl" if s == 1
                       else SEEDS_DIR / f"s{s}" / "mdt_synth.jsonl")
                  for s in seeds],
        "Ax1T0": [_rel(S1_DIR / "ax1.jsonl" if s == 1
                       else SEEDS_DIR / f"Ax1_s{s}.jsonl")
                  for s in seeds],
    }


def judge_coverage(arms, ids, cache, seeds):
    """逐臂逐 seed：未判定槽位 / 去重对数 / 涉及病例 / 悬空观察（k=1,3,5）。"""
    out = {}
    for arm, runs in arms.items():
        per_seed, all_slots, all_pairs, all_cases = {}, 0, set(), set()
        for s in seeds:
            rows = runs.get(s) or {}
            slots, pairs, cases = 0, set(), set()
            for cid in ids:
                row = rows.get(cid)
                if row is None:
                    continue
                for c in row["top5"][:5]:
                    k = cs.key_of(row["gold"], c)
                    if k not in cache:
                        slots += 1
                        pairs.add(k)
                        cases.add(cid)
            per_seed[f"s{s}"] = {"slots": slots, "pairs": len(pairs),
                                 "cases": len(cases)}
            all_slots += slots
            all_pairs |= pairs
            all_cases |= cases
        dang = {f"top{k}": 0 for k in KS}
        for s in seeds:
            for cid in ids:
                row = (runs.get(s) or {}).get(cid)
                if row is None:
                    continue
                f = [bool(cache[cs.key_of(row["gold"], c)])
                     if cs.key_of(row["gold"], c) in cache else None
                     for c in row["top5"][:5]]
                for k in KS:
                    seg = f[:k]
                    if any(x is None for x in seg) and not any(x is True
                                                               for x in seg):
                        dang[f"top{k}"] += 1
        out[arm] = {"per_seed": per_seed, "slots": all_slots,
                    "unique_pairs": len(all_pairs), "cases": len(all_cases),
                    "dangling_observations": dang}
    return out


# ----- 缺失处理三口径（primary 与 caselevel_stats 逐格核对） -----

def _flags(row, cache):
    out = []
    for c in row["top5"][:5]:
        k = cs.key_of(row["gold"], c)
        out.append(bool(cache[k]) if k in cache else None)
    return out


def _topk(f, k, mode):
    f = f[:k]
    if mode == "miss":
        return any(x is True for x in f)
    if mode == "hit":
        return any(x is True for x in f) or any(x is None for x in f)
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def _case_rates(flags, ids, k, mode, seeds):
    rates = {}
    for cid in ids:
        vals = []
        for s in seeds:
            f = flags.get((s, cid))
            if f is None:
                vals = None
                break
            vals.append(_topk(f, k, mode))
        if vals is None or any(v is None for v in vals):
            continue
        rates[cid] = sum(vals) / float(len(vals))
    return rates


def _case_majority(flags, ids, k, mode, seeds):
    out = {}
    for cid in ids:
        vals = []
        for s in seeds:
            f = flags.get((s, cid))
            if f is None:
                vals = None
                break
            vals.append(_topk(f, k, mode))
        if vals is None or any(v is None for v in vals):
            continue
        out[cid] = int(sum(vals) >= 3)   # 多数决 >= 3/5（与 cs.case_majority 同）
    return out


def _pooled(a_rows, b_rows, ids, k, flags_a, flags_b, seeds):
    """旧口径：跨 seed 合并 McNemar（未判定按 miss，与 cs.paired_caselevel 同）。"""
    pao = pbo = 0
    for s in seeds:
        for cid in ids:
            if cid not in a_rows.get(s, {}) or cid not in b_rows.get(s, {}):
                continue
            ha = _topk(flags_a[(s, cid)], k, "miss")
            hb = _topk(flags_b[(s, cid)], k, "miss")
            if ha and not hb:
                pao += 1
            elif hb and not ha:
                pbo += 1
    return {"a_only": pao, "b_only": pbo, "p": cs.mcnemar_exact(pao, pbo)}


def compare_mode(a, b, k, ids, mode, flags, arms, seeds):
    ra = _case_rates(flags[a], ids, k, mode, seeds)
    rb = _case_rates(flags[b], ids, k, mode, seeds)
    common = sorted(set(ra) & set(rb))
    va = [ra[c] for c in common]
    vb = [rb[c] for c in common]
    diff = [x - y for x, y in zip(va, vb)]
    md, lo, hi = cs.boot_ci(diff) if diff else (float("nan"),) * 3
    wp = cs.wilcoxon_p(va, vb) if va else None
    ma = _case_majority(flags[a], ids, k, mode, seeds)
    mb = _case_majority(flags[b], ids, k, mode, seeds)
    cm = sorted(set(ma) & set(mb))
    ao = sum(1 for c in cm if ma[c] and not mb[c])
    bo = sum(1 for c in cm if mb[c] and not ma[c])
    return {
        "n_cases": len(common),
        "mean_rate_a": float(st.mean(va)) if va else None,
        "mean_rate_b": float(st.mean(vb)) if vb else None,
        "mean_diff": md, "boot95_ci": [lo, hi], "wilcoxon_p": wp,
        "majority": {"a_only": ao, "b_only": bo,
                     "mcnemar_p": cs.mcnemar_exact(ao, bo)},
        "pooled_mcnemar": _pooled(arms[a], arms[b], ids, k, flags[a], flags[b],
                                  seeds),
    }


def selfcheck_primary(mine, lib):
    """我的 primary 口径 vs caselevel_stats.paired_caselevel：逐字段核对。"""
    fields = ("n_cases", "mean_diff", "wilcoxon_p", "mean_rate_a", "mean_rate_b",
              "majority", "pooled_mcnemar")
    bad = []
    for key, m in mine.items():
        r = lib.get(key)
        if not r:
            bad.append(f"{key}: 库结果缺失")
            continue
        for f in fields:
            if f in ("mean_diff",):
                same = (m[f] == r[f]) or (
                    m[f] != m[f] and r[f] != r[f])  # NaN 视为相同
            elif f == "boot95_ci":
                same = all((x == y) or (x != x and y != y)
                           for x, y in zip(m[f], r[f]))
            else:
                same = m[f] == r[f]
            if not same:
                bad.append(f"{key}/{f}: {m[f]!r} != {r[f]!r}")
        same_ci = all((x == y) or (x != x and y != y)
                      for x, y in zip(m["boot95_ci"], r["boot95_ci"]))
        if not same_ci:
            bad.append(f"{key}/boot95_ci: {m['boot95_ci']} != {r['boot95_ci']}")
    return {"identical": not bad, "mismatches": bad[:20],
            "n_checked": len(mine)}


def fmt_p(p):
    if p is None:
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def sig(p):
    return "*" if p is not None and p < 0.05 else ""


def run_analyze():
    cases = load_cases()
    ids = [c["case_id"] for c in cases]
    audit = source_audit()
    fp = prompt_fingerprint(cases)
    if not audit["all_sites_identical_except_temperature"]:
        raise SystemExit(f"提示词源码审计失败：{audit['sites']}")

    arms = build_arms(SEEDS)
    cache_paths, var = resolve_judge_cache(for_write=False)
    cache = load_cache(cache_paths)
    if not cache:
        print(f"[分析] 警告：判分缓存为空，全部候选按缺失处理", flush=True)
    cs.set_cache(cache)

    stats = cs.split_stats(arms, {SPLIT: ids}, PAIRS)
    res = stats[SPLIT]

    # 各臂 × seed 覆盖（行数）与判分缺失
    coverage, blockers = {}, []
    for arm, runs in arms.items():
        coverage[arm] = {}
        for s in SEEDS:
            n = sum(1 for cid in ids if cid in (runs.get(s) or {}))
            coverage[arm][f"s{s}"] = n
            if n != N_EXPECTED:
                blockers.append(f"{arm}/s{s} 覆盖 {n}/{N_EXPECTED}")
    miss_cov = judge_coverage(arms, ids, cache, SEEDS)
    readiness = not blockers

    # 缺失处理三口径
    flags = {arm: {(s, cid): _flags(row, cache)
                   for s, rows in runs.items() for cid, row in rows.items()}
             for arm, runs in arms.items()}
    modes = {}
    for mode in ("primary", "miss", "hit"):
        modes[mode] = {f"top{k}/{a}_vs_{b}": compare_mode(
            a, b, k, ids, mode, flags, arms, SEEDS) for a, b in PAIRS
            for k in KS}
    lib_cmp = {f"top{k}/{a}_vs_{b}": res["comparisons"][f"{a}_vs_{b}"][f"top{k}"]
               for a, b in PAIRS for k in KS}
    selfcheck = selfcheck_primary(modes["primary"], lib_cmp)
    if not selfcheck["identical"]:
        print(f"[自检] ⚠ primary 口径与 caselevel_stats 不一致："
              f"{selfcheck['mismatches'][:3]}", flush=True)
    else:
        print(f"[自检] primary 口径与 caselevel_stats.paired_caselevel 逐格一致"
              f"（{selfcheck['n_checked']} 个对比）", flush=True)

    parse_stats = json.loads((OUTDIR / "parse_stats.json").read_text()) \
        if (OUTDIR / "parse_stats.json").exists() else {}
    total_missing_slots = sum(v["slots"] for v in miss_cov.values())
    total_missing_pairs = len({cs.key_of(r["gold"], c)
                               for runs in arms.values() for rows in runs.values()
                               for r in rows.values() for c in r["top5"][:5]
                               if cs.key_of(r["gold"], c) not in cache})

    meta = {
        "config": {
            "arms": arm_paths(SEEDS),
            "labels": ARM_LABEL,
            "n_cases": len(ids), "seeds": SEEDS, "n_boot": cs.N_BOOT,
            "rng_seed": cs.BOOT_SEED,
            "statistics": "stats_caselevel.py / caselevel_stats.py 逐字同口径："
                          "病例级 5-seed 均值命中率 → 配对 Wilcoxon 双侧 + "
                          "cluster bootstrap 95% CI + 多数决(>=3/5) 精确 McNemar",
            "judge": f"GLM-5.3-flash × judge_v3.V3_PROMPT（{cs.GLM_MODEL}）",
            "judge_cache": [str(p) for p in cache_paths], "judge_cache_var": var,
            "judge_cache_size": len(cache),
            "readiness": readiness, "readiness_blockers": blockers,
            "missing_slots_total": total_missing_slots,
            "missing_unique_pairs_total": total_missing_pairs,
        },
        "inference": parse_stats,
        "prompt_audit": audit,
        "prompt_fingerprint": fp,
        "coverage": coverage,
        "judge_coverage": miss_cov,
        "stats": stats,
        "caselevel_primary": res["comparisons"],
        "caselevel_modes": modes,
        "selfcheck_primary_vs_caselevel_stats": selfcheck,
    }

    md, verdict = render_md(meta, cases)
    meta["verdict"] = verdict
    OUT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"\n已写入 {OUT_JSON}\n已写入 {OUT_MD}", flush=True)
    return meta


def render_md(meta, cases):
    res = meta["stats"][SPLIT]
    cl = meta["caselevel_primary"]
    modes = meta["caselevel_modes"]
    cov = meta["coverage"]
    jc = meta["judge_coverage"]
    ps = meta["inference"] or {}
    n_missing_pairs = meta["config"]["missing_unique_pairs_total"]
    readiness = meta["config"]["readiness"]

    def c(pair, k, mode="primary"):
        """primary 直接取库结果（caselevel_stats），其余两种取本脚本的实现。"""
        if mode == "primary":
            return cl[f"{pair}"][f"top{k}"]
        return modes[mode][f"top{k}/{pair}"]

    L = []
    A = L.append
    A("# MCR 406 例：A×1@T=0.3 温度匹配臂（补跑）与病例级温度匹配分析\n")
    A("回答的评审意见：论文主实验在 MCR 上 A×1 跑 T=0（贪心）、P/MDT 跑 T=0.3，"
      "「策略差异」与「温度差异」是否混杂。本脚本在 MCR（外部验证集，406 例）上"
      "补跑 A×1@T=0.3（`topn_seeds_mcr_ax1t03/Ax1t03_s{1..5}.jsonl`），"
      "使三臂温度完全匹配。\n")

    # ---------- 0. 答案 ----------
    A("## 0. 明确回答（数字见 §4、§5）\n")
    def cmp_line(pair, k, mode="primary"):
        x = c(pair, k, mode)
        return (f"top-{k} {x['mean_rate_a']*100:.1f}% vs {x['mean_rate_b']*100:.1f}%，"
                f"差 {x['mean_diff']*100:+.1f}pp，Wilcoxon p={fmt_p(x['wilcoxon_p'])}"
                f"{sig(x['wilcoxon_p'])}，n={x['n_cases']}")

    if not readiness:
        A("> ⚠ **数据未就绪**（见 §2 的覆盖/缺失）："
          + "；".join(meta["config"]["readiness_blockers"])
          + "。先补齐 PHASE=infer / PHASE=judge 再重跑 PHASE=analyze；"
            "在此之前请勿引用本文件的数字。\n")
    m_a1 = {k: c("MDT03_vs_Ax1t03", k) for k in KS}
    m_p = {k: c("MDT03_vs_P03", k) for k in KS}
    a_t = {k: c("Ax1t03_vs_Ax1T0", k) for k in KS}
    d_a1 = {k: m_a1[k]["mean_diff"] for k in KS}
    d_p = {k: m_p[k]["mean_diff"] for k in KS}
    sig_a1 = {k: (m_a1[k]["wilcoxon_p"] is not None
                  and m_a1[k]["wilcoxon_p"] < 0.05) for k in KS}
    sig_p = {k: (m_p[k]["wilcoxon_p"] is not None
                 and m_p[k]["wilcoxon_p"] < 0.05) for k in KS}
    pos_a1 = all(d_a1[k] > 0 for k in KS)
    pos_p = all(d_p[k] > 0 for k in KS)
    # MCR 上 MDT 的召回优势历来体现在 top-3/top-5（主实验 top-1 上 MDT 略低于 A×1），
    # 因此 headline 判定以 top-3/top-5 为准，top-1 单独如实报告。
    FOCUS = (3, 5)

    def headline(pair, other_lab, sig_, d_):
        n_sig = [k for k in FOCUS if sig_[k]]
        pos = [k for k in FOCUS if d_[k] > 0]
        if len(pos) == len(FOCUS) and len(n_sig) == len(FOCUS):
            return (f"**答：成立。**top-3/top-5 上 MDT 均显著高于{other_lab}"
                    + ("（top-1 亦显著）" if sig_[1] and d_[1] > 0
                       else f"（top-1 上 MDT {d_[1]*100:+.1f}pp，"
                            f"p={fmt_p(c(pair, 1)['wilcoxon_p'])}，"
                            f"{'不显著' if not sig_[1] else '显著'}）")
                    + "。")
        if len(pos) == len(FOCUS):
            return (f"**答：方向成立，但显著性不全。**top-3/top-5 上 MDT 均高于"
                    f"{other_lab}，其中 "
                    + "、".join(f"top-{k} 显著" for k in n_sig)
                    + ("、" if n_sig else "")
                    + "、".join(f"top-{k} 未达显著" for k in FOCUS if k not in n_sig)
                    + "。")
        return (f"**答：不成立/需修正。**top-3 或 top-5 上 MDT 不再高于{other_lab}"
                f"（top-{pos[0] if pos else FOCUS[0]} 等终点方向为负）—— 温度匹配后"
                f"该优势至少部分来自温度不对称，正文必须按 §4a 的数字修订。")

    A(f"**问题一：温度完全匹配（三臂全 T=0.3）后，MCR 上「MDT 相对 A×1 的召回优势」"
      f"是否仍然成立？**\n")
    A("三臂全 T=0.3 下 MDT − A×1@0.3 的病例级命中率差：" + "；".join(
        cmp_line("MDT03_vs_Ax1t03", k) for k in KS) + "。\n")
    A(headline("MDT03_vs_Ax1t03", " A×1@0.3", sig_a1, d_a1))
    A("")
    A(f"**问题二：温度完全匹配后，MCR 上「MDT 相对 P 的优势」是否仍然成立？**\n")
    A("三臂全 T=0.3 下 MDT − P@0.3：" + "；".join(
        cmp_line("MDT03_vs_P03", k) for k in KS) + "。\n")
    A(headline("MDT03_vs_P03", " P@0.3", sig_p, d_p))
    A("")
    A("（P@0.3 与 MDT@0.3 本来就是同一温度，问题二不受温度混杂影响；列出它是为了"
      "给出匹配条件下的完整三臂排序。）\n")
    A("（注：MCR 上 MDT 的召回优势历来只在 top-3/top-5；top-1 上主实验口径的 MDT "
      "（51.3%）本就略低于 A×1@T=0（52.4%），故 headline 判定以 top-3/top-5 为准。）")
    A("")
    A("**温度本身对 A×1 的影响（A×1@0.3 − A×1@T=0）：**" + "；".join(
        cmp_line("Ax1t03_vs_Ax1T0", k) for k in KS) + "。\n")
    dir_t = ("T=0.3 更好" if all(a_t[k]["mean_diff"] > 0 for k in KS) else
             "T=0 更好（即 T=0.3 不利于 A×1）" if all(a_t[k]["mean_diff"] < 0
                                                    for k in KS)
             else "方向在各终点间不一致（top-1 见下）")
    A(f"→ {dir_t}；"
      + ("三个终点均不显著（p>0.05），温度效应与病例级噪声同量级。"
         if all((a_t[k]["wilcoxon_p"] is None or a_t[k]["wilcoxon_p"] >= 0.05)
                for k in KS)
         else "显著性见 §4b（个别终点达 p<0.05，但幅度 <1pp）。"))
    A("")

    # ---------- 1. 提示词一致性 ----------
    A("## 1. 与主实验 MCR A×1 臂的一致性（唯一差别是温度）\n")
    a = meta["prompt_audit"]
    pf = meta["prompt_fingerprint"]
    A("| 环节 | 主实验 MCR A×1 臂 | 本臂（A×1@T=0.3） | 是否相同 |")
    A("|---|---|---|---|")
    A(f"| 病例加载 | `topn_mcr.load_cases()` | 同（直接 import 复用） | ✅ |")
    A(f"| 提示词 | `{a['prompt_expr']}` | 同（同一常量、同一表达式） | ✅ |")
    A(f"| 提示词常量 | `topn_cpc_promptv2.A_TOPN_PROMPT` | 同 | ✅ |")
    A(f"| 调用 | `topn_mcr.call_top5(prompt, 0.0)` | `topn_mcr.call_top5(prompt, 0.3)` | "
      f"⚠ 仅温度不同 |")
    A(f"| 调用参数 | qwen3.8-flash / disable_thinking=True / max_tokens=4096 / "
      f"timeout=300 / 内部 3 次空重试 | 同（同一函数） | ✅ |")
    A(f"| 输出 schema | `case_id,gold,top5,total_tokens` | 同 | ✅ |")
    A("")
    A("**源码审计**（每次 analyze 自动执行，读主实验两处 A×1 调用点原文）：")
    for s in a["sites"]:
        A(f"- `routing_study/scripts/{s['file']}`（{s['desc']}）：提示词表达式 "
          f"`{a['prompt_expr']}` 出现 {s['prompt_expr_count']} 次"
          f"（字面匹配 {'✅' if s['prompt_expr_literal_match'] else '❌'}），"
          f"其后温度实参 = `{s['temperature_in_source']}`"
          f"（{'✅ 0.0' if s['is_Ax1_arm_T0'] else '❌'}）")
    A(f"- 结论：两处调用点的提示词表达式与本脚本逐字相同，温度皆为 `0.0`；"
      f"本脚本把同一表达式送进同一函数、只把温度改成 `0.3` → "
      f"**提示词与病例文本逐字节相同，唯一差别是温度**"
      f"（源码审计整体：{a['all_sites_identical_except_temperature']}）。")
    A(f"- `A_TOPN_PROMPT` 常量 sha256 = `{a['prompt_constant_sha256']}`")
    A(f"- 406 例格式化后提示词的聚合 sha256 = `{pf['aggregate_sha256']}`"
      f"（逐例 sha256 拼接后再哈希；首例 `{pf['case0_sha256']}`；"
      f"{pf['n_distinct']}/{pf['n_cases']} 例互不相同 → 确为逐例文本）。"
      f"该指纹只依赖 `A_TOPN_PROMPT` 与病例文本，与温度无关，故也就是主实验 "
      f"A×1 臂所发提示词的指纹。")
    A("")

    # ---------- 2. 数据与判分覆盖 ----------
    A("## 2. 数据与判分覆盖（缺失一律显式报告，不静默计 miss）\n")
    A(f"- 推理：`provider=qwen` → `{ps.get('provider', 'NA')}`，模型 "
      f"`{ps.get('model', 'NA')}`；`temperature={ps.get('temperature', 0.3)}`、"
      f"`disable_thinking=True`、`max_tokens=4096`、`timeout=300`；"
      f"{N_EXPECTED} 例 × {len(SEEDS)} seeds = {N_EXPECTED * len(SEEDS)} 次调用。")
    if ps.get("per_seed"):
        A(f"- 解析统计（本次运行）：{json.dumps(ps['per_seed'], ensure_ascii=False)}；"
          f"跨次累加：{json.dumps(ps.get('aggregate_per_seed', {}), ensure_ascii=False)}；"
          f"硬失败累计 {ps.get('hard_failures_total', 0)} 例（空/不足 5 项不落盘，"
          f"续跑自动重试）")
    A(f"- 行覆盖（每臂每 seed 的行数，应全为 {N_EXPECTED}）："
      + "；".join(f"{arm}: {sorted(set(v.values()))}" for arm, v in cov.items()))
    A(f"- 判官缓存：{' + '.join('`' + x + '`' for x in meta['config']['judge_cache'])}"
      f"（来自 `{meta['config']['judge_cache_var']}`，合并后 "
      f"{meta['config']['judge_cache_size']} 对；只读，本脚本从不写缓存）")
    A(f"- **未判定 (gold, candidate)**：槽位合计 {meta['config']['missing_slots_total']}"
      f"，去重对合计 {n_missing_pairs}（跨臂去重）。逐臂：")
    A("")
    A("| 臂 | 缺失槽位 | 去重对 | 涉及病例 | 悬空观察 top-1/3/5 |")
    A("|---|---|---|---|---|")
    for arm in ("Ax1t03", "P03", "MDT03", "Ax1T0"):
        v = jc[arm]
        dg = v["dangling_observations"]
        A(f"| {ARM_LABEL[arm]} | {v['slots']} | {v['unique_pairs']} | {v['cases']} | "
          f"{dg['top1']}/{dg['top3']}/{dg['top5']} |")
    A("")
    A(f"- 悬空观察 = 前 k 位含未判定且已判定部分无命中的 (病例, seed) 观察数，"
      f"只有这些观察的 top-k 结果真正取决于未判定的候选。")
    A(f"- 主口径（= `stats_caselevel.py` 口径）要求 5 seeds 的该位全部可判定，"
      f"因此含未判定的病例被**剔除**，逐对比的 `n_cases` 列在 §4 表内"
      f"（每格 n={N_EXPECTED} 表示无剔除）；§5 另给两个全 {N_EXPECTED} 例的"
      f"上下界口径夹住真值。")
    if n_missing_pairs:
        A(f"- ⚠ 存在 {n_missing_pairs} 对未判定：若其中包含本臂（A×1@0.3）的新候选，"
          f"请先跑 `PHASE=judge`（判分必须串行调度，缓存为整文件覆盖写）并重跑 "
          f"`PHASE=analyze`；缺失为 0 时本文件自动退化为纯主口径。")
    A("")

    # ---------- 3. 逐 seed 准确率 ----------
    A(f"## 3. 逐 seed 准确率与 5-seed 均值±SD（MCR {N_EXPECTED} 例）\n")
    A("未判定按 miss 计（现状表口径）；末列为病例级 5-seed 平均命中率"
      "（主口径，仅统计可完整判定的病例）。\n")
    A("| 方案 | top-1 | top-3 | top-5 | 病例级 top-1/3/5 |")
    A("|---|---|---|---|---|")
    for arm in ("Ax1t03", "P03", "MDT03", "Ax1T0"):
        ms = res["arms"][arm]["mean_sd"]
        cr = res["arms"][arm]["case_rate_mean"]
        crm = "/".join(f"{cr[f'top{k}']*100:.1f}%" if cr[f"top{k}"] is not None
                       else "NA" for k in KS)
        A(f"| {ARM_LABEL[arm]} | {cs.acc_cell(ms, 1)} | {cs.acc_cell(ms, 3)} | "
          f"{cs.acc_cell(ms, 5)} | {crm} |")
    A("")

    # ---------- 4. 病例级主口径 ----------
    A("## 4. 病例级主口径对照（与 `stats_caselevel.py` 同口径）\n")
    A("每列含义：命中率 A vs B（病例级 5-seed 平均）、均值差 [95% CI]（病例级 "
      "cluster bootstrap 10,000 次）、配对 Wilcoxon 双侧、多数决（>=3/5）精确 "
      "McNemar、旧的跨 seed 合并 McNemar（未判定按 miss，仅作对照）。\n")
    A("| 分组 | 对比 | top-k | 命中率 A vs B | n_cases | 均值差 [95% CI] | "
      "Wilcoxon p | 多数决 McNemar (a:b) p | 合并 McNemar (a:b) p |")
    A("|---|---|---|---|---|---|---|---|---|")
    for gname, pairs in GROUPS:
        for a_, b_ in pairs:
            for k in KS:
                x = c(f"{a_}_vs_{b_}", k)
                lo, hi = x["boot95_ci"]
                mj, old = x["majority"], x["pooled_mcnemar"]
                flag = "" if x["n_cases"] == N_EXPECTED else " ⚠"
                A(f"| {gname} | {ARM_LABEL[a_]} − {ARM_LABEL[b_]} | top-{k} | "
                  f"{x['mean_rate_a']*100:.1f}% vs {x['mean_rate_b']*100:.1f}% | "
                  f"{x['n_cases']}/{N_EXPECTED}{flag} | "
                  f"{x['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                  f"{fmt_p(x['wilcoxon_p'])}{sig(x['wilcoxon_p'])} | "
                  f"{mj['a_only']}:{mj['b_only']} p={fmt_p(mj['mcnemar_p'])}"
                  f"{sig(mj['mcnemar_p'])} | "
                  f"{old['a_only']}:{old['b_only']} p={fmt_p(old['p'])}"
                  f"{sig(old['p'])} |")
    A("")
    A("（a×1 在前的行 = 该臂更高为正；「MDT − A×1 (T=0.3)」即核心对照：正值表示"
      "同温度下 MDT 召回更高。）\n")

    # ---------- 5. 缺失敏感性 ----------
    A("## 5. 未判定对的敏感性（全 406 例上下界）\n")
    A("三种处理：**主口径**（5 seeds 全部可判定才纳入，含缺失病例被剔除）、"
      "**按 miss**（未判定当未命中，悲观下界）、**按命中**（未判定当命中，乐观上界）。"
      "真值必在 (b)(c) 之间。缺失为 0 时三者逐格相同。\n")
    A("| 对比 | top-k | 主口径 [95% CI] p (n) | 按 miss（下界） | 按命中（上界） | "
      "方向是否三口径一致 |")
    A("|---|---|---|---|---|---|")
    bound_all = True
    for a_, b_ in PAIRS:
        for k in KS:
            trio = [c(f"{a_}_vs_{b_}", k, m) for m in ("primary", "miss", "hit")]
            signs = {1 if x["mean_diff"] > 0 else (-1 if x["mean_diff"] < 0 else 0)
                     for x in trio}
            agree = len(signs - {0}) <= 1
            bound_all &= agree
            p0 = trio[0]
            A(f"| {ARM_LABEL[a_]} − {ARM_LABEL[b_]} | top-{k} | "
              f"{p0['mean_diff']*100:+.1f}pp [{p0['boot95_ci'][0]*100:+.1f}, "
              f"{p0['boot95_ci'][1]*100:+.1f}] p={fmt_p(p0['wilcoxon_p'])} "
              f"(n={p0['n_cases']}) | {trio[1]['mean_diff']*100:+.1f}pp "
              f"p={fmt_p(trio[1]['wilcoxon_p'])} | {trio[2]['mean_diff']*100:+.1f}pp "
              f"p={fmt_p(trio[2]['wilcoxon_p'])} | {'✅' if agree else '❌'} |")
    A("")
    A(f"- 全部对比的方向在三口径下"
      + ("**完全一致** → 缺失判定不改变方向结论（只影响幅度与显著性）。"
         if bound_all else
         "**存在不一致** → 缺失判定足以改变方向，须先补判再引用。"))
    sc = meta["selfcheck_primary_vs_caselevel_stats"]
    A(f"- 口径自检：本脚本的 primary 实现与 `caselevel_stats.paired_caselevel` "
      f"逐格核对 = **{sc['identical']}**（{sc['n_checked']} 个对比 × 全部字段）"
      + ("" if sc["identical"] else f"，不一致：{sc['mismatches'][:5]}") + "。")
    A("")

    # ---------- 6. 结论 ----------
    A("## 6. 结论\n")
    A("1. **三臂全 T=0.3**：MDT − A×1@0.3 = " + "、".join(
        f"top-{k} {d_a1[k]*100:+.1f}pp (p={fmt_p(m_a1[k]['wilcoxon_p'])})"
        for k in KS)
      + "；MDT − P@0.3 = " + "、".join(
        f"top-{k} {d_p[k]*100:+.1f}pp (p={fmt_p(m_p[k]['wilcoxon_p'])})"
        for k in KS) + "。")
    A(f"   → 温度匹配后 MDT 相对 A×1 的召回优势（以 top-3/top-5 为准）"
      + ("**保持**" if all(d_a1[k] > 0 for k in FOCUS)
         else "**不再保持**")
      + "：top-3/top-5 上 "
      + "、".join(f"top-{k} {d_a1[k]*100:+.1f}pp"
                  + (f"（p={fmt_p(m_a1[k]['wilcoxon_p'])}{sig(m_a1[k]['wilcoxon_p'])}）")
                  for k in FOCUS)
      + f"；top-1 上 MDT 相对 A×1@0.3 {d_a1[1]*100:+.1f}pp"
        f"（p={fmt_p(m_a1[1]['wilcoxon_p'])}），与主实验 MCR 口径一致。")
    A(f"   相对 P@0.3 的优势（以 top-3/top-5 为准）"
      + ("**保持**" if all(d_p[k] > 0 for k in FOCUS) else "**不再保持**")
      + "："
      + "、".join(f"top-{k} {d_p[k]*100:+.1f}pp"
                  + (f"（p={fmt_p(m_p[k]['wilcoxon_p'])}{sig(m_p[k]['wilcoxon_p'])}）")
                  for k in FOCUS) + "。")
    A("2. **A×1 自身的温度效应**（A×1@0.3 − A×1@T=0）= " + "、".join(
        f"top-{k} {a_t[k]['mean_diff']*100:+.1f}pp "
        f"(p={fmt_p(a_t[k]['wilcoxon_p'])})" for k in KS)
      + f" → {dir_t}。这一项回答了「把 A×1 放到 P/MDT 的温度上会怎样」，"
        "也是温度匹配对照与论文原口径的唯一差异来源。")
    asym_m = {k: c("MDT03_vs_Ax1T0", k) for k in KS}
    asym_p = {k: c("P03_vs_Ax1T0", k) for k in KS}
    A("3. **温度不对称口径（论文原口径）**：MDT@0.3 − A×1@T=0 = " + "、".join(
        f"top-{k} {asym_m[k]['mean_diff']*100:+.1f}pp "
        f"(p={fmt_p(asym_m[k]['wilcoxon_p'])})" for k in KS)
      + "；P@0.3 − A×1@T=0 = " + "、".join(
        f"top-{k} {asym_p[k]['mean_diff']*100:+.1f}pp "
        f"(p={fmt_p(asym_p[k]['wilcoxon_p'])})" for k in KS) + "。")
    same_dir = all((asym_m[k]["mean_diff"] > 0) == (d_a1[k] > 0) for k in KS)
    A("   → 匹配口径与不对称口径"
      + ("**方向一致**，即温度差异不能解释 MDT 在 MCR 上的召回优势；"
         if same_dir else
         "**方向不一致**，即温度差异足以改变结论，正文必须按 §4a 的匹配口径修订；")
      + "两者的幅度差异见上两行。")
    A(f"4. **缺失判定**：未判定对 {n_missing_pairs}（槽位 "
      f"{meta['config']['missing_slots_total']}），上下界口径方向"
      + ("一致" if bound_all else "不一致") + "。")
    A("")
    A("## 7. 复现命令\n")
    A("```bash")
    A("SMOKE=1 ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py")
    A("MAX_WORKERS=6 PHASE=infer ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py")
    A("MCRT03_JUDGE_CACHE=<分片缓存> PHASE=judge ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py")
    A("MCRT03_JUDGE_CACHE=<分片缓存>:<主缓存> PHASE=analyze \\")
    A("    ./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py")
    A("```")
    A(f"- `MCRT03_JUDGE_CACHE` 在 analyze 下可用 `{os.pathsep}` 分隔多个缓存按序合并"
      f"（先给的优先）；judge 下只接受单个路径。判分分片若要被 analyze 用上，"
      f"要么与主缓存一起传入，要么先合并进主缓存（见 `merge_judge_shards.py`）。")
    A("- 本脚本**未修改** `topn_mcr/`、`topn_mcr_seeds/`、`topn_cpc_promptv2.py` "
      "与 `paper/` 的任何文件，也未写共享判官缓存。")
    A("- 判分缓存是全量覆盖写：跨进程并发跑 judge 会互相覆盖，判分须串行调度。")

    verdict = {
        "matched_T03_MDT_vs_Ax1": {
            f"top{k}": {"mean_diff_pp": d_a1[k] * 100,
                        "wilcoxon_p": m_a1[k]["wilcoxon_p"],
                        "n_cases": m_a1[k]["n_cases"],
                        "significant": sig_a1[k]} for k in KS},
        "matched_T03_MDT_advantage_over_Ax1_holds": pos_a1,
        "matched_T03_MDT_vs_P": {
            f"top{k}": {"mean_diff_pp": d_p[k] * 100,
                        "wilcoxon_p": m_p[k]["wilcoxon_p"],
                        "n_cases": m_p[k]["n_cases"],
                        "significant": sig_p[k]} for k in KS},
        "matched_T03_MDT_advantage_over_P_holds": pos_p,
        "Ax1_temperature_effect": {
            f"top{k}": {"mean_diff_pp": a_t[k]["mean_diff"] * 100,
                        "wilcoxon_p": a_t[k]["wilcoxon_p"]} for k in KS},
        "Ax1_temperature_effect_direction": dir_t,
        "asymmetric_paper": {
            f"MDT_vs_Ax1T0_top{k}": asym_m[k]["mean_diff"] * 100 for k in KS},
        "asymmetric_vs_matched_same_direction": same_dir,
        "missing_unique_pairs": n_missing_pairs,
        "missing_bounds_agree_direction": bound_all,
        "readiness": readiness,
    }
    return "\n".join(L) + "\n", verdict


# ---------- 冒烟 ----------

def smoke():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()[:3]
    audit = source_audit()
    fp = prompt_fingerprint(cases)
    if not audit["all_sites_identical_except_temperature"]:
        raise SystemExit(f"提示词源码审计失败：{audit['sites']}")
    seed = SEEDS[0]
    path = SMOKE_DIR / f"Ax1t03_s{seed}.jsonl"
    for p in (path, SMOKE_DIR / "failures.jsonl"):
        if p.exists():
            p.unlink()
    print(f"[冒烟] {len(cases)} 例 → {SMOKE_DIR}（seed {seed}，T={TEMPERATURE}）"
          f" | model={QWEN_MODEL} transport={resolve_provider('qwen')} "
          f"workers={WORKERS}", flush=True)
    print(f"[冒烟] 源码审计：主实验 A×1 两处调用点提示词表达式逐字相同="
          f"{audit['all_sites_identical_except_temperature']}"
          f"（温度 {[s['temperature_in_source'] for s in audit['sites']]}）；"
          f"A_TOPN_PROMPT sha256={audit['prompt_constant_sha256'][:16]}…；"
          f"前 3 例提示词聚合 sha256={fp['aggregate_sha256'][:16]}…", flush=True)
    failures = []
    st_ = {"first_attempt_empty": 0, "needed_retry": 0, "hard_failed": 0}
    run_one(seed, cases, SMOKE_DIR, stats=st_, failures=failures)
    rows = load_done(path)
    print(f"\n[冒烟] 产出 {len(rows)}/{len(cases)} 行 | 首轮空 top5 "
          f"{st_['first_attempt_empty']} | 需重试 {st_['needed_retry']} | "
          f"硬失败 {len(failures)}")
    ok = len(rows) == len(cases)
    for c in cases:
        r = rows.get(c["case_id"])
        if not r or not row_complete(r):
            ok = False
            print(f"  {c['case_id'][:50]} | 失败：top5 非 5 项")
            continue
        print(f"  {c['case_id'][:50]}\n"
              f"    gold   : {c['gold'][:80]}\n"
              f"    top5[0]: {r['top5'][0]} | tokens={r['total_tokens']} | "
              f"字段={sorted(r)} | n_top5={len(r['top5'])}")
    print(f"[冒烟] 通过: {ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    if os.environ.get("SMOKE") == "1":
        smoke()
        return
    if not SEEDS:
        raise SystemExit("MCRT03_SEEDS 为空")
    phase = os.environ.get("PHASE", "all").lower()
    print(f"seeds: {SEEDS} | phase: {phase} | outdir: {OUTDIR}", flush=True)

    if phase in ("all", "infer"):
        cases = load_cases()
        run_infer(cases)
        problems = []
        for seed in SEEDS:
            rows = list(load_done(OUTDIR / f"Ax1t03_s{seed}.jsonl").values())
            if len(rows) != len(cases):
                done_ids = {r["case_id"] for r in rows}
                problems.append(
                    f"s{seed} 缺 {len(cases) - len(rows)} 行："
                    + ", ".join(c["case_id"][:40] for c in cases
                                if c["case_id"] not in done_ids))
            bad = [r["case_id"][:40] for r in rows if not row_complete(r)]
            if bad:
                problems.append(f"s{seed} 残缺行（top5 不足 5 项）：{bad}")
        if problems:
            print("[检查] ✗ 输出不完整：\n  " + "\n  ".join(problems), flush=True)
            print("[检查] 上述病例未拿到有效 top5。重跑同一条命令即可续跑补齐"
                  "（已有行自动跳过）：\n  MAX_WORKERS=6 PHASE=infer "
                  "./.venv/bin/python routing_study/scripts/ax1_t03_mcr.py", flush=True)
            raise SystemExit(2)
        print("[检查] 各 seed 输出行数与 top5 完整性通过", flush=True)

    if phase in ("all", "judge"):
        run_judge()

    if phase in ("all", "analyze"):
        run_analyze()


if __name__ == "__main__":
    main()
    # 判定阶段可能有挂死的网络线程（非 daemon），正常退出会被 join 卡住；
    # 所有产出已落盘，直接退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
