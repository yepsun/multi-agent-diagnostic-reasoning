#!/usr/bin/env python3
"""P 臂「字面双花括号」提示词缺陷的修正版（Pfixed）：CPC 87 例 × 5 seeds。

背景（需向审稿人交代的提示词转写缺陷）
------------------------------------
主实验 P 臂的提示词是 `PERSPECTIVE_PROMPT.format(structured_case=...) +
P_TOPN_SUFFIX`。`P_TOPN_SUFFIX`（`topn_cpc_promptv2.py`）是一段**从不经过
`.format()`** 的普通字符串，其 JSON 块写作字面双花括号
`{{"top5": [{{"rank": 1, ...}}]}}`，因此**实际发出**的 P 提示词里就是双花括号
（逐字节文本见 `routing_study/results/prompt_actually_sent.md`；对照 A×1 用的
`A_TOPN_PROMPT` 本身要经 `.format()`，双花括号被折叠成合法 JSON 示例，故 A×1
不受影响）。qwen3.8-flash 会自行把它改写成单花括号（主实验 2075 行输出无一
畸形），deepseek-flash 则原样照抄 → `extract_json` 解析失败（第二模型族 P 臂
约 7% 的行解析失败）。本脚本量化该缺陷对主实验结论的影响。

替换规则（只在本脚本内做，不改共享常量）
--------------------------------------
    P_FIXED_SUFFIX = P_TOPN_SUFFIX.replace("{{", "{").replace("}}", "}")

即把成对转义残留折叠为单花括号。约束与依据：
- **不修改 `topn_cpc_promptv2.py`**（改它会污染主实验口径，且其他脚本共享该
  常量）；归一化只作用于本脚本进程内的常量副本。
- `P_TOPN_SUFFIX` 中所有花括号都是**成对转义**（`{`、`}` 的极大游程长度均为
  偶数），折叠后无残留 `{{`/`}}`，且折叠后的 JSON 示例块可被 `json.loads`
  解析 —— 模块级 `self_check()` 每次运行都会断言这两点。
- 因此 Pfixed 与主实验 P 臂的提示词**唯一差别是转义花括号被正确折叠**：拼接
  方式、`PERSPECTIVE_PROMPT`、病例文本、其余每一个字符都逐字节相同。该命题不是
  声明而是被逐例审计：`prompt_audit()` 对 87 例逐一比对两条完整提示词，断言
  二者行数相同、**恰有 1 行不同**、且该行的差异恰为花括号折叠；`analyze` 阶段
  会把它写进 JSON 与 md。

与主实验完全一致的推理参数
--------------------------
同模型（`provider="qwen"` → DashScope API，`QWEN_MODEL` = qwen3.8-flash）、
`disable_thinking=True`、`temperature=0.3`、`max_tokens=2048`、`timeout=300`：
直接复用 `topn_cpc.call_top5(prompt, temperature=0.3)`（其内部参数即
`topn_cpc.py` 的主实验调用点，逐字相同）。`QWEN_MODEL` / `QWEN_BACKEND` 一律
不在本脚本里覆盖，沿用仓库 `.env` 的解析结果，保证与主实验同一运行时。

输出
----
`routing_study/results/topn_seeds_pfixed/Pfixed_s{1..5}.jsonl`，行 schema 与
主实验 P 臂一致（`case_id` / `gold` / `top5` / `total_tokens`）；断点续跑
（缺行或不足 5 项的行补齐）；单例空 top5 / 解析失败重试 3 次，仍失败记入
`failures.jsonl`，不落残缺行。另写 `parse_stats.json`（逐 seed 的首轮空输出
数与硬失败数）与 `failures.jsonl`，不改变行 schema。

阶段（PHASE=infer|judge|analyze，默认 all）
- infer：5 seeds × 87 例 = 435 次调用。
- judge：GLM-5.3-flash × `judge_v3.V3_PROMPT`，共享缓存
  `results/judge_cache_glm_v3.json`（键 `gold[:150]+"||"+cand[:150]`），
  **只补缺失对**，写法复用 `caselevel_stats.judge_missing`（6 并发 / 150s 超时 /
  最多 3 轮）。
- analyze：病例级主口径（同 `stats_caselevel.py` / `caselevel_stats.py`）：每例
  5-seed 命中率 → 跨病例配对 Wilcoxon 双侧 + case-level cluster bootstrap
  10,000 次 95% CI + 多数决（>=3/5）精确 McNemar。对比 Pfixed vs 主实验 P、
  Pfixed vs A×1、Pfixed vs MDT，并附主实验 P vs A×1 / vs MDT 作同口径对照，
  在 full87 / held-out 46（`data/mgh_qa_dataset_new_cases.json`）/ dev 41
  （`data/mgh_qa_dataset.json`）三个病例集上各重算。产出
  `results/p_fixed_cpc.json` + `.md`。

环境变量：PFIXED_OUTDIR（默认 results/topn_seeds_pfixed）、PFIXED_SEEDS
（默认 "1,2,3,4,5"）、MAX_WORKERS（默认 4，同主实验）。SMOKE=1 只跑前 3 例到
`/tmp/pfixed_smoke`（可用 SMOKE_DIR 覆盖），不判定、不分析。
"""
import difflib
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

import caselevel_stats as cs  # noqa: E402
from topn_cpc import load_done, append_row, call_top5, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import P_TOPN_SUFFIX  # noqa: E402  主实验原常量（只读）
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from run_inference import resolve_provider, QWEN_MODEL  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("PFIXED_OUTDIR", RESULTS / "topn_seeds_pfixed"))
MAIN_SEEDS_DIR = RESULTS / "topn_seeds"
MDT_DIR = RESULTS / "topn_mdt"
OUT_JSON = RESULTS / "p_fixed_cpc.json"
OUT_MD = RESULTS / "p_fixed_cpc.md"
SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/pfixed_smoke"))
SEEDS = [int(s) for s in
         os.environ.get("PFIXED_SEEDS", "1,2,3,4,5").replace(",", " ").split()]
TEMPERATURE = 0.3
MAX_ATTEMPTS = 3
ARM_LABEL = {
    "Pfixed": "Pfixed（转义花括号已修正）",
    "P": "P（主实验，提示词含字面双花括号）",
    "Ax1": "A×1（T=0，作参照）",
    "MDT": "MDT（T=0.3，作参照）",
}
PAIRS = [("Pfixed", "P"), ("Pfixed", "Ax1"), ("Pfixed", "MDT"),
         ("P", "Ax1"), ("P", "MDT")]


def fold_braces(s):
    """把成对转义花括号 `{{`/`}}` 折叠为 `{`/`}`（提示词缺陷的唯一修正）。"""
    return s.replace("{{", "{").replace("}}", "}")


P_FIXED_SUFFIX = fold_braces(P_TOPN_SUFFIX)
SUFFIX_SHA_MAIN = hashlib.sha256(P_TOPN_SUFFIX.encode("utf-8")).hexdigest()
SUFFIX_SHA_FIXED = hashlib.sha256(P_FIXED_SUFFIX.encode("utf-8")).hexdigest()


def build_prompt(case_text, fixed=True):
    """主实验 P 臂的拼接方式逐字不变，只换后缀常量。"""
    return (PERSPECTIVE_PROMPT.format(structured_case=case_text)
            + (P_FIXED_SUFFIX if fixed else P_TOPN_SUFFIX))


def self_check():
    """断言折叠合法且是"唯一差别"：折叠后无残留双花括号、JSON 示例可解析。"""
    assert "{{" not in P_FIXED_SUFFIX and "}}" not in P_FIXED_SUFFIX, \
        "折叠后仍有双花括号残留"
    for runs in (re.findall(r"\{+", P_TOPN_SUFFIX)
                 + re.findall(r"\}+", P_TOPN_SUFFIX)):
        assert len(runs) % 2 == 0, f"原后缀存在非成对花括号: {runs!r}"
    js = re.search(r"```json\n(.*?)\n```", P_FIXED_SUFFIX, re.S).group(1)
    example = json.loads(js)
    assert len(example["top5"]) == 5, "JSON 示例不是 5 项"
    assert P_FIXED_SUFFIX == fold_braces(P_TOPN_SUFFIX)
    print(f"[自检] 后缀折叠合法：JSON 示例可解析（{len(example['top5'])} 项）；"
          f"P_TOPN_SUFFIX sha256={SUFFIX_SHA_MAIN[:16]}… → "
          f"P_FIXED_SUFFIX sha256={SUFFIX_SHA_FIXED[:16]}…", flush=True)
    return example


def prompt_audit(cases, sample_n=3):
    """逐例审计：两条完整提示词行数相同、恰有 1 行不同、差异恰为花括号折叠。"""
    bad, samples, sha_main, sha_fixed = [], [], [], []
    for c in cases:
        orig = build_prompt(c["text"], fixed=False)
        fixed = build_prompt(c["text"], fixed=True)
        lo, lf = orig.splitlines(), fixed.splitlines()
        sha_main.append(hashlib.sha256(orig.encode()).hexdigest())
        sha_fixed.append(hashlib.sha256(fixed.encode()).hexdigest())
        idx = [i for i, (a, b) in enumerate(zip(lo, lf)) if a != b]
        if len(lo) != len(lf) or len(idx) != 1 or fold_braces(lo[idx[0]]) != lf[idx[0]]:
            bad.append({"case_id": c["case_id"], "n_diff_lines": len(idx),
                        "same_linecount": len(lo) == len(lf)})
            continue
        if len(samples) < sample_n:
            i = idx[0]
            samples.append({
                "case_id": c["case_id"], "line_no": i + 1,
                "main": lo[i], "fixed": lf[i],
                "char_delta": len(lf[i]) - len(lo[i])})
    c0 = cases[0]
    diff = list(difflib.unified_diff(
        build_prompt(c0["text"], False).splitlines(),
        build_prompt(c0["text"], True).splitlines(),
        fromfile="P arm (main, literal {{ }})",
        tofile="Pfixed (braces folded)", lineterm="", n=0))
    return {
        "n_cases": len(cases),
        "n_single_line_diff": len(cases) - len(bad),
        "n_unexpected": len(bad),
        "unexpected": bad[:5],
        "suffix_sha256_main": SUFFIX_SHA_MAIN,
        "suffix_sha256_fixed": SUFFIX_SHA_FIXED,
        "prompt_sha256_main_case0": sha_main[0] if sha_main else None,
        "prompt_sha256_fixed_case0": sha_fixed[0] if sha_fixed else None,
        "prompt_sha256_main_all": hashlib.sha256("".join(sha_main).encode()).hexdigest(),
        "prompt_sha256_fixed_all": hashlib.sha256("".join(sha_fixed).encode()).hexdigest(),
        "per_case_char_delta": (len(build_prompt(c0["text"], True))
                                - len(build_prompt(c0["text"], False))),
        "sample_changed_lines": samples,
        "unified_diff_case0_nocontext": diff,
    }


# ---------- 阶段 1：推理 ----------

def row_complete(row):
    return bool(row) and len(row.get("top5") or []) == 5


def compact(path):
    """补采会追加多行；按 case_id 去重保留最后一行（顺序按首次出现），原子写回。"""
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
    path = outdir / f"Pfixed_s{seed}.jsonl"
    done = load_done(path)
    todo = [c for c in cases if not row_complete(done.get(c["case_id"]))]
    print(f"[Pfixed s{seed}] 已完成 {len(cases) - len(todo)}/{len(cases)}，"
          f"待跑 {len(todo)}", flush=True)

    def work(c):
        prompt = build_prompt(c["text"], fixed=True)
        empty_first = False
        last_err = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                top5, tokens = call_top5(prompt, temperature=TEMPERATURE)
            except Exception as e:  # 网络/解析异常一律重试
                top5, tokens, last_err = [], 0, f"{type(e).__name__}: {e}"
            if top5:
                return ({"case_id": c["case_id"], "gold": c["gold"],
                         "top5": top5, "total_tokens": tokens},
                        empty_first, attempt + 1)
            empty_first = True
            last_err = last_err or "empty top5 after parse"
            print(f"[Pfixed s{seed}] {c['case_id'][:40]} 空 top5/解析失败，"
                  f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续 {MAX_ATTEMPTS} 次无有效 top5"
                           f"（末次：{last_err}）")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                row, empty_first, attempts = fut.result()
            except Exception as e:
                (failures if failures is not None else []).append(
                    {"seed": seed, "case_id": c["case_id"], "error": str(e)})
                if stats is not None:
                    # 硬失败的病例同样计入重试统计（此前会漏掉，低估空输出率）
                    stats["first_attempt_empty"] += 1
                    stats["needed_retry"] += 1
                    stats["hard_failed"] += 1
                print(f"[失败] Pfixed s{seed} {c['case_id'][:40]}: {e}", flush=True)
                continue
            append_row(path, row)
            if stats is not None:
                if empty_first:
                    stats["first_attempt_empty"] += 1
                if attempts > 1:
                    stats["needed_retry"] += 1
            if i % 20 == 0 or i == len(todo):
                print(f"[Pfixed s{seed}] {i}/{len(todo)}", flush=True)
    compact(path)
    return path


def load_failures(path):
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


RETRY_LINE = re.compile(
    r"\[Pfixed s(\d+)\] .*空 top5/解析失败，重试 (\d+)/(\d+)")


def log_accounting(logs):
    """从运行日志统计「失败尝试」次数（尝试级记录只在日志里）。

    硬失败的病例不落行，其尝试次数只能从日志还原，故此处把日志当作尝试级账本，
    与 parse_stats 的病例级计数互为补充。"""
    per_log, total = {}, 0
    for p in logs:
        p = Path(p)
        if not p.exists():
            continue
        by_seed = {}
        for line in open(p, errors="replace"):
            m = RETRY_LINE.search(line)
            if m:
                by_seed[f"s{int(m.group(1))}"] = by_seed.get(f"s{int(m.group(1))}", 0) + 1
        if by_seed:
            per_log[p.name] = {"failed_attempts_by_seed": by_seed,
                               "failed_attempts_total": sum(by_seed.values())}
            total += sum(by_seed.values())
    return {"logs": per_log, "failed_attempts_total": total,
            "note": "failed_attempts = 返回空/解析失败的调用次数（=重试次数）；"
                    "病例级计数见 per_seed"}


def run_infer(cases, outdir=None):
    outdir = Path(outdir or OUTDIR)
    print(f"[推理] provider=qwen（解析后 transport={resolve_provider('qwen')}）"
          f" model={QWEN_MODEL} | T={TEMPERATURE} disable_thinking=True "
          f"max_tokens=2048（复用 topn_cpc.call_top5）", flush=True)
    per_seed, failures = {}, []
    for seed in SEEDS:
        s = {"first_attempt_empty": 0, "needed_retry": 0, "hard_failed": 0}
        run_one(seed, cases, outdir, stats=s, failures=failures)
        per_seed[f"s{seed}"] = s
    # 硬失败记录跨次续跑累积（补齐成功后旧记录仍保留，作为历史）
    fpath = outdir / "failures.jsonl"
    history = load_failures(fpath)
    history.extend(failures)
    if history:
        fpath.write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in history) + "\n",
            encoding="utf-8")
    if failures:
        print(f"[推理] 硬失败 {len(failures)} 例（连续 {MAX_ATTEMPTS} 次无有效 "
              f"top5，未落盘）："
              + "；".join(f"{f['seed']}:{f['case_id'][:36]}" for f in failures),
              flush=True)
    stats = {
        "model": QWEN_MODEL, "provider": resolve_provider("qwen"),
        "temperature": TEMPERATURE, "disable_thinking": True,
        "max_tokens": 2048, "n_cases": len(cases), "seeds": SEEDS,
        "n_calls_planned": len(cases) * len(SEEDS),
        "per_seed": per_seed, "hard_failures": failures,
        "note": "first_attempt_empty = 首轮 top5 为空/解析失败的行数；needed_retry = "
                "需要重试才成功的行数；hard_failed = 连续重试后仍未落盘的病例数",
    }
    # parse_stats.json 跨次续跑累积：runs 保留每次运行的病例级计数，
    # aggregate_per_seed 为累加值，log_accounting 为尝试级账本（含未落盘的尝试）。
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
                     "hard_failures": len(prev.get("hard_failures") or [])})
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
    logs = sorted(RESULTS.glob("pfixed_infer*.log"))
    stats.update({"runs": runs, "aggregate_per_seed": agg,
                  "hard_failures_total": sum(len(r.get("hard_failures") or [])
                                             for r in runs),
                  "attempt_accounting": log_accounting(logs),
                  "counter_note": "per_seed/aggregate_per_seed 是病例级计数，"
                                  "只在运行结束时写入；首轮运行（2026-09-19 "
                                  "08:53–09:43，日志 pfixed_infer.log）的计数早于"
                                  "累积格式，被续跑覆盖，其等价证据保留在 "
                                  "attempt_accounting（逐 seed 失败尝试数，14 次）"
                                  "与 failures.jsonl 中。要完整病例级账本请以 "
                                  "attempt_accounting 为准。"})
    pstats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[推理] parse_stats（本次）: {json.dumps(per_seed, ensure_ascii=False)}"
          f" | 硬失败 {len(failures)}", flush=True)
    print(f"[推理] 尝试级账本（{len(logs)} 个日志）: "
          f"{json.dumps(stats['attempt_accounting']['failed_attempts_total'])} 次"
          f"失败尝试", flush=True)
    return stats


# ---------- 阶段 2：判分（GLM v3，只补缺失对） ----------

def load_arm(scheme, seed, outdir=None):
    return cs.load(Path(outdir or OUTDIR) / f"{scheme}_s{seed}.jsonl")


def mdt_rows(seed, base=None):
    base = Path(base or MDT_DIR)
    return cs.load(base / ("synthesis.jsonl" if seed == 1
                           else f"s{seed}/synthesis.jsonl"))


def run_judge(seeds=None):
    seeds = seeds or SEEDS
    rows = []
    for seed in seeds:
        rows.extend(load_arm("Pfixed", seed).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text())) if cs.GLM_CACHE.exists() else 0
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}"
          f"（新增 {len(cache) - n_before} 对）", flush=True)
    return cache


# ---------- 阶段 3：分析（病例级主口径） ----------

def build_arms(seeds, outdir=None):
    return {
        "Pfixed": {s: load_arm("Pfixed", s, outdir) for s in seeds},
        "P": {s: cs.load(MAIN_SEEDS_DIR / f"P_s{s}.jsonl") for s in seeds},
        "Ax1": {s: cs.load(MAIN_SEEDS_DIR / f"Ax1_s{s}.jsonl") for s in seeds},
        "MDT": {s: mdt_rows(s) for s in seeds},
    }


def cmp_consistency(fix_c, main_c, tol=0.02):
    """同一对比在 Pfixed 口径与主实验 P 口径下的方向/效应量/显著性是否一致。"""
    if (not fix_c or not main_c or fix_c["mean_rate_a"] is None
            or main_c["mean_rate_a"] is None):
        return None
    d_fix, d_main = fix_c["mean_diff"], main_c["mean_diff"]
    return {
        "mean_diff_fixed": d_fix,
        "mean_diff_main": d_main,
        "direction_same": (d_fix > 0) == (d_main > 0),
        "effect_delta_pp": (d_fix - d_main) * 100,
        "effect_within_tol": abs(d_fix - d_main) <= tol,
        "wilcoxon_p_fixed": fix_c["wilcoxon_p"],
        "wilcoxon_p_main": main_c["wilcoxon_p"],
        "wilcoxon_sig_same": ((fix_c["wilcoxon_p"] or 1) < 0.05)
                             == ((main_c["wilcoxon_p"] or 1) < 0.05),
        "majority_p_fixed": fix_c["majority"]["mcnemar_p"],
        "majority_p_main": main_c["majority"]["mcnemar_p"],
        "majority_sig_same": (fix_c["majority"]["mcnemar_p"] < 0.05)
                             == (main_c["majority"]["mcnemar_p"] < 0.05),
    }


def consistency_table(stats, splits_order, n_expected, opps=("Ax1", "MDT")):
    """同一对比在两条口径下的一致性；n_cases 与预期不符时标 aligned=False。

    两条口径的样本必须落在同一批病例上（否则"方向/效应量"的差异只是样本差），
    故要求两侧 n_cases 都等于该病例集规模且彼此相等。"""
    out = {}
    for split in splits_order:
        res = stats.get(split) or {}
        n_exp = n_expected.get(split)
        out[split] = {}
        for opp in opps:
            key_f, key_m = f"Pfixed_vs_{opp}", f"P_vs_{opp}"
            entry = {}
            for k in cs.METHOD_KEYS:
                c = cmp_consistency(
                    (res.get("comparisons", {}).get(key_f) or {}).get(f"top{k}"),
                    (res.get("comparisons", {}).get(key_m) or {}).get(f"top{k}"))
                if c:
                    nf = (res["comparisons"][key_f][f"top{k}"] or {}).get("n_cases")
                    nm = (res["comparisons"][key_m][f"top{k}"] or {}).get("n_cases")
                    c.update({"n_cases_fixed": nf, "n_cases_main": nm,
                              "n_cases_expected": n_exp,
                              "aligned": None not in (nf, nm, n_exp)
                              and nf == nm == n_exp})
                entry[f"top{k}"] = c
            out[split][opp] = entry
    return out


def seed_spread(runs, ids, k):
    """某方案的逐 seed top-k 准确率 → (mean, sd, min, max, n_seeds)。"""
    keys = {s: "top5" for s in runs}
    per = cs.per_seed_metrics(runs, ids, keys)
    acc = [p[f"top{k}_acc"] for p in per if p[f"top{k}_acc"] is not None]
    if not acc:
        return None
    return {"mean": st.mean(acc), "sd": st.stdev(acc) if len(acc) > 1 else 0.0,
            "min": min(acc), "max": max(acc), "n_seeds": len(acc)}


def render_md(meta, stats, cons, audit, splits_order):
    L = []
    L.append("# Pfixed：P 臂「字面双花括号」提示词缺陷的修正版对照\n")
    L.append("主实验 P 臂实际发出的提示词里，JSON 块是**字面双花括号**"
             "（`{{\"top5\": [{{\"rank\": 1, ...}}]}}`），因为 "
             "`topn_cpc_promptv2.P_TOPN_SUFFIX` 是普通字符串、从不经过 "
             "`.format()`（对照 A×1 的 `A_TOPN_PROMPT` 经 `.format()`，折叠成合法 "
             "JSON 示例）。qwen3.8-flash 会自愈，deepseek-flash 会照抄 → 第二模型族 "
             "P 臂约 7% 的行解析失败。本节量化该缺陷对主实验（qwen）结论的影响："
             "跑一份**只折叠花括号**的 P 臂（Pfixed），其余与主实验逐字节相同。\n")

    L.append("## 1. 替换规则与「唯一差别」证明\n")
    L.append("替换规则（只在本脚本进程内，未修改 `topn_cpc_promptv2.py`）：\n")
    L.append("```python\nP_FIXED_SUFFIX = P_TOPN_SUFFIX.replace(\"{{\", \"{\")"
             ".replace(\"}}\", \"}\")\n```\n")
    L.append(f"- `P_TOPN_SUFFIX` sha256 = `{audit['suffix_sha256_main']}`"
             f"（与 `results/prompt_actually_sent.md` 记录的指纹一致）；"
             f"`P_FIXED_SUFFIX` sha256 = `{audit['suffix_sha256_fixed']}`")
    L.append(f"- 原后缀中所有花括号均为**成对转义**（`{{`/`}}` 游程长度全为偶数），"
             f"折叠后无残留双花括号，折叠后的 JSON 示例可被 `json.loads` 解析"
             f"（每次运行由 `self_check()` 断言）。")
    L.append(f"- **逐例审计（{audit['n_cases']} 例）**：两条完整提示词行数相同、"
             f"恰有 1 行不同、且该行的差异恰为花括号折叠 → "
             f"一致 {audit['n_single_line_diff']}/{audit['n_cases']} 例，"
             f"异常 {audit['n_unexpected']} 例。每例字符数差 "
             f"{audit['per_case_char_delta']}（正好是 6 对 `{{{{`→`{{`、`}}}}`→`}}`）。")
    L.append(f"- 完整提示词 sha256：主实验 "
             f"`{str(audit['prompt_sha256_main_case0'])[:16]}…` / Pfixed "
             f"`{str(audit['prompt_sha256_fixed_case0'])[:16]}…`（首例）")
    if audit["unified_diff_case0_nocontext"]:
        L.append("\n首例提示词的统一 diff（`-c 0`，仅此一处）：\n")
        L.append("```diff")
        L.extend(audit["unified_diff_case0_nocontext"])
        L.append("```")
    L.append("")

    L.append("## 2. 方案与数据\n")
    L.append(f"- 推理：`provider=qwen` → `{meta['provider']}`，模型 `"
             f"{meta['model']}`；`temperature={meta['temperature']}`、"
             f"`disable_thinking=True`、`max_tokens=2048`、`timeout=300` —— 直接复用 "
             f"`topn_cpc.call_top5`，与主实验 `seeds_87.py` 的 P 臂调用点逐字相同")
    L.append(f"- 病例文本：`topn_cpc_promptv2_87.load_merged()`；"
             f"`PERSPECTIVE_PROMPT.format(structured_case=...) + P_FIXED_SUFFIX`")
    L.append(f"- {meta['n_cases']} 例 × {len(meta['seeds'])} seeds = "
             f"{meta['n_calls_planned']} 次调用；输出 `{meta['outdir']}"
             f"/Pfixed_s{{1..5}}.jsonl`")
    ps = meta["parse_stats"]
    acct = ps.get("attempt_accounting") or {}
    L.append(f"- 空 top5 / 解析失败：本次运行 "
             f"{json.dumps(ps.get('per_seed', {}), ensure_ascii=False)}；"
             f"跨次累加 {json.dumps(ps.get('aggregate_per_seed', {}), ensure_ascii=False)}；"
             f"尝试级账本（含未落盘的尝试，来自 "
             f"{sorted((acct.get('logs') or {}).keys())}）共 "
             f"{acct.get('failed_attempts_total', 'NA')} 次失败尝试"
             f"（{json.dumps({k: v['failed_attempts_by_seed'] for k, v in (acct.get('logs') or {}).items()}, ensure_ascii=False)}）")
    L.append(f"- ⚠ 该空输出**与花括号无关**：直测显示原始响应在 "
             f"`CONFIDENCE_SCORE:` 后直接结束、没有 JSON 块（`PERSPECTIVE_PROMPT` "
             f"自身要求的格式先于追加的 JSON 要求），输出中不含 `{{{{`；"
             f"主实验 P 臂 435 行全部为 5 项（qwen 自愈 + 运气），本脚本按规格重试，"
             f"只替换本会空输出的行，不改变行集合。逐 seed 计数："
             f"{json.dumps(ps.get('per_seed', {}), ensure_ascii=False)}")
    L.append(f"- 判官：GLM-5.3-flash × v3 规则，共享缓存 `{meta['judge_cache']}`"
             f"，只补缺失对；本表缺失对 "
             f"{ {k: v['judge_missing_pairs'] for k, v in meta['stats'].items()} }")
    L.append(f"- 数据完备性（各臂覆盖病例集全集、对比样本对齐）："
             f"{ {k: bool(v) for k, v in (meta.get('readiness') or {}).items()} }"
             + (f"；⚠ 阻塞项 {meta.get('readiness_blockers', [])[:8]}"
                if meta.get("readiness_blockers") else ""))
    L.append("- 统计口径（同 `stats_caselevel.py`）：每例 5-seed 命中率 → 跨病例配对 "
             "Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次 95% CI + 多数决"
             "（>=3/5）精确 McNemar；另附跨 seed 合并 McNemar 作旧口径对照。\n")

    for name in splits_order:
        res = meta["stats"].get(name)
        if not res:
            continue
        L.append(f"\n## 3. 病例集 {name}（n={res['n_cases']}）\n")
        L.append("| 方案 | 覆盖 n/预期 | top-1 | top-3 | top-5 | 病例级命中率 top-1/3/5 |")
        L.append("|---|---|---|---|---|---|")
        n_exp = len((meta["splits_ids"].get(name) or []))
        for arm in ("Pfixed", "P", "Ax1", "MDT"):
            a = (res["arms"] or {}).get(arm)
            if not a:
                continue
            cr = a["case_rate_mean"]
            crm = "/".join(
                f"{cr[f'top{k}']*100:.1f}%" if cr[f"top{k}"] is not None else "NA"
                for k in cs.METHOD_KEYS)
            covered = [p["n"] for p in a["per_seed"]]
            cov = f"{min(covered)}–{max(covered)}/{n_exp}" if covered else "NA"
            L.append(f"| {ARM_LABEL[arm]} | {cov} | "
                     f"{cs.acc_cell(a['mean_sd'], 1)} | "
                     f"{cs.acc_cell(a['mean_sd'], 3)} | {cs.acc_cell(a['mean_sd'], 5)} "
                     f"| {crm} |")
        L.append("\n（逐 seed 准确率均值 ± SD；末列为病例级 5-seed 命中率均值；"
                 "「覆盖」为该臂逐 seed 实际参与统计的病例数范围 —— 只有覆盖达到 "
                 f"{n_exp}/{n_exp} 时该臂的数字才与其它臂严格可比）\n")
        L.append("| 对比 | top-k | 命中率 A vs B | n_cases/预期 | 均值差 [95% CI] | "
                 "Wilcoxon p | 多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p |")
        L.append("|---|---|---|---|---|---|---|---|")
        for pair in ("Pfixed_vs_P", "Pfixed_vs_Ax1", "Pfixed_vs_MDT",
                     "P_vs_Ax1", "P_vs_MDT"):
            by_k = (res["comparisons"] or {}).get(pair) or {}
            a, b = pair.split("_vs_")
            for k in cs.METHOD_KEYS:
                c = by_k.get(f"top{k}")
                if not c or c["mean_rate_a"] is None:
                    continue
                lo, hi = c["boot95_ci"]
                maj, old = c["majority"], c["pooled_mcnemar"]
                flag = "" if c["n_cases"] == n_exp else "⚠"
                L.append(
                    f"| {ARM_LABEL[a].split('（')[0]} vs {ARM_LABEL[b].split('（')[0]} "
                    f"| top-{k} | {c['mean_rate_a']*100:.1f}% vs "
                    f"{c['mean_rate_b']*100:.1f}% | {c['n_cases']}/{n_exp}{flag} | "
                    f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                    f"{maj['a_only']}:{maj['b_only']} p="
                    f"{cs.fmt_p(maj['mcnemar_p'])}{cs.sig(maj['mcnemar_p'])} | "
                    f"{old['a_only']}:{old['b_only']} p="
                    f"{cs.fmt_p(old['p'])}{cs.sig(old['p'])} |")
        L.append("")

    L.append("\n## 4. 修正花括号后 P 是否与主实验 P 一致（方向 / 效应量 / 显著性）\n")
    L.append("### 4.1 两臂正面比（Pfixed vs 主实验 P）\n")
    L.append("| 病例集 | top-k | n_cases/预期 | Pfixed vs P 均值差 [95% CI] | "
             "Wilcoxon p | 多数决 McNemar (Pfixed:P) p |")
    L.append("|---|---|---|---|---|---|")
    for split in splits_order:
        by_k = ((meta["stats"].get(split) or {}).get("comparisons") or {}).get(
            "Pfixed_vs_P") or {}
        n_exp = len((meta["splits_ids"].get(split) or []))
        for k in cs.METHOD_KEYS:
            c = by_k.get(f"top{k}")
            if not c or c["mean_rate_a"] is None:
                continue
            lo, hi = c["boot95_ci"]
            flag = "" if c["n_cases"] == n_exp else " ⚠"
            L.append(f"| {split} | top-{k} | {c['n_cases']}/{n_exp}{flag} | "
                     f"{c['mean_diff']*100:+.1f}pp "
                     f"[{lo*100:+.1f}, {hi*100:+.1f}] | "
                     f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                     f"{c['majority']['a_only']}:{c['majority']['b_only']} p="
                     f"{cs.fmt_p(c['majority']['mcnemar_p'])}"
                     f"{cs.sig(c['majority']['mcnemar_p'])} |")
    L.append("")

    L.append("### 4.2 下游结论是否被改写（Pfixed vs 参照系，对照主实验 P vs 同一参照系）\n")
    L.append("| 病例集 | 对比 | top-k | 主实验 P 口径 | Pfixed 口径 | n(Pfixed/P) | 方向 | "
             "效应量差 | Wilcoxon 显著性 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    changed, misaligned = [], []
    for split in splits_order:
        for opp in ("Ax1", "MDT"):
            for k in cs.METHOD_KEYS:
                c = (cons.get(split) or {}).get(opp, {}).get(f"top{k}")
                if not c:
                    continue
                if not c["aligned"]:
                    misaligned.append(f"{split} P vs {opp} top-{k}")
                    L.append(
                        f"| {split} | P vs {opp} | top-{k} | "
                        f"{c['mean_diff_main']*100:+.1f}pp | "
                        f"{c['mean_diff_fixed']*100:+.1f}pp | "
                        f"{c['n_cases_fixed']}/{c['n_cases_main']}"
                        f"（预期 {c['n_cases_expected']}）| ⚠ 样本不对齐，下表判定无效 |"
                        f" — | — |")
                    continue
                verdict = ("同" if c["direction_same"] else "**反转**")
                eff = ("同(≤2pp)" if c["effect_within_tol"]
                       else f"**差{c['effect_delta_pp']:+.1f}pp**")
                sg = ("同" if c["wilcoxon_sig_same"] else "**改变**")
                if not (c["direction_same"] and c["effect_within_tol"]
                        and c["wilcoxon_sig_same"] and c["majority_sig_same"]):
                    changed.append(f"{split} P vs {opp} top-{k}")
                L.append(
                    f"| {split} | P vs {opp} | top-{k} | "
                    f"{c['mean_diff_main']*100:+.1f}pp, p="
                    f"{cs.fmt_p(c['wilcoxon_p_main'])}{cs.sig(c['wilcoxon_p_main'])} | "
                    f"{c['mean_diff_fixed']*100:+.1f}pp, p="
                    f"{cs.fmt_p(c['wilcoxon_p_fixed'])}{cs.sig(c['wilcoxon_p_fixed'])} | "
                    f"{c['n_cases_fixed']}/{c['n_cases_main']} | "
                    f"{verdict} | {eff} | {sg} |")
    L.append("")
    if misaligned:
        L.append(f"⚠ **{len(misaligned)} 条对比的样本不对齐**（n_cases ≠ 病例集规模，"
                 f"通常意味着某个臂缺行或缺判官判定）："
                 + "；".join(misaligned) + "。这些条目的方向/效应量差不可解读。")
    if changed:
        L.append(f"方向/效应量/显著性**发生改变**的条目（{len(changed)}）："
                 + "；".join(changed))
    elif not misaligned:
        L.append("逐条比对：所有对比在 Pfixed 口径下的**方向、效应量（≤2pp）、"
                 "Wilcoxon 与多数决 McNemar 显著性**均与主实验 P 口径一致。")

    L.append("\n### 4.3 与 seed 间自身波动的比较\n")
    L.append("| 病例集 | top-k | 主实验 P 逐 seed 准确率 mean±SD [min,max] | "
             "Pfixed 逐 seed 准确率 mean±SD [min,max] | 两臂均值差 |")
    L.append("|---|---|---|---|---|")
    for split in splits_order:
        ids = meta["splits_ids"].get(split) or []
        for k in cs.METHOD_KEYS:
            sp_main = seed_spread(meta["arms_raw"]["P"], ids, k)
            sp_fix = seed_spread(meta["arms_raw"]["Pfixed"], ids, k)
            if not sp_main or not sp_fix:
                continue
            L.append(f"| {split} | top-{k} | {sp_main['mean']*100:.1f} ± "
                     f"{sp_main['sd']*100:.1f} [{sp_main['min']*100:.1f},"
                     f"{sp_main['max']*100:.1f}] | {sp_fix['mean']*100:.1f} ± "
                     f"{sp_fix['sd']*100:.1f} [{sp_fix['min']*100:.1f},"
                     f"{sp_fix['max']*100:.1f}] | "
                     f"{(sp_fix['mean']-sp_main['mean'])*100:+.1f}pp |")
    L.append("\n（若两臂均值差落在 P 臂自身的 seed 间 SD / 极差之内，则该差异与"
             "「同一提示词重复采样」的波动同量级，不能归因于花括号修正。）\n")

    L.append("\n## 5. 结论\n")
    L.append(meta["verdict"])
    L.append("")
    L.append("## 6. 说明与接口\n")
    L.append("- 本脚本**未修改** `topn_cpc_promptv2.py` 与任何主实验结果文件；"
             "修正只作用于本进程内的后缀副本，主实验口径不受污染。")
    L.append("- 判分阶段与其它脚本共享 `judge_cache_glm_v3.json`（只补缺失对）；"
             "该文件的全量写入不是原子的，跨进程并发跑 judge 会互相覆盖 —— "
             "本脚本的 judge 阶段须串行调度，不要与其它 judge 任务并行。")
    L.append(f"- 生成脚本 `routing_study/scripts/p_fixed_cpc.py`；"
             f"命令：`PHASE={meta['phase']} ./.venv/bin/python "
             f"routing_study/scripts/p_fixed_cpc.py`")
    return "\n".join(L) + "\n"


def render_verdict(cons, stats, splits_order, readiness, tol_pp=2.0):
    """程序化给出最终判定句；数据不完整时拒绝下结论。"""
    # 数据完备性：两条口径必须落在同一批完整病例上，否则任何"一致/不一致"都无意义
    not_ready = [s for s in splits_order if not readiness.get(s, False)]
    if not_ready:
        return ("**答：暂不能判定。**以下病例集的数据未就绪（某臂缺行、样本数不足"
                "病例集规模、或缺判官判定）：" + "；".join(not_ready)
                + "。补齐 `PHASE=infer` 与 `PHASE=judge` 后重跑 `PHASE=analyze`。"
                + "在此之前请勿引用本文件中的数字。")
    head_changed = []
    for split in splits_order:
        by_k = ((stats.get(split) or {}).get("comparisons") or {}).get(
            "Pfixed_vs_P") or {}
        for k in cs.METHOD_KEYS:
            c = by_k.get(f"top{k}")
            if not c or c["mean_rate_a"] is None:
                continue
            lo, hi = c["boot95_ci"]
            if (c["wilcoxon_p"] or 1) < 0.05 or lo > 0 or hi < 0:
                head_changed.append(f"{split} top-{k}（Pfixed vs P 自身显著差异）")
    head = ("**答：一致。**修正花括号后 P 的结果与主实验 P 在方向、效应量与显著性"
            "三个维度上均无法区分" if not head_changed else
            "**答：不完全一致**，两臂正面比出现显著差异："
            + "；".join(head_changed))
    flipped, eff_only = [], []
    for split in splits_order:
        for opp in ("Ax1", "MDT"):
            for k in cs.METHOD_KEYS:
                c = (cons.get(split) or {}).get(opp, {}).get(f"top{k}")
                if not c or not c["aligned"]:
                    continue
                if not (c["direction_same"] and c["wilcoxon_sig_same"]
                        and c["majority_sig_same"]):
                    detail = []
                    if not c["direction_same"]:
                        detail.append("方向反转")
                    if not c["wilcoxon_sig_same"]:
                        detail.append(f"Wilcoxon p {cs.fmt_p(c['wilcoxon_p_main'])}→"
                                      f"{cs.fmt_p(c['wilcoxon_p_fixed'])}")
                    if not c["majority_sig_same"]:
                        detail.append("多数决 McNemar 显著性改变")
                    flipped.append(f"{split} P vs {opp} top-{k}（"
                                   + "、".join(detail) + "）")
                elif not c["effect_within_tol"]:
                    eff_only.append(f"{split} P vs {opp} top-{k}"
                                    f"（{c['effect_delta_pp']:+.1f}pp）")
    if flipped:
        tail = ("主结论被改写：" + "；".join(flipped) + "，需按 Pfixed 口径修正正文表述。")
    elif eff_only:
        tail = ("下游对比（P vs A×1、P vs MDT）在三个病例集 × top-1/3/5 上的**方向与"
                "显著性状态全部一致**；仅以下条目的效应量差超过 " + str(tol_pp)
                + "pp：" + "；".join(eff_only)
                + "。请对照 §4.3 的 seed 间波动判断该差异是否超出重复采样噪声。")
    else:
        tail = ("下游对比（P vs A×1、P vs MDT）在三个病例集 × top-1/3/5 上全部保持"
                "方向相同、效应量差 ≤2pp、Wilcoxon 与多数决 McNemar 显著性状态一致 "
                "→ 主实验中与 P 臂相关的结论对花括号缺陷稳健，无需改写；"
                "该缺陷只需要在方法/局限里如实交代（主实验 P 的实际提示词含字面"
                "双花括号，qwen 自愈；第二模型族因此加了纯解析层兜底）。")
    return head + "\n\n" + tail


def run_analyze():
    cases = load_merged()
    audit = prompt_audit(cases)
    print(f"[审计] 逐例 diff：{audit['n_single_line_diff']}/{audit['n_cases']} 例"
          f"恰有 1 行不同（花括号折叠），异常 {audit['n_unexpected']}", flush=True)
    if audit["unexpected"]:
        raise SystemExit(f"提示词审计失败：{audit['unexpected']}")

    arms = build_arms(SEEDS)
    cache_path = Path(cs.GLM_CACHE)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cs.set_cache(cache)
    spans = cs.split_ids()
    stats = cs.split_stats(arms, spans, PAIRS)
    cons = consistency_table(stats, ["full87", "heldout46", "dev41"],
                             {k: len(v) for k, v in spans.items()})

    # 数据完备性：每个臂的每个 seed 必须覆盖病例集全集，且每条对比的 n_cases 达到
    # 病例集规模 —— 否则"方向/效应量"的差异只是样本差，不能下结论。
    readiness, blockers = {}, []
    for split, ids in spans.items():
        ok = True
        for arm, runs in arms.items():
            for s, rows in runs.items():
                if sum(1 for cid in ids if cid in rows) != len(ids):
                    blockers.append(f"{split}/{arm}/s{s} 覆盖 {sum(1 for cid in ids if cid in rows)}/{len(ids)}")
                    ok = False
        by = (stats.get(split) or {}).get("comparisons") or {}
        for pair in (f"Pfixed_vs_{o}" for o in ("P", "Ax1", "MDT")):
            for k in cs.METHOD_KEYS:
                c = (by.get(pair) or {}).get(f"top{k}")
                if not c or c["mean_rate_a"] is None or c["n_cases"] != len(ids):
                    blockers.append(f"{split}/{pair}/top{k} 样本不对齐")
                    ok = False
        readiness[split] = ok
    if blockers:
        print(f"[分析] ⚠ 数据不完备（{len(blockers)} 项，节选 "
              f"{blockers[:5]}）—— §5 结论将标注为不可判定", flush=True)

    parse_stats = json.loads((OUTDIR / "parse_stats.json").read_text()) \
        if (OUTDIR / "parse_stats.json").exists() else {
            "per_seed": {}, "hard_failures": [], "n_calls_planned": 87 * len(SEEDS)}
    meta = {
        "phase": os.environ.get("PHASE", "all"),
        "provider": resolve_provider("qwen"), "model": QWEN_MODEL,
        "temperature": TEMPERATURE, "disable_thinking": True, "max_tokens": 2048,
        "timeout": 300, "call": "topn_cpc.call_top5",
        "n_cases": len(cases), "seeds": SEEDS,
        "n_calls_planned": parse_stats.get("n_calls_planned"),
        "outdir": str(OUTDIR), "main_seeds_dir": str(MAIN_SEEDS_DIR),
        "mdt_dir": str(MDT_DIR),
        "judge_cache": str(cache_path), "judge_model": cs.GLM_MODEL,
        "judge_cache_size": len(cache),
        "judge_cache_added_by_this_script": sum(
            1 for s in SEEDS for r in load_arm("Pfixed", s).values()
            for c in r["top5"][:5] if cs.key_of(r["gold"], c) not in cache),
        "parse_stats": parse_stats,
        "stats": stats, "splits_ids": spans, "arms_raw": arms,
        "consistency": cons, "readiness": readiness, "readiness_blockers": blockers,
        "prompt_audit": audit,
        "prompt": "PERSPECTIVE_PROMPT.format(structured_case=...) + P_FIXED_SUFFIX"
                  "（P_FIXED_SUFFIX = P_TOPN_SUFFIX 折叠 {{ }}）",
    }
    missing = sum(v["judge_missing_pairs"] for v in stats.values())
    if missing:
        print(f"[分析] 警告：{missing} 个 (病例×seed×候选) 判官缺失，"
              f"相关病例已从病例级统计中剔除（先跑 PHASE=judge）", flush=True)
    meta["verdict"] = render_verdict(cons, stats, ["full87", "heldout46", "dev41"],
                                     readiness)
    md = render_md(meta, stats, cons, audit, ["full87", "heldout46", "dev41"])
    payload = {k: v for k, v in meta.items() if k != "arms_raw"}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"\n已写入 {OUT_JSON}\n已写入 {OUT_MD}", flush=True)


# ---------- 冒烟 ----------

def smoke():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    example = self_check()
    cases = load_merged()[:3]
    audit = prompt_audit(cases)
    seed = SEEDS[0]
    path = SMOKE_DIR / f"Pfixed_s{seed}.jsonl"
    for p in (path, SMOKE_DIR / "failures.jsonl"):
        if p.exists():
            p.unlink()
    print(f"\n[冒烟] {len(cases)} 例 → {SMOKE_DIR}（seed {seed}，T={TEMPERATURE}）",
          flush=True)
    print(f"[冒烟] 折叠后的 JSON 示例：{example}", flush=True)
    print(f"[冒烟] 逐例审计：{audit['n_single_line_diff']}/{audit['n_cases']} 例恰有 1 "
          f"行不同（花括号折叠）；每例字符差 {audit['per_case_char_delta']}", flush=True)
    for s in audit["sample_changed_lines"]:
        print(f"[冒烟] diff（第 {s['line_no']} 行，{s['char_delta']} 字符）\n"
              f"  main : {s['main']}\n  fixed: {s['fixed']}", flush=True)
    failures = []
    st_ = {"first_attempt_empty": 0, "needed_retry": 0}
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
            print(f"  {c['case_id'][:50]} | 失败：无完整 top5")
            continue
        print(f"  {c['case_id'][:50]}\n"
              f"    gold   : {c['gold'][:80]}\n"
              f"    top5[0]: {r['top5'][0]} | tokens={r['total_tokens']} | "
              f"字段={sorted(r)}")
    print(f"[冒烟] 通过: {ok}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    if os.environ.get("SMOKE") == "1":
        smoke()
        return
    self_check()
    if not SEEDS:
        raise SystemExit("PFIXED_SEEDS 为空")
    phase = os.environ.get("PHASE", "all").lower()
    print(f"seeds: {SEEDS} | phase: {phase} | outdir: {OUTDIR}", flush=True)

    if phase in ("all", "infer"):
        cases = load_merged()
        run_infer(cases)
        # 完整性检查：硬失败的病例不会落残缺行（缺行），重跑本命令即续跑补齐
        problems = []
        for seed in SEEDS:
            rows = list(load_done(OUTDIR / f"Pfixed_s{seed}.jsonl").values())
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
            print("[检查] 上述病例已连续 "
                  f"{MAX_ATTEMPTS} 次拿不到有效 top5（多为超长病例的"
                  "「不写 JSON 块」失败，非花括号问题）。"
                  "直接重跑同一条命令即可续跑补齐（已有行自动跳过）：\n"
                  f"  PHASE=infer ./.venv/bin/python "
                  f"routing_study/scripts/p_fixed_cpc.py", flush=True)
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
