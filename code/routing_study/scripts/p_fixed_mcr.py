#!/usr/bin/env python3
"""MCR 406 例外部队列：P 臂「字面双花括号」提示词缺陷的修正版（Pfixed）。

背景（与 CPC 版 `p_fixed_cpc.py` 同源）
------------------------------------
主实验 P 臂的提示词是 `PERSPECTIVE_PROMPT.format(structured_case=...) +
P_TOPN_SUFFIX`。`P_TOPN_SUFFIX`（`topn_cpc_promptv2.py`）是一段**从不经过
`.format()`** 的普通字符串，其 JSON 块写作字面双花括号
`{{"top5": [{{"rank": 1, ...}}]}}`，因此**实际发出**的 P 提示词里就是双花括号
（逐字节文本见 `results/prompt_actually_sent.md`；MCR 主实验 P 臂见
`topn_mcr.py:274-276` / `seeds_mcr.py:158-161`）。qwen3.8-flash 会自行改写为
单花括号，deepseek-flash 会原样照抄 → `extract_json` 失败（第二模型族 P 臂
约 7% 行解析失败）。CPC 上的修正版复测显示：修正后 P 的召回变高，MDT 相对 P
的优势被压缩（见 `results/p_fixed_cpc.md`）。本脚本在**外部验证集 MCR** 上
复测同一对比。

替换规则（只在本脚本内做，不改共享常量）
--------------------------------------
    P_FIXED_SUFFIX = P_TOPN_SUFFIX.replace("{{", "{").replace("}}", "}")

- **不修改 `topn_cpc_promptv2.py`**（改它会污染主实验口径，且多个脚本共享该
  常量）；归一化只作用于本进程内的常量副本。
- `P_TOPN_SUFFIX` 中所有花括号都是**成对转义**（`{`、`}` 极大游程长度均为
  偶数），折叠后无残留 `{{`/`}}`，折叠后的 JSON 示例可被 `json.loads` 解析
  —— `self_check()` 每次运行都会断言。
- "唯一差别"由**逐例审计**保证：`prompt_audit()` 对 406 例逐一比对两条完整
  提示词，断言行数相同、**恰有 1 行不同**、且该行差异恰为花括号折叠；抽样 10
  例打印具体行。结果写入 `results/p_fixed_mcr.json` + `.md`。

与主实验 MCR P 臂完全一致的推理参数
----------------------------------
同模型（`provider="qwen"` → DashScope API，`QWEN_MODEL` = qwen3.8-flash）、
`temperature=0.3`、`disable_thinking=True`、`max_tokens=4096`、`timeout=300`：
直接复用 `topn_mcr.call_top5(prompt, 0.3)`（其内部参数与主实验 MCR P 臂逐字
相同）；病例文本用 `topn_mcr.load_cases()`。`QWEN_MODEL`/`QWEN_BACKEND` 一律不
在本脚本里覆盖，沿用仓库 `.env` 的解析结果。注意 `topn_mcr.call_top5` **自身
已含 3 次空重试**；本脚本在其外再做最多 3 轮外层尝试（与 CPC 版口径一致），
仍无有效 top5 则不落残缺行、记入 `failures.jsonl`。

输出
----
`results/topn_seeds_mcr_pfixed/Pfixed_s{1..5}.jsonl`，行 schema 与主实验 MCR P
臂一致（`case_id` / `gold` / `top5` / `total_tokens`）；断点续跑（缺行或不足
5 项的行补齐）。另写 `parse_stats.json`（逐 seed 首轮空输出数、重试数、硬失败、
跨次累加与尝试级账本）与 `failures.jsonl`，不改变行 schema。

阶段 PHASE=infer|judge|analyze（默认 all）
- infer：406 例 × 5 seeds = 2030 次调用（另加重试）。
- judge：`caselevel_stats.judge_missing(rows, cache_path=JUDGE_CACHE)`，
  GLM-5.3-flash × `judge_v3.V3_PROMPT`，**只补缺失对**。缓存路径**必须**由
  环境变量显式指定（`PFIXED_MCR_JUDGE_CACHE`，其次 `GLM_CACHE`）；未指定则
  拒绝运行 —— 以免误写共享的 `judge_cache_glm_v3.json`。
- analyze：病例级主口径（同 `stats_caselevel.py` / `caselevel_stats.py`）：
  每例 5-seed 平均命中率 → 跨病例配对 Wilcoxon 双侧 + 病例级 cluster bootstrap
  10,000 次 95% CI + 多数决（>=3/5）精确 McNemar。对比 Pfixed vs 主实验 P /
  A×1 / MDT，并给出 MDT vs P 与 MDT vs Pfixed 的对照，回答「修正花括号后 MCR 上
  MDT 是否仍显著优于 P」。产出 `results/p_fixed_mcr.json` + `.md`。

环境变量：PFIXED_MCR_OUTDIR（默认 results/topn_seeds_mcr_pfixed）、
PFIXED_MCR_SEEDS（默认 "1,2,3,4,5"）、MAX_WORKERS（默认 4，同
`topn_cpc.MAX_WORKERS`）、PFIXED_MCR_JUDGE_CACHE / GLM_CACHE（判分缓存路径）。
SMOKE=1 只跑前 3 例到 `/tmp/pfixed_mcr_smoke`（可用 SMOKE_DIR 覆盖），不判定、
不分析。
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
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2 import P_TOPN_SUFFIX  # noqa: E402  主实验原常量（只读）
from topn_mcr import load_cases, call_top5  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from run_inference import resolve_provider, QWEN_MODEL  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("PFIXED_MCR_OUTDIR", RESULTS / "topn_seeds_mcr_pfixed"))
S1_DIR = RESULTS / "topn_mcr"           # seed 1 的主实验臂
SEEDS_DIR = RESULTS / "topn_mcr_seeds"  # seed 2–5 的主实验臂
OUT_JSON = RESULTS / "p_fixed_mcr.json"
OUT_MD = RESULTS / "p_fixed_mcr.md"
SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/pfixed_mcr_smoke"))
SEEDS = [int(s) for s in
         os.environ.get("PFIXED_MCR_SEEDS", "1,2,3,4,5").replace(",", " ").split()]
TEMPERATURE = 0.3
MAX_ATTEMPTS = 3
SPLIT = "mcr406"
ARM_LABEL = {
    "Pfixed": "Pfixed（转义花括号已修正）",
    "P": "P（主实验，提示词含字面双花括号）",
    "Ax1": "A×1（T=0，作参照）",
    "MDT": "MDT（T=0.3，作参照）",
}
# A 在前：mean_diff = A - B，正数表示 A 更高
# （"P","MDT") 与 ("MDT","P") 是同一对比的两种朝向，前者供 §4.2 的
# 「主实验 P vs 参照系」对照使用，后者供 §4 的 headline 表使用。
PAIRS = [("Pfixed", "P"), ("Pfixed", "Ax1"), ("Pfixed", "MDT"),
         ("MDT", "P"), ("MDT", "Ax1"), ("P", "MDT"), ("P", "Ax1")]


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
    """断言折叠合法且是「唯一差别」：无残留双花括号、JSON 示例可解析。"""
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


def prompt_audit(cases, sample_n=10):
    """逐例审计：两条完整提示词行数相同、恰有 1 行不同、差异恰为花括号折叠。"""
    bad, samples, sha_main, sha_fixed = [], [], [], []
    diff_line_nos = []
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
        i = idx[0]
        diff_line_nos.append(i + 1)
        if len(samples) < sample_n:
            samples.append({
                "case_id": c["case_id"], "line_no": i + 1,
                "total_lines": len(lo),
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
        "diff_line_nos_min_max": [min(diff_line_nos), max(diff_line_nos)]
                                 if diff_line_nos else None,
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
                # topn_mcr.call_top5 内部已含 3 次空重试，参数与主实验 MCR P 臂
                # 逐字相同（max_tokens=4096 / timeout=300 / disable_thinking）
                top5, tokens = call_top5(prompt, TEMPERATURE)
            except Exception as e:  # 网络异常一律重试
                top5, tokens, last_err = [], 0, f"{type(e).__name__}: {e}"
            if top5:
                return ({"case_id": c["case_id"], "gold": c["gold"],
                         "top5": top5, "total_tokens": tokens},
                        empty_first, attempt + 1)
            empty_first = True
            last_err = last_err or "empty top5 after parse"
            print(f"[Pfixed s{seed}] {c['case_id'][:40]} 空 top5/解析失败，"
                  f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续 {MAX_ATTEMPTS} 轮无有效 top5"
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
            if i % 25 == 0 or i == len(todo):
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


RETRY_LINE = re.compile(r"\[Pfixed s(\d+)\] .*空 top5/解析失败，重试 (\d+)/(\d+)")


def log_accounting(logs):
    """从运行日志统计「失败尝试」次数（尝试级记录只在日志里）。"""
    per_log, total = {}, 0
    for p in logs:
        p = Path(p)
        if not p.exists():
            continue
        by_seed = {}
        for line in open(p, errors="replace"):
            m = RETRY_LINE.search(line)
            if m:
                k = f"s{int(m.group(1))}"
                by_seed[k] = by_seed.get(k, 0) + 1
        if by_seed:
            per_log[p.name] = {"failed_attempts_by_seed": by_seed,
                               "failed_attempts_total": sum(by_seed.values())}
            total += sum(by_seed.values())
    return {"logs": per_log, "failed_attempts_total": total,
            "note": "failed_attempts = 外层尝试中返回空/解析失败的次数"
                    "（每轮外层尝试内部 topn_mcr.call_top5 还会自带 3 次调用）"}


def run_infer(cases, outdir=None):
    outdir = Path(outdir or OUTDIR)
    print(f"[推理] provider=qwen（解析后 transport={resolve_provider('qwen')}）"
          f" model={QWEN_MODEL} | T={TEMPERATURE} disable_thinking=True "
          f"max_tokens=4096（复用 topn_mcr.call_top5）| MAX_WORKERS={MAX_WORKERS}",
          flush=True)
    per_seed, failures = {}, []
    for seed in SEEDS:
        s = {"first_attempt_empty": 0, "needed_retry": 0, "hard_failed": 0}
        run_one(seed, cases, outdir, stats=s, failures=failures)
        per_seed[f"s{seed}"] = s
    fpath = outdir / "failures.jsonl"
    history = load_failures(fpath)
    history.extend(failures)
    if history:
        fpath.write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in history) + "\n",
            encoding="utf-8")
    if failures:
        print(f"[推理] 硬失败 {len(failures)} 例（连续 {MAX_ATTEMPTS} 轮无有效 "
              f"top5，未落盘）："
              + "；".join(f"{f['seed']}:{f['case_id'][:36]}" for f in failures),
              flush=True)
    stats = {
        "model": QWEN_MODEL, "provider": resolve_provider("qwen"),
        "temperature": TEMPERATURE, "disable_thinking": True,
        "max_tokens": 4096, "timeout": 300, "call": "topn_mcr.call_top5",
        "n_cases": len(cases), "seeds": SEEDS, "max_workers": MAX_WORKERS,
        "n_calls_planned": len(cases) * len(SEEDS),
        "per_seed": per_seed, "hard_failures": failures,
        "note": "first_attempt_empty = 首轮（含 call_top5 内部 3 次）无有效 top5 的"
                "行数；needed_retry = 需要第二轮外层尝试才成功的行数；"
                "hard_failed = 外层重试用尽仍未落盘的病例数",
    }
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
    logs = sorted(RESULTS.glob("pfixed_mcr_infer*.log"))
    stats.update({"runs": runs, "aggregate_per_seed": agg,
                  "hard_failures_total": sum(len(r.get("hard_failures") or [])
                                             for r in runs),
                  "attempt_accounting": log_accounting(logs),
                  "counter_note": "per_seed/aggregate_per_seed 为病例级计数，"
                                  "只在每次运行结束时写入；尝试级账本在 "
                                  "attempt_accounting（含未落盘的尝试）。"})
    pstats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[推理] parse_stats（本次）: {json.dumps(per_seed, ensure_ascii=False)}"
          f" | 硬失败 {len(failures)}", flush=True)
    print(f"[推理] 尝试级账本（{len(logs)} 个日志）: "
          f"{stats['attempt_accounting']['failed_attempts_total']} 次失败尝试",
          flush=True)
    return stats


# ---------- 判分缓存路径（必须显式指定，避免误写共享缓存） ----------

def _same_path(a, b):
    """比较两个路径是否指向同一文件（相对/绝对/符号链接都要认出来）。"""
    try:
        return os.path.realpath(str(a)) == os.path.realpath(str(b))
    except Exception:
        return False


def resolve_judge_cache(for_write=False, required=True):
    """判分缓存路径必须由环境变量显式指定（默认不写共享缓存）。

    for_write=True（judge 阶段）时**拒绝**指向共享主缓存
    `judge_cache_glm_v3.json`（该文件由主代理统一调度）。路径比较走
    `realpath`，相对路径/绝对路径/软链都能识别。analyze 只读，允许任意路径。
    """
    for var in ("PFIXED_MCR_JUDGE_CACHE", "GLM_CACHE"):
        v = os.environ.get(var)
        if not v:
            continue
        p = Path(v)
        if for_write and _same_path(p, cs.GLM_CACHE):
            raise SystemExit(
                f"{var}={v} 解析后等于共享主缓存 {cs.GLM_CACHE}，本脚本拒绝写入"
                "该文件。请指定分片路径，例如 "
                "PFIXED_MCR_JUDGE_CACHE=routing_study/results/judge_shard_01.json")
        return p, var
    if required:
        raise SystemExit(
            "judge/analyze 需要显式指定判分缓存路径（默认不写共享的 "
            "judge_cache_glm_v3.json）：\n"
            "  PFIXED_MCR_JUDGE_CACHE=/path/to/judge_shard_xx.json PHASE=judge "
            "./.venv/bin/python routing_study/scripts/p_fixed_mcr.py")
    return None, None


# ---------- 阶段 2：判分（GLM v3，只补缺失对） ----------

def load_pfixed(seed, outdir=None):
    return cs.load(Path(outdir or OUTDIR) / f"Pfixed_s{seed}.jsonl")


def main_arm_rows(arm, seed):
    """主实验 MCR 的各臂：seed 1 在 topn_mcr/，seed 2–5 在 topn_mcr_seeds/。"""
    if arm == "P":
        return cs.load(S1_DIR / "p.jsonl" if seed == 1
                       else SEEDS_DIR / f"P_s{seed}.jsonl")
    if arm == "Ax1":
        return cs.load(S1_DIR / "ax1.jsonl" if seed == 1
                       else SEEDS_DIR / f"Ax1_s{seed}.jsonl")
    return cs.load(S1_DIR / "mdt_synth.jsonl" if seed == 1
                   else SEEDS_DIR / f"s{seed}" / "mdt_synth.jsonl")


def run_judge(seeds=None):
    seeds = seeds or SEEDS
    cache_path, var = resolve_judge_cache(for_write=True)
    rows = []
    for seed in seeds:
        rows.extend(load_pfixed(seed).values())
    n_before = len(json.loads(cache_path.read_text())) if cache_path.exists() else 0
    print(f"[判定] 缓存 {cache_path}（来自 ${var}）现有 {n_before} 对", flush=True)
    cache = cs.judge_missing(rows, cache_path=cache_path)
    print(f"[判定] 缓存 {n_before} → {len(cache)}"
          f"（新增 {len(cache) - n_before} 对）", flush=True)
    return cache


# ---------- 阶段 3：分析（病例级主口径） ----------

def build_arms(seeds, outdir=None):
    return {
        "Pfixed": {s: load_pfixed(s, outdir) for s in seeds},
        "P": {s: main_arm_rows("P", s) for s in seeds},
        "Ax1": {s: main_arm_rows("Ax1", s) for s in seeds},
        "MDT": {s: main_arm_rows("MDT", s) for s in seeds},
    }


def flip_cmp(c):
    """把 paired_caselevel 结果转成 A/B 互换后的同一对比（MDT 相对 P 的优势）。"""
    if not c or c["mean_rate_a"] is None:
        return None
    lo, hi = c["boot95_ci"]
    return {"n_cases": c["n_cases"],
            "mean_rate_a": c["mean_rate_b"], "mean_rate_b": c["mean_rate_a"],
            "mean_diff": -c["mean_diff"], "boot95_ci": [-hi, -lo],
            "wilcoxon_p": c["wilcoxon_p"],
            "majority": {"a_only": c["majority"]["b_only"],
                         "b_only": c["majority"]["a_only"],
                         "mcnemar_p": c["majority"]["mcnemar_p"]},
            "pooled_mcnemar": {"a_only": c["pooled_mcnemar"]["b_only"],
                               "b_only": c["pooled_mcnemar"]["a_only"],
                               "p": c["pooled_mcnemar"]["p"]}}


def cmp_consistency(fix_c, main_c, tol=0.02):
    """同一对比在 Pfixed 口径与主实验 P 口径下的方向/效应量/显著性是否一致。"""
    if (not fix_c or not main_c or fix_c["mean_rate_a"] is None
            or main_c["mean_rate_a"] is None):
        return None
    d_fix, d_main = fix_c["mean_diff"], main_c["mean_diff"]
    return {
        "mean_diff_fixed": d_fix, "mean_diff_main": d_main,
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
        "n_cases_fixed": fix_c.get("n_cases"),
        "n_cases_main": main_c.get("n_cases"),
    }


def seed_spread(runs, ids, k):
    """某方案的逐 seed top-k 准确率 → mean/sd/min/max。"""
    per = cs.per_seed_metrics(runs, ids, {s: "top5" for s in runs})
    acc = [p[f"top{k}_acc"] for p in per if p[f"top{k}_acc"] is not None]
    if not acc:
        return None
    return {"mean": st.mean(acc), "sd": st.stdev(acc) if len(acc) > 1 else 0.0,
            "min": min(acc), "max": max(acc), "n_seeds": len(acc)}


def md_is_sig(p):
    return p is not None and p < 0.05


def render_md(meta, audit, stats, cons, n_expected):
    res = stats[SPLIT]
    L = []
    L.append("# MCR 406 例：P 臂「字面双花括号」提示词缺陷的修正版（Pfixed）对照\n")
    L.append("主实验 P 臂实际发出的提示词里，JSON 块是**字面双花括号**"
             "（`{{\"top5\": [{{\"rank\": 1, ...}}]}}`），因为 "
             "`topn_cpc_promptv2.P_TOPN_SUFFIX` 是普通字符串、从不经过 `.format()`。"
             "qwen3.8-flash 会自愈，deepseek-flash 会照抄 → 第二模型族 P 臂约 7% 的"
             "行解析失败。CPC 上的修正版复测显示修正后 P 的召回变高、MDT 相对 P 的"
             "优势被压缩；本节在**外部验证集 MedCaseReasoning** 上复测同一对比。\n")

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
    L.append(f"- **逐例审计（全部 {audit['n_cases']} 例）**：两条完整提示词行数相同、"
             f"**恰有 1 行不同**、且该行的差异恰为花括号折叠 → 一致 "
             f"{audit['n_single_line_diff']}/{audit['n_cases']} 例，异常 "
             f"{audit['n_unexpected']} 例。每例字符数差 {audit['per_case_char_delta']}"
             f"（正好 6 对 `{{{{`→`{{`、`}}}}`→`}}`）；差异行号范围 "
             f"{audit['diff_line_nos_min_max']}（病例文本长度不同 → 行号随例变化）。")
    L.append(f"- 完整提示词 sha256：主实验 "
             f"`{str(audit['prompt_sha256_main_case0'])[:16]}…` / Pfixed "
             f"`{str(audit['prompt_sha256_fixed_case0'])[:16]}…`（首例）；"
             f"406 例聚合 sha256（主实验 / Pfixed）= "
             f"`{audit['prompt_sha256_main_all'][:16]}…` / "
             f"`{audit['prompt_sha256_fixed_all'][:16]}…`")
    L.append(f"\n抽样 10 例（每例仅此一行不同）：\n")
    L.append("| # | case_id | 差异行号 | 该系统提示词总行数 | 字符差 |")
    L.append("|---|---|---|---|---|")
    for i, s in enumerate(audit["sample_changed_lines"], 1):
        L.append(f"| {i} | `{s['case_id'][:46]}` | {s['line_no']} | "
                 f"{s['total_lines']} | {s['char_delta']} |")
    s0 = audit["sample_changed_lines"][0]
    L.append(f"\n抽样首例（`{s0['case_id'][:46]}`）该行的内容：\n")
    L.append("```text\n主实验: " + s0["main"] + "\nPfixed: " + s0["fixed"] + "\n```")
    if audit["unified_diff_case0_nocontext"]:
        L.append("\n首例提示词的统一 diff（`-c 0`，仅此一处）：\n")
        L.append("```diff")
        L.extend(audit["unified_diff_case0_nocontext"])
        L.append("```")
    L.append("")

    ps = meta["parse_stats"]
    acct = ps.get("attempt_accounting") or {}
    L.append("## 2. 方案与数据\n")
    L.append(f"- 推理：`provider=qwen` → `{meta['provider']}`，模型 "
             f"`{meta['model']}`；`temperature={meta['temperature']}`、"
             f"`disable_thinking=True`、`max_tokens=4096`、`timeout=300` —— 直接复用 "
             f"`topn_mcr.call_top5`，与主实验 MCR P 臂逐字相同")
    L.append(f"- 病例文本：`topn_mcr.load_cases()`（{meta['n_cases']} 例）；"
             f"`PERSPECTIVE_PROMPT.format(structured_case=...) + P_FIXED_SUFFIX`")
    L.append(f"- {meta['n_cases']} 例 × {len(meta['seeds'])} seeds = "
             f"{meta['n_calls_planned']} 次调用；输出 `{meta['outdir']}"
             f"/Pfixed_s{{1..5}}.jsonl`")
    L.append(f"- 空 top5 / 解析失败：本次运行 "
             f"{json.dumps(ps.get('per_seed', {}), ensure_ascii=False)}；跨次累加 "
             f"{json.dumps(ps.get('aggregate_per_seed', {}), ensure_ascii=False)}；"
             f"尝试级账本共 {acct.get('failed_attempts_total', 'NA')} 次失败尝试")
    L.append(f"- 判官：GLM-5.3-flash × v3 规则，缓存 `{meta['judge_cache']}`"
             f"（来自 `${meta['judge_cache_var']}`）")
    L.append(f"- **各臂判分缺失**（槽位 = 病例×seed×候选，对数 = 去重后的 "
             f"(gold,cand) 对，缓存里出现任一项缺失该病例就会被剔出病例级统计）："
             f"{json.dumps(meta['judge_missing_by_arm'], ensure_ascii=False)}")
    L.append(f"- 数据完备性（各臂覆盖 {n_expected} 例、对比样本对齐）："
             f"{meta['readiness']}")
    L.append("- 统计口径（同 `stats_caselevel.py`）：每例 5-seed 平均命中率 → 跨病例"
             "配对 Wilcoxon 双侧 + 病例级 cluster bootstrap 10,000 次 95% CI + "
             "多数决（>=3/5）精确 McNemar；另附跨 seed 合并 McNemar 作旧口径对照。\n")

    L.append(f"\n## 3. 结果（MCR {res['n_cases']} 例，{len(meta['seeds'])} seeds）\n")
    if not meta["readiness"]:
        L.append("> ⚠ **本节及下节的数字暂不可用**：数据未就绪（见 §2 的数据完备性与"
                 "各臂判分缺失；通常因为判分阶段尚未跑完）。请先跑 `PHASE=judge`"
                 "再重跑 `PHASE=analyze`。\n")
    L.append("| 方案 | 逐 seed 覆盖 | top-1 | top-3 | top-5 | 病例级命中率 top-1/3/5 |")
    L.append("|---|---|---|---|---|---|")
    for arm in ("Pfixed", "P", "Ax1", "MDT"):
        a = res["arms"].get(arm)
        if not a:
            continue
        cr = a["case_rate_mean"]
        crm = "/".join(f"{cr[f'top{k}']*100:.1f}%"
                       if cr[f"top{k}"] is not None else "NA"
                       for k in cs.METHOD_KEYS)
        cov = [p["n"] for p in a["per_seed"]]
        L.append(f"| {ARM_LABEL[arm]} | {min(cov)}–{max(cov)}/{n_expected} | "
                 f"{cs.acc_cell(a['mean_sd'], 1)} | {cs.acc_cell(a['mean_sd'], 3)} | "
                 f"{cs.acc_cell(a['mean_sd'], 5)} | {crm} |")
    L.append(f"\n（逐 seed 准确率均值 ± SD；末列为病例级 5-seed 命中率均值）\n")
    L.append("| 对比 | top-k | 命中率 A vs B | n_cases/预期 | 均值差 [95% CI] | "
             "Wilcoxon p | 多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pair in ("Pfixed_vs_P", "Pfixed_vs_Ax1", "Pfixed_vs_MDT",
                 "MDT_vs_P", "MDT_vs_Ax1", "P_vs_Ax1"):
        by_k = res["comparisons"].get(pair) or {}
        a, b = pair.split("_vs_")
        for k in cs.METHOD_KEYS:
            c = by_k.get(f"top{k}")
            if not c or c["mean_rate_a"] is None:
                continue
            lo, hi = c["boot95_ci"]
            maj, old = c["majority"], c["pooled_mcnemar"]
            flag = "" if c["n_cases"] == n_expected else "⚠"
            L.append(f"| {ARM_LABEL[a].split('（')[0]} vs "
                     f"{ARM_LABEL[b].split('（')[0]} | top-{k} | "
                     f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                     f"{c['n_cases']}/{n_expected}{flag} | "
                     f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                     f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                     f"{maj['a_only']}:{maj['b_only']} p="
                     f"{cs.fmt_p(maj['mcnemar_p'])}{cs.sig(maj['mcnemar_p'])} | "
                     f"{old['a_only']}:{old['b_only']} p="
                     f"{cs.fmt_p(old['p'])}{cs.sig(old['p'])} |")
    L.append("")

    L.append("## 4. 修正花括号后，MCR 上 MDT 是否仍显著优于 P？\n")
    L.append("| top-k | 主实验口径：MDT vs P | 修正口径：MDT vs Pfixed | "
             "MDT 优势的变化 | Wilcoxon 显著性 |")
    L.append("|---|---|---|---|---|")
    flip = {k: flip_cmp((res["comparisons"].get("Pfixed_vs_MDT") or {}).get(f"top{k}"))
            for k in cs.METHOD_KEYS}
    for k in cs.METHOD_KEYS:
        main_c = (res["comparisons"].get("MDT_vs_P") or {}).get(f"top{k}")
        fix_c = flip.get(k)
        if not main_c or not fix_c or main_c["mean_rate_a"] is None:
            continue
        lo_m, hi_m = main_c["boot95_ci"]
        lo_f, hi_f = fix_c["boot95_ci"]
        sig_m = md_is_sig(main_c["wilcoxon_p"])
        sig_f = md_is_sig(fix_c["wilcoxon_p"])
        sig_txt = ("两者都显著 *" if sig_m and sig_f else
                   "主实验显著 → 修正后不显著" if sig_m else
                   "主实验不显著 → 修正后显著" if sig_f else "两者都不显著")
        L.append(f"| top-{k} | {main_c['mean_rate_a']*100:.1f}% vs "
                 f"{main_c['mean_rate_b']*100:.1f}% | "
                 f"{fix_c['mean_rate_a']*100:.1f}% vs {fix_c['mean_rate_b']*100:.1f}% | "
                 f"{main_c['mean_diff']*100:+.1f}pp [{lo_m*100:+.1f}, {hi_m*100:+.1f}] "
                 f"p={cs.fmt_p(main_c['wilcoxon_p'])} → "
                 f"{fix_c['mean_diff']*100:+.1f}pp [{lo_f*100:+.1f}, {hi_f*100:+.1f}] "
                 f"p={cs.fmt_p(fix_c['wilcoxon_p'])} | {sig_txt}{cs.sig(fix_c['wilcoxon_p'])} |")
    L.append("")
    L.append("（「MDT vs Pfixed」= 表 3 中 `Pfixed_vs_MDT` 的 A/B 互换，"
             "故两个口径的样本、判官、统计方法完全相同，只有提示词后缀不同。）\n")

    L.append("### 4.1 两臂正面比（Pfixed vs 主实验 P）\n")
    L.append("| top-k | n_cases/预期 | Pfixed vs P 均值差 [95% CI] | Wilcoxon p | "
             "多数决 McNemar (Pfixed:P) p |")
    L.append("|---|---|---|---|---|")
    by_k = res["comparisons"].get("Pfixed_vs_P") or {}
    for k in cs.METHOD_KEYS:
        c = by_k.get(f"top{k}")
        if not c or c["mean_rate_a"] is None:
            continue
        lo, hi = c["boot95_ci"]
        L.append(f"| top-{k} | {c['n_cases']}/{n_expected}"
                 f"{'' if c['n_cases'] == n_expected else ' ⚠'} | "
                 f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                 f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                 f"{c['majority']['a_only']}:{c['majority']['b_only']} p="
                 f"{cs.fmt_p(c['majority']['mcnemar_p'])}"
                 f"{cs.sig(c['majority']['mcnemar_p'])} |")
    L.append("")

    L.append("### 4.2 下游结论是否被改写（Pfixed vs 参照系，对照主实验 P vs 同一参照系）\n")
    L.append("| 对比 | top-k | 主实验 P 口径 | Pfixed 口径 | n(Pfixed/P) | 方向 | "
             "效应量差 | Wilcoxon 显著性 |")
    L.append("|---|---|---|---|---|---|---|---|")
    changed, misaligned = [], []
    for opp in ("Ax1", "MDT"):
        for k in cs.METHOD_KEYS:
            c = (cons.get(opp) or {}).get(f"top{k}")
            if not c:
                continue
            nf, nm = c.get("n_cases_fixed"), c.get("n_cases_main")
            if nf != n_expected or nm != n_expected:
                misaligned.append(f"P vs {opp} top-{k}")
                L.append(f"| P vs {opp} | top-{k} | "
                         f"{c['mean_diff_main']*100:+.1f}pp | "
                         f"{c['mean_diff_fixed']*100:+.1f}pp | {nf}/{nm}"
                         f"（预期 {n_expected}）| ⚠ 样本不对齐，判定无效 | — | — |")
                continue
            ver = "同" if c["direction_same"] else "**反转**"
            eff = ("同(≤2pp)" if c["effect_within_tol"]
                   else f"**差{c['effect_delta_pp']:+.1f}pp**")
            sg = "同" if c["wilcoxon_sig_same"] else "**改变**"
            if not (c["direction_same"] and c["effect_within_tol"]
                    and c["wilcoxon_sig_same"] and c["majority_sig_same"]):
                changed.append(f"P vs {opp} top-{k}")
            L.append(f"| P vs {opp} | top-{k} | {c['mean_diff_main']*100:+.1f}pp, "
                     f"p={cs.fmt_p(c['wilcoxon_p_main'])}"
                     f"{cs.sig(c['wilcoxon_p_main'])} | "
                     f"{c['mean_diff_fixed']*100:+.1f}pp, p="
                     f"{cs.fmt_p(c['wilcoxon_p_fixed'])}"
                     f"{cs.sig(c['wilcoxon_p_fixed'])} | {nf}/{nm} | {ver} | {eff} | {sg} |")
    L.append("")
    if misaligned:
        L.append(f"⚠ **{len(misaligned)} 条对比的样本不对齐**："
                 + "；".join(misaligned) + "。这些条目的方向/效应量差不可解读。")
    if changed:
        L.append(f"方向/效应量/显著性**发生改变**的条目（{len(changed)}）："
                 + "；".join(changed))
    elif not misaligned:
        L.append("逐条比对：所有对比在 Pfixed 口径下的**方向、效应量（≤2pp）、"
                 "Wilcoxon 与多数决 McNemar 显著性**均与主实验 P 口径一致。")

    L.append("\n### 4.3 与 seed 间自身波动的比较\n")
    L.append("| top-k | 主实验 P 逐 seed 准确率 mean±SD [min,max] | "
             "Pfixed 逐 seed 准确率 mean±SD [min,max] | MDT 逐 seed | 两臂均值差 |")
    L.append("|---|---|---|---|---|")
    ids = meta["ids"]
    for k in cs.METHOD_KEYS:
        sp_m = seed_spread(meta["_arms"]["P"], ids, k)
        sp_f = seed_spread(meta["_arms"]["Pfixed"], ids, k)
        sp_d = seed_spread(meta["_arms"]["MDT"], ids, k)
        if not sp_m or not sp_f:
            continue
        L.append(f"| top-{k} | {sp_m['mean']*100:.1f} ± {sp_m['sd']*100:.1f} "
                 f"[{sp_m['min']*100:.1f},{sp_m['max']*100:.1f}] | "
                 f"{sp_f['mean']*100:.1f} ± {sp_f['sd']*100:.1f} "
                 f"[{sp_f['min']*100:.1f},{sp_f['max']*100:.1f}] | "
                 f"{sp_d['mean']*100:.1f} ± {sp_d['sd']*100:.1f} | "
                 f"{(sp_f['mean']-sp_m['mean'])*100:+.1f}pp |")
    L.append("\n（若两臂均值差落在 P 臂自身的 seed 间 SD / 极差之内，则该差异与"
             "「同一提示词重复采样」的波动同量级，不能归因于花括号修正。）\n")

    L.append("\n## 5. 结论\n")
    L.append(meta["verdict"])
    L.append("")
    L.append("## 6. 说明与接口\n")
    L.append("- 本脚本**未修改** `topn_cpc_promptv2.py`、`topn_mcr/`、"
             "`topn_mcr_seeds/` 与任何主实验结果文件；修正只作用于本进程内的后缀"
             "副本，主实验口径不受污染。")
    L.append("- 判分阶段走 `caselevel_stats.judge_missing(rows, cache_path=...)`，"
             "只补缺失对；缓存路径必须由 `PFIXED_MCR_JUDGE_CACHE`（或 `GLM_CACHE`）"
             "显式指定，以免误写共享的 `judge_cache_glm_v3.json`。"
             "该缓存是全量覆盖写，跨进程并发跑 judge 会互相覆盖 —— 判分须串行调度。")
    L.append(f"- 生成脚本 `routing_study/scripts/p_fixed_mcr.py`；命令："
             f"`PHASE={meta['phase']} ./.venv/bin/python "
             f"routing_study/scripts/p_fixed_mcr.py`")
    return "\n".join(L) + "\n"


def render_verdict(res, cons, flip, readiness, n_expected, tol_pp=2.0):
    """程序化给出「MDT 是否仍显著优于 P」的判定句。"""

    def pk(pair, k):
        return (res["comparisons"].get(pair) or {}).get(f"top{k}") or {}

    if not readiness:
        return ("**答：暂不能判定。**数据未就绪（某臂缺行、样本数不足 "
                + str(n_expected) + " 例，或缺判官判定）。"
                "补齐 `PHASE=infer` 与 `PHASE=judge`（并正确指定判分缓存路径）后"
                "重跑 `PHASE=analyze`。在此之前请勿引用本文件中的数字。")
    blocks = []
    # 1) 两臂正面比
    diff_bits = []
    for k in cs.METHOD_KEYS:
        c = pk("Pfixed_vs_P", k)
        if not c or c["mean_rate_a"] is None:
            continue
        lo, hi = c["boot95_ci"]
        diff_bits.append(f"top-{k} {c['mean_diff']*100:+.1f}pp "
                         f"[{lo*100:+.1f}, {hi*100:+.1f}] p="
                         f"{cs.fmt_p(c['wilcoxon_p'])}")
    blocks.append("Pfixed 与主实验 P 的正面比：" + "；".join(diff_bits) + "。")
    # 2) headline：MDT vs P 是否稳健
    parts = []
    for k in cs.METHOD_KEYS:
        m = pk("MDT_vs_P", k)
        f = flip.get(k)
        if not m or not f or m["mean_rate_a"] is None:
            continue
        lo_m, hi_m = m["boot95_ci"]
        lo_f, hi_f = f["boot95_ci"]
        sm, sf = md_is_sig(m["wilcoxon_p"]), md_is_sig(f["wilcoxon_p"])
        parts.append(
            f"top-{k}：MDT {m['mean_rate_a']*100:.1f}% vs P "
            f"{m['mean_rate_b']*100:.1f}%（{m['mean_diff']*100:+.1f}pp "
            f"[{lo_m*100:+.1f}, {hi_m*100:+.1f}], p={cs.fmt_p(m['wilcoxon_p'])}"
            f"{'*' if sm else ''}）→ vs Pfixed {f['mean_rate_a']*100:.1f}% vs "
            f"{f['mean_rate_b']*100:.1f}%（{f['mean_diff']*100:+.1f}pp "
            f"[{lo_f*100:+.1f}, {hi_f*100:+.1f}], p={cs.fmt_p(f['wilcoxon_p'])}"
            f"{'*' if sf else ''}），显著性{'不变' if sm == sf else '改变'}")
    n_sig_m = sum(1 for k in cs.METHOD_KEYS
                  if md_is_sig(pk("MDT_vs_P", k).get("wilcoxon_p")))
    n_sig_f = sum(1 for k in cs.METHOD_KEYS
                  if md_is_sig((flip.get(k) or {}).get("wilcoxon_p")))
    nk = len(cs.METHOD_KEYS)
    if n_sig_f >= n_sig_m and n_sig_f > 0:
        answer = (f"**答：是，仍显著。**修正花括号后 MDT 相对 P 的优势仍在 "
                  f"{n_sig_f}/{nk} 个 top-k 上达到 p<0.05"
                  + ("（与主实验口径相同）" if n_sig_f == n_sig_m
                     else f"（主实验口径为 {n_sig_m}/{nk}）") + "。")
    elif n_sig_f == 0 and n_sig_m > 0:
        answer = (f"**答：不再显著。**主实验口径下 MDT 优于 P 的差异在 "
                  f"{n_sig_m}/{nk} 个 top-k 上显著，修正花括号后全部失去显著性"
                  f" → 主实验的「MDT 优于 P」表述在 MCR 上对提示词缺陷不稳健，"
                  f"需按 Pfixed 口径弱化。")
    elif n_sig_m == 0:
        answer = ("**答：主实验口径下 MCR 上 MDT 与 P 的差异本就不显著**"
                  + (f"（修正后 {n_sig_f}/{nk} 个 top-k 变为显著）"
                     if n_sig_f else "")
                  + "，故本对比在 MCR 上不构成「MDT 优于 P」的证据。")
    else:
        answer = (f"**答：部分稳健。**MDT 优于 P 的显著 top-k 个数从 {n_sig_m}/{nk} "
                  f"变为 {n_sig_f}/{nk}，需逐条看下列数字。")
    blocks.append(answer)
    blocks.append(f"口径对照 × {nk} 个 top-k：" + "；".join(parts) + "。")
    # 3) 下游其它对比
    flipped, eff_only = [], []
    for opp in ("Ax1", "MDT"):
        for k in cs.METHOD_KEYS:
            c = (cons.get(opp) or {}).get(f"top{k}")
            if not c or c["n_cases_fixed"] != n_expected:
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
                flipped.append(f"P vs {opp} top-{k}（" + "、".join(detail) + "）")
            elif not c["effect_within_tol"]:
                eff_only.append(f"P vs {opp} top-{k}（{c['effect_delta_pp']:+.1f}pp）")
    if flipped:
        blocks.append("P 与其它臂的对比中被改写：" + "；".join(flipped)
                      + "，需按 Pfixed 口径修正表述。")
    elif eff_only:
        blocks.append("P vs A×1 / MDT 的**方向与显著性状态全部一致**，仅效应量差"
                      "超过 " + str(tol_pp) + "pp：" + "；".join(eff_only)
                      + "（对照 §4.3 的 seed 间波动判断该差异是否超出采样噪声）。")
    else:
        blocks.append("P vs A×1、P vs MDT 在方向、效应量（≤2pp）与显著性上"
                      "全部与主实验 P 口径一致。")
    return "\n\n".join(blocks)


def run_analyze():
    cases = load_cases()
    audit = prompt_audit(cases)
    print(f"[审计] 逐例 diff：{audit['n_single_line_diff']}/{audit['n_cases']} 例"
          f"恰有 1 行不同（花括号折叠），异常 {audit['n_unexpected']}", flush=True)
    if audit["unexpected"]:
        raise SystemExit(f"提示词审计失败：{audit['unexpected']}")

    arms = build_arms(SEEDS)
    ids = [c["case_id"] for c in cases]
    n_expected = len(ids)
    cache_path, var = resolve_judge_cache()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if not cache_path.exists():
        print(f"[分析] 警告：判分缓存 {cache_path} 不存在，全部按缺失计", flush=True)
    cs.set_cache(cache)
    stats = cs.split_stats(arms, {SPLIT: ids}, PAIRS)
    res = stats[SPLIT]

    readiness, blockers = True, []
    for arm, runs in arms.items():
        for s, rows in runs.items():
            cov = sum(1 for cid in ids if cid in rows)
            if cov != n_expected:
                blockers.append(f"{arm}/s{s} 覆盖 {cov}/{n_expected}")
                readiness = False
    # 各臂判分缺失量（槽位 + 去重对）：judge 阶段的工作量与"该用哪个缓存"的依据
    judge_missing_by_arm = {}
    for arm, runs in arms.items():
        slots, pairs = 0, set()
        for rows in runs.values():
            for r in rows.values():
                for c in r["top5"][:5]:
                    k = cs.key_of(r["gold"], c)
                    if k not in cache:
                        slots += 1
                        pairs.add(k)
        judge_missing_by_arm[arm] = {"slots": slots, "pairs": len(pairs)}
    for pair in (f"Pfixed_vs_{o}" for o in ("P", "Ax1", "MDT")):
        for k in cs.METHOD_KEYS:
            c = (res["comparisons"].get(pair) or {}).get(f"top{k}")
            if not c or c["mean_rate_a"] is None or c["n_cases"] != n_expected:
                blockers.append(f"{pair}/top{k} 样本不对齐")
                readiness = False
    if blockers:
        print(f"[分析] ⚠ 数据不完备（{len(blockers)} 项，节选 {blockers[:5]}）"
              f"—— §5 结论将标注为不可判定", flush=True)

    cons = {opp: {f"top{k}": cmp_consistency(
        (res["comparisons"].get(f"Pfixed_vs_{opp}") or {}).get(f"top{k}"),
        (res["comparisons"].get(f"P_vs_{opp}") or {}).get(f"top{k}"))
        for k in cs.METHOD_KEYS} for opp in ("Ax1", "MDT")}
    flip = {k: flip_cmp((res["comparisons"].get("Pfixed_vs_MDT") or {}).get(f"top{k}"))
            for k in cs.METHOD_KEYS}

    parse_stats = json.loads((OUTDIR / "parse_stats.json").read_text()) \
        if (OUTDIR / "parse_stats.json").exists() else {
            "per_seed": {}, "hard_failures": [], "n_calls_planned": n_expected * len(SEEDS)}
    meta = {
        "phase": os.environ.get("PHASE", "all"),
        "provider": resolve_provider("qwen"), "model": QWEN_MODEL,
        "temperature": TEMPERATURE, "disable_thinking": True, "max_tokens": 4096,
        "timeout": 300, "call": "topn_mcr.call_top5",
        "n_cases": n_expected, "seeds": SEEDS,
        "n_calls_planned": parse_stats.get("n_calls_planned"),
        "outdir": str(OUTDIR), "s1_dir": str(S1_DIR), "seeds_dir": str(SEEDS_DIR),
        "judge_cache": str(cache_path), "judge_cache_var": var,
        "judge_model": cs.GLM_MODEL, "judge_cache_size": len(cache),
        "judge_cache_added_by_this_script": sum(
            1 for s in SEEDS for r in load_pfixed(s).values()
            for c in r["top5"][:5] if cs.key_of(r["gold"], c) not in cache),
        "parse_stats": parse_stats,
        "stats": stats, "ids": ids, "_arms": arms,
        "readiness": readiness, "readiness_blockers": blockers,
        "judge_missing_by_arm": judge_missing_by_arm,
        "consistency": cons, "prompt_audit": audit,
        "prompt": "PERSPECTIVE_PROMPT.format(structured_case=...) + P_FIXED_SUFFIX"
                  "（P_FIXED_SUFFIX = P_TOPN_SUFFIX 折叠 {{ }}）",
    }
    if res["judge_missing_pairs"]:
        print(f"[分析] 警告：{res['judge_missing_pairs']} 个 (病例×seed×候选) "
              f"判官缺失，相关病例已从病例级统计中剔除（先跑 PHASE=judge）",
              flush=True)
    meta["verdict"] = render_verdict(res, cons, flip, readiness, n_expected)
    md = render_md(meta, audit, stats, cons, n_expected)
    payload = {k: v for k, v in meta.items() if k != "_arms"}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"\n已写入 {OUT_JSON}\n已写入 {OUT_MD}", flush=True)


# ---------- 冒烟 ----------

def smoke():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    example = self_check()
    cases = load_cases()[:3]
    audit = prompt_audit(cases, sample_n=3)
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
        print(f"[冒烟] diff（{s['case_id'][:30]} 第 {s['line_no']}/{s['total_lines']} "
              f"行，{s['char_delta']} 字符）\n  main : {s['main']}\n"
              f"  fixed: {s['fixed']}", flush=True)
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
        raise SystemExit("PFIXED_MCR_SEEDS 为空")
    phase = os.environ.get("PHASE", "all").lower()
    print(f"seeds: {SEEDS} | phase: {phase} | outdir: {OUTDIR}", flush=True)

    if phase in ("all", "infer"):
        cases = load_cases()
        run_infer(cases)
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
            print(f"[检查] 上述病例已连续 {MAX_ATTEMPTS} 轮拿不到有效 top5。"
                  "直接重跑同一条命令即可续跑补齐（已有行自动跳过）：\n"
                  "  PHASE=infer ./.venv/bin/python "
                  "routing_study/scripts/p_fixed_mcr.py", flush=True)
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
