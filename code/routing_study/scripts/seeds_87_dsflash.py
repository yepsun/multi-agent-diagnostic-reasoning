#!/usr/bin/env python3
"""第二模型族（deepseek-flash）在 CPC 上的复现：A×1 / P，87 例 × 3 seeds。

回应评审 (b)：主结果只来自 qwen3.8-flash 一个模型族，需第二个模型族复现。
本脚本镜像 seeds_87.py，但把 A×1 与 P 的推理 provider 换成 deepseek-flash
（topn_cpc.call_top5 把 provider 写死为 qwen，此处不改动它，而是在本脚本内
实现一个带 provider 参数的等价 call_top5：提示词/参数/解析逻辑与
topn_cpc.py:97-100 逐字一致，仅 provider 不同）。

- 提示词：topn_cpc_promptv2.A_TOPN_PROMPT（A×1，T=0）与
  PERSPECTIVE_PROMPT + P_TOPN_SUFFIX（P，T=0.3）——与主实验逐字一致。
- 病例文本：topn_cpc_promptv2_87.load_merged()（87 例 merged）。
- 推理：provider="deepseek-flash"，DEEPSEEK_MODEL=deepseek-flash，
  disable_thinking=True，max_tokens=2048。

**解析层兜底（brace repair）**：`topn_cpc_promptv2.P_TOPN_SUFFIX` 是一段
从未经过 `.format()` 的普通字符串，其 JSON 块写作字面双花括号
（`{{"top5": [{{"rank": 1, ...}}]}}`），而所有调用方都是
`PERSPECTIVE_PROMPT.format(...) + P_TOPN_SUFFIX`，所以实际发给模型的 P 提示词
里带的就是双花括号（A_TOPN_PROMPT 经 `.format()` 折叠过，所以 A×1 不受影响）。
qwen 侧会自行改写成单花括号，deepseek-flash 侧则原样照抄，导致主解析器
`extract_json` 解析失败 → top5 为空。
本脚本因此加一层**解析兜底**：先用既有 `topn_cpc.parse_top5` 解析，拿不到
5 项时，把成对转义残留 `{{`→`{`、`}}`→`}` 归一化后再解析一次；命中兜底的行
标记 `"brace_repaired": true`。**这只是解析层修复**：提示词与主实验逐字节
相同（两个模型族因此严格可比），诊断内容也未被改写，只让照抄了转义花括号的
模型输出可被解析。保守性：兜底只在主解析不足 5 项时启用，且归一化后必须
解析出长度恰为 5 的 `top5` 列表（既有 schema），否则仍按失败计。

输出：results/topn_seeds_dsflash/{Ax1,P}_s{1,2,3}.jsonl，断点续跑（缺行或
不足 5 项的行会被补采）。
MDT（deepseek-flash）走既有 mdt_cpc.py，不改代码：
  MDT_OUTDIR=topn_mdt_dsflash MDT_SEED={1,2,3} \\
  MDT_ROLE_PROVIDERS=deepseek-flash,deepseek-flash,deepseek-flash,deepseek-flash,deepseek-flash \\
  MDT_SYNTH_PROVIDER=deepseek-flash MDT_SKIP_JUDGE=1 \\
  ./.venv/bin/python routing_study/scripts/mdt_cpc.py
  产出 results/topn_mdt_dsflash/synthesis.jsonl（s1）与 s{2,3}/synthesis.jsonl。

阶段：PHASE=infer|judge|analyze|all（默认 all）。judge 与 analyze 需在
MDT 三条命令跑完后执行（analyze 还要等 judge 完成）。
  judge   → GLM-5.3-flash × judge_v3.V3_PROMPT，共享缓存
            results/judge_cache_glm_v3.json，只补缺失对。
  analyze → results/dsflash_family_cpc.json + .md：deepseek-flash 族 A×1/P/MDT 的
            3-seed 均值±SD 与病例级配对检验（Wilcoxon + cluster bootstrap
            10000 次 95% CI + 多数决精确 McNemar，口径同 stats_caselevel.py），
            附 qwen 族同指标作为跨族对照；给 held-out 46 / dev 41 子集。

环境变量：DSFLASH_OUTDIR（默认 results/topn_seeds_dsflash）、
DSFLASH_SEEDS（默认 1,2,3）、DSFLASH_LIMIT（只跑前 N 例，冒烟用）、
DSFLASH_MDT_OUTDIR（默认 results/topn_mdt_dsflash）、GLM_CACHE。
"""
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
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

import caselevel_stats as cs  # noqa: E402
from run_inference import call_llm  # noqa: E402
from topn_cpc import (load_done, append_row, parse_top5,  # noqa: E402
                      MAX_WORKERS)
from topn_cpc_promptv2 import A_TOPN_PROMPT, P_TOPN_SUFFIX  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402

RESULTS = ROOT / "routing_study" / "results"
OUTDIR = Path(os.environ.get("DSFLASH_OUTDIR", RESULTS / "topn_seeds_dsflash"))
MDT_OUTDIR = Path(os.environ.get("DSFLASH_MDT_OUTDIR",
                                 RESULTS / "topn_mdt_dsflash"))
QWEN_SEEDS = RESULTS / "topn_seeds"
QWEN_MDT = RESULTS / "topn_mdt"
SEEDS = [int(s) for s in os.environ.get("DSFLASH_SEEDS", "1,2,3").split(",")
         if s.strip()]
LIMIT = int(os.environ.get("DSFLASH_LIMIT", "0"))
PROVIDER = "deepseek-flash"
TIMEOUT = 300
MAX_TOKENS = 2048
MAX_ATTEMPTS = 3
OUT_JSON = RESULTS / "dsflash_family_cpc.json"
OUT_MD = RESULTS / "dsflash_family_cpc.md"
PAIRS = [("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1"),
         ("MDT_qwen", "MDT"), ("Ax1_qwen", "Ax1"), ("P_qwen", "P")]
ARM_LABEL = {"Ax1": "A×1 (deepseek-flash)", "P": "P (deepseek-flash)",
             "MDT": "MDT (deepseek-flash)",
             "Ax1_qwen": "A×1 (qwen3.8-flash)", "P_qwen": "P (qwen3.8-flash)",
             "MDT_qwen": "MDT (qwen3.8-flash)"}


def row_path(scheme, seed):
    return OUTDIR / f"{scheme}_s{seed}.jsonl"


# ---------- 推理（provider 参数化；其余与 topn_cpc.call_top5 逐字一致） ----------

def parse_top5_repair(raw):
    """主解析（既有 topn_cpc.parse_top5）+ 双花括号归一化兜底。

    返回 (top5, repaired)。repaired=True 表示主解析拿不到 5 项、经
    `{{`→`{`、`}}`→`}` 归一化后重解析才得到完整的 5 项。

    保守性：(1) 只在主解析结果不是 5 项时启用；(2) 归一化后必须解析出
    去重后长度恰为 5 的 top5（既有 schema），否则返回主解析结果、按失败计；
    (3) 不改提示词、不改写诊断文本，只做字符串级归一化。"""
    top5 = parse_top5(raw)
    if len(top5) == 5 or not raw:
        return top5, False
    fixed = raw.replace("{{", "{").replace("}}", "}")
    if fixed == raw:
        return top5, False
    top5_fixed = parse_top5(fixed)
    if len(top5_fixed) == 5:
        return top5_fixed, True
    return top5, False


def call_top5_raw(prompt, temperature, provider=PROVIDER):
    """调用参数与 topn_cpc.call_top5 逐字一致（仅 provider 可换），
    额外回传原始响应与是否命中解析兜底。"""
    raw, usage = call_llm(prompt, temperature=temperature, max_tokens=MAX_TOKENS,
                          timeout=TIMEOUT, provider=provider,
                          disable_thinking=True)
    top5, repaired = parse_top5_repair(raw)
    return top5, (usage or {}).get("total_tokens", 0), repaired, raw


def row_complete(row):
    """行是否已完成：存在、且 top5 为 5 项（否则需补采）。"""
    return bool(row) and len(row.get("top5") or []) == 5


def compact(path):
    """补采同一 case 会追加多行；按 case_id 去重保留最后一行，
    顺序按首次出现，`.tmp` + `os.replace` 原子写回。无重复时不改文件。"""
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
    print(f"[整理] {path.name}: {len(rows)} 行 → {len(order)} 行"
          f"（同一 case 补采后取最后一行）", flush=True)


# 失败与兜底记录（供阶段末尾汇总；进程内有效）
FAILURES = []
REPAIRED = []


def run_one(scheme, seed, cases):
    path = row_path(scheme, seed)
    done = load_done(path)
    todo = [c for c in cases if not row_complete(done.get(c["case_id"]))]
    n_partial = sum(1 for c in cases if c["case_id"] in done
                    and not row_complete(done[c["case_id"]]))
    print(f"[{scheme} s{seed} | {PROVIDER}] 已完成 "
          f"{len(cases) - len(todo)}/{len(cases)}（其中不足 5 项待补采 "
          f"{n_partial}），待跑 {len(todo)}", flush=True)

    def work(c):
        if scheme == "Ax1":
            prompt = A_TOPN_PROMPT.format(case_text=c["text"])
            temperature = 0.0
        else:
            prompt = PERSPECTIVE_PROMPT.format(
                structured_case=c["text"]) + P_TOPN_SUFFIX
            temperature = 0.3
        last_raw = ""
        for attempt in range(MAX_ATTEMPTS):
            top5, tokens, repaired, raw = call_top5_raw(prompt, temperature)
            last_raw = raw or ""
            if top5:
                if repaired:
                    snippet = raw.strip()[:200].replace("\n", " ")
                    REPAIRED.append({"scheme": scheme, "seed": seed,
                                     "case_id": c["case_id"],
                                     "raw_head": snippet})
                    print(f"[兜底] {scheme} s{seed} {c['case_id'][:40]}: "
                          f"双花括号归一化后解析成功 | 原始片段: {snippet!r}",
                          flush=True)
                return {"case_id": c["case_id"], "gold": c["gold"],
                        "top5": top5, "total_tokens": tokens,
                        "brace_repaired": repaired}
            print(f"[{scheme} s{seed}] {c['case_id'][:40]} 空输出，"
                  f"重试 {attempt + 1}/{MAX_ATTEMPTS}", flush=True)
        raise RuntimeError(f"{c['case_id']} 连续空输出 | 末次响应片段: "
                           f"{last_raw.strip()[:400]!r}")

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                FAILURES.append({"scheme": scheme, "seed": seed,
                                 "case_id": c["case_id"], "error": str(e)})
                print(f"[失败] {scheme} s{seed} {c['case_id'][:40]}: {e}",
                      flush=True)
                continue
            append_row(path, row)
            print(f"[{scheme} s{seed}] {i}/{len(todo)} {row['case_id'][:40]}: "
                  f"{row['top5'][:1]}", flush=True)
    compact(path)


# ---------- 判分（GLM v3，只补缺失对） ----------

def load_arm(scheme, seed, outdir=None):
    return cs.load((outdir or OUTDIR) / f"{scheme}_s{seed}.jsonl")


def run_judge():
    rows = []
    for seed in SEEDS:
        for scheme in ("Ax1", "P"):
            rows.extend(load_arm(scheme, seed).values())
        rows.extend(mdt_rows(seed).values())
    n_before = len(json.loads(cs.GLM_CACHE.read_text())) if cs.GLM_CACHE.exists() else 0
    cache = cs.judge_missing(rows)
    print(f"[判定] 缓存 {n_before} → {len(cache)}（新增 {len(cache) - n_before} 对）",
          flush=True)
    return cache


# ---------- 分析 ----------

def mdt_rows(seed, outdir=None):
    base = outdir or MDT_OUTDIR
    return cs.load(base / ("synthesis.jsonl" if seed == 1
                           else f"s{seed}/synthesis.jsonl"))


def build_arms():
    arms = {
        "Ax1": {s: load_arm("Ax1", s) for s in SEEDS},
        "P": {s: load_arm("P", s) for s in SEEDS},
        "MDT": {s: mdt_rows(s) for s in SEEDS},
        "Ax1_qwen": {s: cs.load(QWEN_SEEDS / f"Ax1_s{s}.jsonl") for s in SEEDS},
        "P_qwen": {s: cs.load(QWEN_SEEDS / f"P_s{s}.jsonl") for s in SEEDS},
        "MDT_qwen": {s: cs.load(QWEN_MDT
                                / ("synthesis.jsonl" if s == 1
                                   else f"s{s}/synthesis.jsonl"))
                     for s in SEEDS},
    }
    return arms


def render_md(meta, stats, splits_order):
    L = []
    L.append("# 第二模型族（deepseek-flash）CPC 复现：A×1 / P / MDT\n")
    L.append(f"回应评审 (b)：主结果只来自 qwen3.8-flash 一个模型族。本表用 "
             f"deepseek-flash 重跑 CPC 的 A×1 / P（87 例 × {meta['n_seeds']} seeds）与 MDT "
             "（五角色 + 主持人，同族 deepseek-flash），并用主判官（GLM-5.3-flash × "
             "v3 规则，共享缓存 `results/judge_cache_glm_v3.json`）计分。\n")
    L.append(f"- 推理 provider：`{meta['provider']}`，模型 id `"
             f"{meta['model']}`（端点 `{meta['endpoint']}`）；A×1 T=0、"
             f"P T=0.3、MDT T=0.3，disable_thinking=True，max_tokens=2048/4096")
    L.append(f"- 提示词与主实验逐字一致：A×1 = `topn_cpc_promptv2.A_TOPN_PROMPT`；"
             f"P = `PERSPECTIVE_PROMPT + P_TOPN_SUFFIX`；病例文本 "
             f"`load_merged()`（87 例）")
    L.append(f"- 推理调用数：A×1/P 共 {meta['n_calls']} 次（{meta['n_cases']} 例 × "
             f"{meta['n_seeds']} seeds × 2 方案）；MDT 走既有 `mdt_cpc.py`，"
             f"每 seed {meta['n_cases']}×6 次")
    L.append(f"- 判官缺失对：{ {k: v['judge_missing_pairs'] for k, v in stats.items()} }")
    L.append(f"- 生成脚本 `routing_study/scripts/seeds_87_dsflash.py`；"
             f"输出 `{meta['outdir']}/{{Ax1,P}}_s{{1..{meta['n_seeds']}}}.jsonl`\n")

    for name in splits_order:
        res = stats[name]
        L.append(f"\n## {name}（n={res['n_cases']} 病例）\n")
        L.append("| 方案 | top-1 | top-3 | top-5 | 病例级命中率 top-1/3/5 |")
        L.append("|---|---|---|---|---|")
        for arm in ARM_LABEL:
            if arm not in res["arms"]:
                continue
            a = res["arms"][arm]
            cr = a["case_rate_mean"]
            crm = "/".join(f"{cr[f'top{k}']*100:.1f}%"
                           if cr[f"top{k}"] is not None else "NA"
                           for k in (1, 3, 5))
            L.append(f"| {ARM_LABEL[arm]} | {cs.acc_cell(a['mean_sd'], 1)} | "
                     f"{cs.acc_cell(a['mean_sd'], 3)} | "
                     f"{cs.acc_cell(a['mean_sd'], 5)} | {crm} |")
        L.append(f"\n（百分比为 {meta['n_seeds']} seeds 准确率均值 ± SD；末列为病例级 {meta['n_seeds']}-seed 均值）\n")

        L.append("| 对比 | top-k | 命中率 A vs B | 均值差 [95% CI] | Wilcoxon p | "
                 "多数决 McNemar (a:b) p | 旧:合并 McNemar (a:b) p |")
        L.append("|---|---|---|---|---|---|---|")
        for pair, by_k in res["comparisons"].items():
            a, b = pair.split("_vs_")
            for k in (1, 3, 5):
                c = by_k[f"top{k}"]
                if c["mean_rate_a"] is None:
                    continue
                lo, hi = c["boot95_ci"]
                maj, old = c["majority"], c["pooled_mcnemar"]
                L.append(
                    f"| {ARM_LABEL[a]} vs {ARM_LABEL[b]} | top-{k} | "
                    f"{c['mean_rate_a']*100:.1f}% vs {c['mean_rate_b']*100:.1f}% | "
                    f"{c['mean_diff']*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}] | "
                    f"{cs.fmt_p(c['wilcoxon_p'])}{cs.sig(c['wilcoxon_p'])} | "
                    f"{maj['a_only']}:{maj['b_only']} p={cs.fmt_p(maj['mcnemar_p'])}"
                    f"{cs.sig(maj['mcnemar_p'])} | "
                    f"{old['a_only']}:{old['b_only']} p={cs.fmt_p(old['p'])}"
                    f"{cs.sig(old['p'])} |")
        L.append("")

    n_rep = meta["brace_repaired"]
    L.append("\n## 解析兜底（brace repair）说明\n")
    L.append("`topn_cpc_promptv2.P_TOPN_SUFFIX` 是一段从未经过 `.format()` 的普通"
             '字符串，其 JSON 块写作**字面双花括号** `{{"top5": [{{"rank": 1, ...}}]}}`；'
             "所有调用方都是 `PERSPECTIVE_PROMPT.format(...) + P_TOPN_SUFFIX`，"
             "所以实际发给模型的 P 提示词里带的就是双花括号（A_TOPN_PROMPT 经 "
             "`.format()` 折叠，故 A×1 不受影响）。qwen 侧会自行改写成单花括号，"
             "deepseek-flash 侧会原样照抄 → 主解析器 `extract_json` 失败 → top5 为空。\n")
    L.append(f"本实验**不修改提示词**（deepseek-flash 臂与主实验逐字节相同，两个模型族"
             f"因此严格可比），只在解析层加兜底：主解析拿不到 5 项时，把成对转义"
             f"残留 `{{{{`→`{{`、`}}}}`→`}}` 归一化后重解析，且必须解析出长度恰为 5 "
             f"的 `top5` 才接受；命中行标记 `brace_repaired: true`。诊断内容未被"
             f"改写，仅使照抄转义花括号的输出可被解析。\n")
    L.append(f"- 本次命中兜底的行数：A×1 {n_rep['Ax1']} 行、P {n_rep['P']} 行"
             f"（各 seed 明细：{n_rep['by_seed']}）")
    L.append(f"- 逐字节实际发出的 P 提示词见 "
             f"`routing_study/results/prompt_actually_sent.md`\n")

    full = stats.get("full87")
    if full:
        def line(pair, k):
            c = full["comparisons"].get(pair, {}).get(f"top{k}")
            if c is None or c["mean_rate_a"] is None:
                return f"{pair} top-{k}: 数据不足"
            return (f"{pair} top-{k}: {c['mean_rate_a']*100:.1f}% vs "
                    f"{c['mean_rate_b']*100:.1f}%（差 {c['mean_diff']*100:+.1f}pp, "
                    f"Wilcoxon p={cs.fmt_p(c['wilcoxon_p'])}, 多数决 McNemar "
                    f"p={cs.fmt_p(c['majority']['mcnemar_p'])}）")
        L.append(f"\n## 结论（deepseek-flash 族复现，全量 87 例，{meta['n_seeds']} seeds）\n")
        for pair in ("MDT_vs_Ax1", "MDT_vs_P", "P_vs_Ax1"):
            L.append(f"- 族内：{line(pair, 1)}；{line(pair, 5)}")
        L.append("- 跨族一致性（qwen 族 vs deepseek-flash 族，同架构同提示词）：")
        for pair in ("Ax1_qwen_vs_Ax1", "P_qwen_vs_P", "MDT_qwen_vs_MDT"):
            L.append(f"  - {line(pair, 1)}")
            L.append(f"  - {line(pair, 5)}")
        c3 = full["comparisons"].get("MDT_vs_Ax1", {}).get("top3")
        c5 = full["comparisons"].get("MDT_vs_Ax1", {}).get("top5")
        keep = any(c is not None and c["mean_rate_a"] is not None
                   and c["wilcoxon_p"] is not None and c["wilcoxon_p"] < 0.05
                   for c in (c3, c5))
        L.append(f"\n**判定**：deepseek-flash 族上 MDT 相对 A×1 的 top-3/5 优势"
                 f"{'仍然显著 → 主结论跨模型族复现。' if keep else '不再显著 → 需按此口径弱化表述。'}")
    return "\n".join(L) + "\n"


def run_analyze():
    arms = build_arms()
    cache = json.loads(cs.GLM_CACHE.read_text())
    cs.set_cache(cache)
    splits = cs.split_ids()
    if LIMIT:
        present = {r["case_id"] for r in arms["Ax1"][SEEDS[0]].values()}
        splits = {k: [c for c in v if c in present] for k, v in splits.items()}
    stats = cs.split_stats(arms, splits, PAIRS)
    rep = {"Ax1": 0, "P": 0, "by_seed": {}}
    for seed in SEEDS:
        for scheme in ("Ax1", "P"):
            n = sum(1 for r in load_arm(scheme, seed).values()
                    if r.get("brace_repaired"))
            rep[scheme] += n
            rep["by_seed"][f"{scheme}_s{seed}"] = n
    meta = {"provider": PROVIDER, "model": os.environ["DEEPSEEK_MODEL"],
            "endpoint": os.environ.get("DEEPSEEK_API_BASE",
                                       "https://api.deepseek.com/v1"),
            "n_cases": len(arms["Ax1"][SEEDS[0]]), "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "n_calls": len(arms["Ax1"][SEEDS[0]]) * len(SEEDS) * 2,
            "outdir": str(OUTDIR), "mdt_outdir": str(MDT_OUTDIR),
            "judge_cache": str(cs.GLM_CACHE), "judge_model": cs.GLM_MODEL,
            "brace_repaired": rep,
            "prompt": "topn_cpc_promptv2.A_TOPN_PROMPT / PERSPECTIVE_PROMPT+P_TOPN_SUFFIX"}
    OUT_JSON.write_text(json.dumps({"meta": meta, "splits": stats},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    md = render_md(meta, stats, ["full87", "heldout46", "dev41"])
    OUT_MD.write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"已写入 {OUT_JSON} 与 {OUT_MD}", flush=True)


def main():
    phase = os.environ.get("PHASE", "all").lower()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"数据集: {len(cases)} 例 | seeds {SEEDS} | provider {PROVIDER} | "
          f"模型 {os.environ['DEEPSEEK_MODEL']} | 输出: {OUTDIR} | "
          f"MDT 目录: {MDT_OUTDIR}", flush=True)

    if phase in ("all", "infer"):
        t0 = time.time()
        for seed in SEEDS:
            for scheme in ("Ax1", "P"):
                run_one(scheme, seed, cases)
        # 完整性汇总（先打印全部统计，再对缺行/不足 5 项报警）
        print(f"\n===== 补齐结果（{time.time() - t0:.0f}s）=====", flush=True)
        problems = []
        for scheme in ("Ax1", "P"):
            for seed in SEEDS:
                p = row_path(scheme, seed)
                rows = load_arm(scheme, seed)
                nlines = sum(1 for l in open(p) if l.strip()) if p.exists() else 0
                n_rep = sum(1 for r in rows.values() if r.get("brace_repaired"))
                short = [c for c, r in rows.items() if len(r["top5"]) != 5]
                miss = [c["case_id"] for c in cases if c["case_id"] not in rows]
                print(f"{scheme}_s{seed}: 行数 {nlines} | 唯一 case {len(rows)}/"
                      f"{len(cases)} | 兜底救回 {n_rep} | 不足5项 {len(short)} | "
                      f"缺行 {len(miss)}", flush=True)
                if miss:
                    problems.append(f"{scheme}_s{seed} 缺 {len(miss)} 行: {miss}")
                if short:
                    problems.append(f"{scheme}_s{seed} 不足 5 项 {len(short)} 行: "
                                    f"{short}")
        if FAILURES:
            print(f"\n[失败] {len(FAILURES)} 行（含末次原始响应片段）：", flush=True)
            for f in FAILURES:
                print(f"  - {f['scheme']}_s{f['seed']} {f['case_id']}\n"
                      f"    {f['error']}", flush=True)
        if REPAIRED:
            print(f"\n[兜底] 共救回 {len(REPAIRED)} 行：", flush=True)
            for r in REPAIRED:
                print(f"  - {r['scheme']}_s{r['seed']} {r['case_id']}", flush=True)
        if problems:
            print("\n[警告] 仍不完整：\n  " + "\n  ".join(problems), flush=True)
            print("[警告] 推理阶段以退出码 1 结束（数据不完整，"
                  "judge/analyze 请勿在此状态下进行）", flush=True)
            raise SystemExit(1)
        print("[检查] 全部 seed × 方案齐 87 行、top5 均为 5 项", flush=True)

    if phase in ("all", "judge"):
        run_judge()

    if phase in ("all", "analyze"):
        run_analyze()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
