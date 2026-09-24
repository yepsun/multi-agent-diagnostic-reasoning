#!/usr/bin/env python3
"""87 例 MGH CPC（merged = 41 旧 + 46 新）：v2 提示词 A×1 / P 的 top-5 运行。

复用 topn_ablation_promptv2 已跑的 41 例（同提示词同参数）作为种子，
仅新跑 46 例。判定复用共享 judge_cache。最后输出 87 例全量指标 +
41 例子集与基线的翻转对比。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from topn_cpc import load_done, call_top5, run_judge_phase  # noqa: E402
from topn_cpc_promptv2 import (A_TOPN_PROMPT, P_TOPN_SUFFIX,  # noqa: E402
                               run_scheme, summarize, compare_with_baseline)
from case_extraction import format_structured_case, preprocess_case_text  # noqa: E402
from scheme_perspective import PERSPECTIVE_PROMPT  # noqa: E402
from run_static_routing import _gold_text  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_ablation_promptv2_87"
MERGED = ROOT / "data" / "mgh_qa_dataset_merged.json"


def load_merged():
    cases = json.loads(MERGED.read_text())
    return [{"case_id": c["case_id"],
             "text": format_structured_case(preprocess_case_text(c.get("Q", ""))),
             "gold": _gold_text(c)} for c in cases]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    print(f"数据集: {len(cases)} 例", flush=True)

    run_scheme("A×1", OUTDIR / "ax1.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5(A_TOPN_PROMPT.format(
                                             case_text=c["text"]),
                                             temperature=0.0)))}),
               cases)
    run_scheme("P", OUTDIR / "p.jsonl",
               lambda c: (c, {"case_id": c["case_id"], "gold": c["gold"],
                              **dict(zip(("top5", "total_tokens"),
                                         call_top5(
                                             PERSPECTIVE_PROMPT.format(
                                                 structured_case=c["text"])
                                             + P_TOPN_SUFFIX,
                                             temperature=0.3)))}),
               cases)

    scheme_rows = {
        "Ax1": list(load_done(OUTDIR / "ax1.jsonl").values()),
        "P": list(load_done(OUTDIR / "p.jsonl").values()),
    }
    hits_fn = run_judge_phase(cases, scheme_rows)
    summary = summarize(scheme_rows, hits_fn)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 41 例子集与基线（promptv2 之前的基线 41 例运行）对比
    base41 = json.loads((ROOT / "routing_study" / "results" / "topn_ablation"
                         / "summary.json").read_text())
    old_ids = {c["case_id"] for c in
               json.loads((ROOT / "data" / "mgh_qa_dataset.json").read_text())}
    sub = {}
    for scheme in ("Ax1", "P"):
        sub[scheme] = [r for r in scheme_rows[scheme] if r["case_id"] in old_ids]
    sub_summary = summarize(sub, hits_fn)
    print("\n===== 41 例旧子集 vs 基线（same-promptv2）=====", flush=True)
    compare_with_baseline({**sub_summary})


if __name__ == "__main__":
    main()
