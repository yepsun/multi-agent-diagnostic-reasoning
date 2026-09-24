#!/usr/bin/env python3
"""ER-Reason A x1 @ T=0.3, seeds 2-5 (extending the temperature-matched arm).

Seed 1 of this arm was produced by `ax1_t03_sensitivity.run_er()` into
`results/topn_erreason_ax1t03.jsonl`. This driver runs seeds 2-5 and copies
seed 1 into `results/topn_erreason_ax1t03/` so that the arm has a uniform
five-seed layout for the case-level temperature-matched analysis.

Prompt, case text and call parameters are identical to the main ER A x1 arm
(`topn_erreason.run_simple` + `topn_cpc_promptv2.A_TOPN_PROMPT`, temperature
0.0); the only difference here is temperature 0.3.

  SMOKE=1 ./.venv/bin/python routing_study/scripts/erreason_ax1t03_seeds.py
          -> 3 cases into /tmp/erreason_ax1t03_smoke/
  ./.venv/bin/python routing_study/scripts/erreason_ax1t03_seeds.py
          -> seeds 2-5 x 364 cases, resumable
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

from topn_cpc_promptv2 import A_TOPN_PROMPT  # noqa: E402
import os as _os
_os.environ.setdefault("OPENROUTER_ER", "1")  # ER 合规路由：生成调用经 OpenRouter
from topn_erreason import load_cases, run_simple  # noqa: E402

OUTDIR = ROOT / "routing_study" / "results" / "topn_erreason_ax1t03"
SEED1_SRC = ROOT / "routing_study" / "results" / "topn_erreason_ax1t03.jsonl"
SEEDS = (2, 3, 4, 5)
PROMPT = lambda c: A_TOPN_PROMPT.format(case_text=c["text"])  # noqa: E731


def smoke():
    out = Path("/tmp/erreason_ax1t03_smoke")
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases()[:3]
    run_simple("SMOKE", out / "Ax1t03_s2.jsonl", PROMPT, 0.3, cases)
    rows = [json.loads(l) for l in open(out / "Ax1t03_s2.jsonl") if l.strip()]
    for r in rows:
        print(f"  {r['case_id'][:20]} gold={r['gold'][:40]!r} "
              f"top1={r['top5'][:1]} tokens={r['total_tokens']}")
    print(f"[冒烟] {len(rows)}/3 行，全部 5 项: "
          f"{all(len(r['top5']) == 5 for r in rows)}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst1 = OUTDIR / "Ax1t03_s1.jsonl"
    if not dst1.exists():
        shutil.copy2(SEED1_SRC, dst1)
        print(f"[复制] seed 1: {SEED1_SRC.name} -> {dst1}", flush=True)
    cases = load_cases()
    print(f"ER-Reason {len(cases)} 例 × seeds {list(SEEDS)}，T=0.3", flush=True)
    for seed in SEEDS:
        run_simple(f"Ax1t03 s{seed}", OUTDIR / f"Ax1t03_s{seed}.jsonl",
                   PROMPT, 0.3, cases)
    for seed in (1,) + SEEDS:
        rows = [json.loads(l) for l in open(OUTDIR / f"Ax1t03_s{seed}.jsonl")
                if l.strip()]
        assert len(rows) == len(cases), f"s{seed} 缺行：{len(rows)}"
        assert all(len(r["top5"]) == 5 for r in rows), f"s{seed} top5 不完整"
    print("[检查] 5 seeds × 364 例完整、top5 均为 5 项", flush=True)


if __name__ == "__main__":
    if os.environ.get("SMOKE") == "1":
        smoke()
    else:
        main()
