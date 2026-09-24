"""标出 workup 条件下"参考标签字面出现在客观数据中"的病例。

判据（论文 Methods 所述规则的精确实现，可复现 24/364）：
1. 把参考标签与客观段都归一化：转小写，把所有非字母字符替换为空格；
2. 取参考标签中归一化后长度 > 4 的词；
3. 若这些词**全部**作为子串出现在归一化后的客观段中，则该病例被标记。

注意：第 3 步是子串匹配而非词边界匹配（词边界口径得 23 例）。
输出：routing_study/results/workup_answer_visible_cases.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "er_reason_workup_subset.json"
OUT = ROOT / "routing_study" / "results" / "workup_answer_visible_cases.json"

MIN_TOKEN_LEN = 5  # 保留长度 > 4 的词


def normalise(text: str) -> str:
    return re.sub(r"[^a-z ]", " ", (text or "").lower())


def label_tokens(gold: str) -> list[str]:
    return [t for t in normalise(gold).split() if len(t) >= MIN_TOKEN_LEN]


def main() -> None:
    cases = json.loads(SUBSET.read_text())
    flagged, n_with_objective = [], 0
    for c in cases:
        objective = normalise(c.get("objective", ""))
        if c.get("objective"):
            n_with_objective += 1
        toks = label_tokens(c.get("gold", ""))
        if objective and toks and all(t in objective for t in toks):
            flagged.append({
                "case_id": c["case_id"],
                "gold": c["gold"],
                "label_tokens": toks,
            })

    n = len(cases)
    payload = {
        "criterion": (
            "every word of the reference label longer than four characters "
            "(after lowercasing and replacing non-letters with spaces) appears as a "
            "substring of the normalised objective text"
        ),
        "n_cases": n,
        "n_cases_with_objective": n_with_objective,
        "n_flagged": len(flagged),
        "fraction_flagged": len(flagged) / n,
        "cases": flagged,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"标记 {len(flagged)}/{n} 例（{len(flagged)/n*100:.1f}%），has_objective={n_with_objective}")
    print(f"写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
