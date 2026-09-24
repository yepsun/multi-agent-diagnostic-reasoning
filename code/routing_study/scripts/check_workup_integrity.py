"""检查 workup 条件下各 seed 的文件完整性，列出需要补跑的内容。

用法：
    ./.venv/bin/python routing_study/scripts/check_workup_integrity.py            # 检查全部
    ./.venv/bin/python routing_study/scripts/check_workup_integrity.py --fix-empty  # 删除空角色行以便重试

每个 seed 应满足：ax1/p/mdt_synth 各 364 个唯一 case 且无空 top5；
mdt_roles 覆盖 364 case × 5 角色且无空 top5。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "routing_study" / "results" / "topn_erreason_workup"
N_CASES = 364
ROLES_PER_CASE = 5


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_dirs() -> list[tuple[str, Path]]:
    out = [("s1", BASE)]
    for s in range(2, 6):
        d = BASE / f"s{s}"
        if d.exists():
            out.append((f"s{s}", d))
    return out


def check(name: str, d: Path, fix_empty: bool) -> list[str]:
    problems: list[str] = []
    rows = {f: load(d / f"{f}.jsonl") for f in ("ax1", "p", "mdt_roles", "mdt_synth")}

    for stage in ("ax1", "p", "mdt_synth"):
        r = rows[stage]
        ids = {x["case_id"] for x in r}
        empty = [x["case_id"] for x in r if not x.get("top5")]
        if len(ids) != N_CASES:
            problems.append(f"{stage}: 唯一 case {len(ids)}/{N_CASES}")
        if empty:
            problems.append(f"{stage}: 空 top5 {len(empty)} 例")

    roles = rows["mdt_roles"]
    by_case: dict[str, int] = {}
    empty_roles = []
    for r in roles:
        by_case[r["case_id"]] = by_case.get(r["case_id"], 0) + 1
        if not r.get("top5"):
            empty_roles.append((r["case_id"], r["role"]))
    incomplete = {c: n for c, n in by_case.items() if n < ROLES_PER_CASE}
    if len(by_case) != N_CASES:
        problems.append(f"mdt_roles: 覆盖 case {len(by_case)}/{N_CASES}")
    if incomplete:
        problems.append(f"mdt_roles: 角色不足的 case {len(incomplete)} 个")
    if empty_roles:
        problems.append(f"mdt_roles: 空 top5 角色 {len(empty_roles)} 个")

    # 汇总短缺：有 5 角色但缺汇总的 case
    synth_ids = {x["case_id"] for x in rows["mdt_synth"]}
    full_role_cases = {c for c, n in by_case.items() if n >= ROLES_PER_CASE}
    missing_synth = full_role_cases - synth_ids
    if missing_synth:
        problems.append(f"mdt_synth: 有完整角色但缺汇总 {len(missing_synth)} 例")

    if fix_empty and empty_roles:
        keep = [r for r in roles if r.get("top5")]
        (d / "mdt_roles.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
        problems.append(f"→ 已删除 {len(empty_roles)} 个空角色行（重跑时会补）")

    return problems


def main() -> None:
    fix = "--fix-empty" in sys.argv
    print(f"检查目录: {BASE}\n")
    any_issue = False
    for name, d in seed_dirs():
        problems = check(name, d, fix)
        if problems:
            any_issue = True
            print(f"[{name}] 待补：")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[{name}] 完整")
    if any_issue:
        print("\n补跑方式：对该 seed 重新执行推理命令（脚本只补缺失项，幂等）：")
        print('  MAX_WORKERS=8 ER_DATA="$PWD/data/er_reason_workup_subset.json" '
              'ER_OUTDIR="topn_erreason_workup/sN" SKIP_JUDGE=1 '
              './.venv/bin/python routing_study/scripts/topn_erreason.py')


if __name__ == "__main__":
    main()
