import os as _os
#!/usr/bin/env python3
"""ER-Reason seed 输出完整性校验 + 坏行清理。

用法：python er_seeds_clean.py <outdir>
- 剔除 ax1/p/mdt_synth 中空 top5 的行、mdt_roles 中空 top5 的角色行（断点续跑会重补）。
- 齐全（ax1/p/mdt_synth 各 364 个唯一 case_id、mdt_roles 364×5 角色且全非空）
  时退出码 0，否则退出码 1。
"""
import json
import sys
from pathlib import Path

N_CASES = 364
N_ROLES = 5


def clean_simple(path):
    if not path.exists():
        return {}
    good = {}
    bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("top5"):
            good[row["case_id"]] = row
        else:
            bad += 1
    if bad:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in good.values()))
        print(f"  {path.name}: 剔除空 top5 行 {bad}", flush=True)
    return good


def clean_roles(path):
    if not path.exists():
        return {}
    good = {}
    bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("top5"):
            good[(row["case_id"], row["role"])] = row
        else:
            bad += 1
    if bad:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in good.values()))
        print(f"  {path.name}: 剔除空 top5 行 {bad}", flush=True)
    return good


def main():
    outdir = Path(sys.argv[1])
    ax1 = clean_simple(outdir / "ax1.jsonl")
    p = clean_simple(outdir / "p.jsonl")
    synth = clean_simple(outdir / "mdt_synth.jsonl")
    roles = clean_roles(outdir / "mdt_roles.jsonl")
    ok = (len(ax1) == N_CASES and len(p) == N_CASES
          and len(synth) == N_CASES and len(roles) == N_CASES * N_ROLES)
    print(f"{outdir}: ax1 {len(ax1)}/{N_CASES} | p {len(p)}/{N_CASES} | "
          f"mdt_synth {len(synth)}/{N_CASES} | "
          f"mdt_roles {len(roles)}/{N_CASES * N_ROLES} -> "
          f"{'OK' if ok else 'INCOMPLETE'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
