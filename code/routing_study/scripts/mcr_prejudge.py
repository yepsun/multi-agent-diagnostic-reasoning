#!/usr/bin/env python3
"""MCR 预判分运行器：主链主持人阶段推进期间，抢先判分已冻结的行。

与主链（PHASE=all 单进程，mod 完成后立即自判）的写互斥保证：
- 每批 ≤ PREJUDGE_CAP(250) 对（约 5 分钟），批间重新评估；
- 一旦 Mod 行总数进入尾段（≥ PREJUDGE_ABORT_AT，即只剩最后一个 seed 的余量，
  主链最快数分钟内就会进入自己的 judge 阶段），立即退出让位；
- 只读行对象（不碰主链正在 append 的文件写入光标），共享缓存每批由
  cs.judge_missing 整体落盘一次。
用法：.venv/bin/python routing_study/scripts/mcr_prejudge.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402
import ax5_mod_mcr as mcr  # noqa: E402

PREJUDGE_CAP = 250          # 每批最多判对数（约 5 分钟）
PREJUDGE_ABORT_AT = 1625    # Mod 行总数达到此值即退出（余 ≥406 次主持人调用）
LIFETIME_S = 8 * 3600
LOG = ROOT / "routing_study" / "results" / "mcr_prejudge.log"


def log(msg):
    line = f"[prejudge] {time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_rows():
    rows = []
    for seed in mcr.SEEDS:
        rows.extend(cs.load(mcr.sample_path(seed)).values())
        p = mcr.mod_path(seed)
        if p.exists():
            rows.extend(cs.load(p).values())
    return rows


def mod_total():
    n = 0
    for seed in mcr.SEEDS:
        p = mcr.mod_path(seed)
        if p.exists():
            n += len(cs.load(p))
    return n


def bounded_judge(rows, cache):
    """选缺失累计 ≤CAP 的最小行子集交给 judge_missing（单次整体落盘）。"""
    selected, acc = [], 0
    for r in rows:
        m = sum(1 for c in r["top5"][:5] if cs.key_of(r["gold"], c) not in cache)
        if m == 0:
            continue
        selected.append(r)
        acc += m
        if acc >= PREJUDGE_CAP:
            break
    if not selected:
        return cache, 0
    return cs.judge_missing(selected), acc


def main():
    t0 = time.time()
    log(f"启动（cap={PREJUDGE_CAP}, abort_at={PREJUDGE_ABORT_AT}）")
    while time.time() - t0 < LIFETIME_S:
        total = mod_total()
        if total >= PREJUDGE_ABORT_AT:
            log(f"Mod 行总数 {total} ≥ {PREJUDGE_ABORT_AT}，主链即将自判，退出让位")
            break
        rows = collect_rows()
        cache = json.loads(cs.GLM_CACHE.read_text())
        missing = cs.missing_pairs(rows, cache)
        if not missing:
            time.sleep(300)
            continue
        cache2, n = bounded_judge(rows, cache)
        left = len(cs.missing_pairs(rows, cache2))
        log(f"批完成：新增约 {n} 对判定，缓存 → {len(cache2)}，本快照余 {left} 对未判")
        if total >= PREJUDGE_ABORT_AT:
            log("批后发现进入尾段，退出让位")
            break
        time.sleep(15)
    log("预判分结束")


if __name__ == "__main__":
    main()
