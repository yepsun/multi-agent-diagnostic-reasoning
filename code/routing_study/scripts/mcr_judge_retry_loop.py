#!/usr/bin/env python3
"""MCR 判分补齐循环：反复 judge_missing 直到 0 缺失，然后跑收尾监督器。

背景：PHASE=all 的 judge 阶段在 GLM 限流下 3 轮放弃 5,988 对，随后的分析
按 miss 计入——该版分析无效（ax5_mod_mcr.md 已打作废标记）。本循环以
flock 互斥反复补判；清零后调用 pipeline_supervisor_v2.sh 做最终收尾
（全阶段幂等跳过，judge 即时通过，analyze 出有效版并写退出码行，
从而让 chain2 自动解锁）。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import caselevel_stats as cs  # noqa: E402
import ax5_mod_mcr as mcr  # noqa: E402

LOG = ROOT / "routing_study" / "results" / "mcr_judge_retry.log"


def log(msg):
    line = f"[mcr-retry] {time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_rows():
    rows = []
    for seed in mcr.SEEDS:
        rows.extend(cs.load(mcr.sample_path(seed)).values())
        rows.extend(cs.load(mcr.mod_path(seed)).values())
    return rows


def bounded_batch(rows, cache, cap=250):
    """选缺失累计 ≤cap 的最小行子集交给 judge_missing（单次整体落盘）。"""
    selected, acc = [], 0
    for r in rows:
        m = sum(1 for c in r["top5"][:5] if cs.key_of(r["gold"], c) not in cache)
        if m == 0:
            continue
        selected.append(r)
        acc += m
        if acc >= cap:
            break
    if not selected:
        return cache, 0
    return cs.judge_missing(selected), acc


def main():
    rows = collect_rows()
    cache = json_load_cache()
    missing = cs.missing_pairs(rows, cache)
    log(f"初始缺失 {len(missing)} 对")
    for rnd in range(1, 31):
        if not missing:
            break
        log(f"轮 {rnd}：尝试 {len(missing)} 对（小批量 paced：250 对/批，批间 45s）")
        # 探针门控：服务黑障时单发探针（1 对），失败长休 10 分钟，避免整批陪葬
        while True:
            probe_key = next(iter(missing))
            try:
                cs.glm_verdict(missing[probe_key][0], missing[probe_key][1])
                break
            except Exception as e:
                log(f"  探针失败（{type(e).__name__}），智谱不可用，休 600s 后重探")
                time.sleep(600)
        drained = 0
        zero_streak = 0
        while True:
            cache = json_load_cache()
            missing = cs.missing_pairs(rows, cache)
            if not missing:
                break
            cache2, n = bounded_batch(rows, cache)
            drained += n
            log(f"  批完成 +{n}，缓存 → {len(cache2)}")
            cache = cache2
            missing = cs.missing_pairs(rows, cache)
            if not missing:
                break
            if n == 0:
                zero_streak += 1
                pause = min(600 * (2 ** (zero_streak - 1)), 3600)
                log(f"  批零产出（连续第 {zero_streak} 次），退避 {pause}s 等服务恢复")
                time.sleep(pause)
                continue
            zero_streak = 0
            time.sleep(45)
        log(f"轮 {rnd}：剩余 {len(missing)} 对（本轮判 {drained}）")
        if missing:
            time.sleep(120)
    if missing:
        log(f"30 轮后仍有 {len(missing)} 对未判，放弃（需人工介入）")
        sys.exit(1)
    log("缺失清零，运行收尾监督器（幂等）生成有效分析")
    rc = subprocess.run(
        ["bash", str(ROOT / "routing_study" / "scripts" / "pipeline_supervisor_v2.sh")],
        stdout=open(ROOT / "routing_study" / "results" / "resume_20260920.log", "a"),
        stderr=subprocess.STDOUT,
    ).returncode
    log(f"收尾监督器退出码 {rc}")


def json_load_cache():
    return __import__("json").loads(cs.GLM_CACHE.read_text())


if __name__ == "__main__":
    main()
