#!/usr/bin/env python3
"""统一判官重判：全部历史 (gold, candidate) 对用 deepseek-flash（v2 规则）重判。

背景：判定缓存内容寻址、无判官模型标记；2026-09-10 前的历史判定由
deepseek-v4-pro 完成（topn_ablation/judge_cache.json 等），无法区分来源。
为彻底剔除 v4-pro，将各实验目录 judge_cache*.json 中出现过的全部唯一
(gold, cand) 对用当前判官 deepseek-flash 重判，产出统一缓存：

  routing_study/results/judge_cache_dsflash_unified.json

旧缓存文件保持不动（留作审计对照）。断点续跑：已在统一缓存中的 key 跳过。
结束后打印相对各旧缓存的翻转统计（判官档位稳健性证据）。

key 与下游 key_of() 一致：gold[:150] + "||" + cand[:150]（截断判定与
下游查询使用同一字符串，行为一致）。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")

from run_inference import call_llm_judge  # noqa: E402
from mdt_cpc import JUDGE_PROMPT_T  # noqa: E402 与 run_static_routing.judge 同文

RESULTS = ROOT / "routing_study" / "results"
UNIFIED = RESULTS / "judge_cache_dsflash_unified.json"
WORKERS = 16
SAVE_EVERY = 200


def collect():
    """返回 (all_keys, per_cache) ：全部唯一 key 与每个缓存的 {key: verdict}。"""
    per_cache = {}
    for p in sorted(RESULTS.glob("**/judge_cache*.json")):
        if p == UNIFIED:
            continue
        per_cache[str(p.relative_to(RESULTS))] = json.loads(p.read_text())
    all_keys = set()
    for d in per_cache.values():
        all_keys.update(d.keys())
    return all_keys, per_cache


def judge_one(key):
    gold, cand = key.split("||", 1)
    try:
        raw, _ = call_llm_judge(JUDGE_PROMPT_T(gold, cand), timeout=60,
                                max_retries=2)
        v = (raw or "").strip().upper()
        if v.startswith("YES"):
            return key, True
        if v.startswith("NO"):
            return key, False
    except Exception:
        pass
    return key, None


def main():
    all_keys, per_cache = collect()
    done = json.loads(UNIFIED.read_text()) if UNIFIED.exists() else {}
    todo = sorted(all_keys - set(done))
    print(f"唯一判定对 {len(all_keys)}，已完成 {len(done)}，待判 {len(todo)}",
          flush=True)

    pending = todo
    for round_no in (1, 2, 3):
        if not pending:
            break
        nxt = []
        n_since_save = 0
        with ThreadPoolExecutor(WORKERS) as ex:
            for i, (key, verdict) in enumerate(ex.map(judge_one, pending), 1):
                if verdict is None:
                    nxt.append(key)
                else:
                    done[key] = verdict
                    n_since_save += 1
                if n_since_save >= SAVE_EVERY:
                    UNIFIED.write_text(json.dumps(done, ensure_ascii=False))
                    n_since_save = 0
                if i % 500 == 0 or i == len(pending):
                    print(f"[第{round_no}轮] {i}/{len(pending)}", flush=True)
        pending = nxt
        UNIFIED.write_text(json.dumps(done, ensure_ascii=False))
    print(f"统一缓存 {len(done)} 条，未解析 {len(pending)}", flush=True)

    print("\n===== 相对旧缓存的翻转（判官档位稳健性）=====", flush=True)
    flips = {}
    for name, old in per_cache.items():
        both = [k for k in old if k in done]
        flip = sum(1 for k in both if bool(old[k]) != bool(done[k]))
        flips[name] = {"overlap": len(both), "flipped": flip,
                       "flip_rate": round(flip / len(both), 4) if both else None}
        print(f"{name}: 重叠 {len(both)}，翻转 {flip} "
              f"({flip / len(both):.1%})" if both else f"{name}: 无重叠",
              flush=True)
    (RESULTS / "rejudge_dsflash_report.json").write_text(json.dumps(
        {"total": len(done), "unresolved": len(pending), "flips": flips},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {UNIFIED} 与 rejudge_dsflash_report.json", flush=True)


if __name__ == "__main__":
    main()
