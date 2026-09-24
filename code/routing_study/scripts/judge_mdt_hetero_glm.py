#!/usr/bin/env python3
"""MDT-异构判定补判（GLM-5.3-flash × v3 规则）→ judge_cache_glm_v3.json。

收集 topn_mdt_hetero/{s1 顶层,s2,s3}/synthesis.jsonl 中全部 (gold, cand) 对，
只补判 judge_cache_glm_v3.json 中缺失的键；已有判定一律复用，绝不重判。

GLM 调用参数与 judge_erreason_glm.py 一致：6 并发、150s 超时、3 轮重试、
thinking 低 effort + 8192 token 预算、temperature 0。
判官提示词复用 judge_v3.V3_PROMPT（v3 版本，不新写）。
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

import requests

from judge_v3 import V3_PROMPT  # noqa: E402

B = ROOT / "routing_study" / "results"
GLM_CACHE = B / "judge_cache_glm_v3.json"
HETERO = B / "topn_mdt_hetero"
WORKERS = 6
TIMEOUT = 150
ERRLOG = []

KEY = [l.split("=", 1)[1].strip() for l in open(ROOT / "scripts" / ".env")
       if l.startswith("ZHIPU_API_KEY=")][0]


def glm_verdict(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        if len(ERRLOG) < 20:
            ERRLOG.append(f"HTTP {r.status_code}: {r.text[:160]}")
        return None
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    if len(ERRLOG) < 20:
        ERRLOG.append(f"UNPARSED: {content.strip()[:100]!r}")
    return None


def hetero_keys():
    keys = set()
    for s in (1, 2, 3):
        p = HETERO / "synthesis.jsonl" if s == 1 else HETERO / f"s{s}" / "synthesis.jsonl"
        if not p.exists():
            print(f"[跳过] {p} 不存在", flush=True)
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for c in r["top5"][:5]:
                keys.add(r["gold"][:150] + "||" + c[:150])
    return keys


def main():
    keys = hetero_keys()
    cache = json.loads(GLM_CACHE.read_text())
    n_before = len(cache)
    todo = sorted(k for k in keys if k not in cache)
    print(f"hetero 唯一判定对 {len(keys)}；GLM 缓存已有 {len(keys)-len(todo)}，"
          f"待判 {len(todo)}", flush=True)

    for round_no in (1, 2, 3):
        if not todo:
            break
        errs = []
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(glm_verdict, *k.split("||", 1)): k
                    for k in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                k = futs[fut]
                try:
                    v = fut.result()
                except Exception:
                    v = None
                if v is None:
                    errs.append(k)
                else:
                    cache[k] = v
                if i % 100 == 0:
                    GLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                    print(f"  轮{round_no} {i}/{len(todo)} | 失败 {len(errs)}",
                          flush=True)
        GLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        todo = errs
        print(f"轮{round_no} 结束：失败 {len(errs)}", flush=True)
        if ERRLOG:
            print("  失败样例：" + " | ".join(ERRLOG[-3:]), flush=True)

    GLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    still_missing = [k for k in keys if k not in cache]
    print(f"完成：缓存 {n_before} → {len(cache)}（新增 {len(cache)-n_before} 对），"
          f"hetero 仍缺 {len(still_missing)} 对", flush=True)


if __name__ == "__main__":
    main()
