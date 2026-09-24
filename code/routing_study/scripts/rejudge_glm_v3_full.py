#!/usr/bin/env python3
"""GLM-5.3-flash × v3 规则全量重判 v3 缓存全部对 → judge_cache_glm_v3.json。

主判官切换的数据基础：对 judge_cache_dsflash_v3.json 的每个 key 用同一
V3_PROMPT 以 GLM 判定，独立缓存、不动 DS 缓存。断点续跑；严格解析，
失败对留待下轮重试（最多 3 轮）；结尾输出与 DS-v3 的翻转统计。
GLM 思考不可关：effort low + max_tokens 8192。

LIMIT=N 环境变量可限制判定对数（冒烟测试用）。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import requests

from judge_v3 import V3_PROMPT  # noqa: E402

DS_V3 = ROOT / "routing_study" / "results" / "judge_cache_dsflash_v3.json"
GLM_CACHE = ROOT / "routing_study" / "results" / "judge_cache_glm_v3.json"
WORKERS = 8
LIMIT = int(os.environ.get("LIMIT", "0"))

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
                      json=body, timeout=420)
    content = (r.json()["choices"][0]["message"].get("content") or "")
    v = content.strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def main():
    ds = json.loads(DS_V3.read_text())
    glm = json.loads(GLM_CACHE.read_text()) if GLM_CACHE.exists() else {}
    todo = [k for k in ds if k not in glm]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"DS-v3 缓存 {len(ds)} 对；GLM 已有 {len(glm)}，待判 {len(todo)}",
          flush=True)

    for round_no in (1, 2, 3):
        if not todo:
            break
        errs = []

        def work(key):
            gold, _, cand = key.partition("||")
            try:
                return key, glm_verdict(gold, cand)
            except Exception:
                return key, None

        with ThreadPoolExecutor(WORKERS) as ex:
            for i, (key, verdict) in enumerate(ex.map(work, todo), 1):
                if verdict is None:
                    errs.append(key)
                else:
                    glm[key] = verdict
                if i % 500 == 0 or i == len(todo):
                    GLM_CACHE.write_text(json.dumps(glm, ensure_ascii=False))
                    print(f"  轮{round_no} {i}/{len(todo)} | 失败 {len(errs)}",
                          flush=True)
        GLM_CACHE.write_text(json.dumps(glm, ensure_ascii=False))
        todo = errs
        print(f"轮{round_no} 结束：失败 {len(errs)}", flush=True)

    both = [k for k in glm if k in ds]
    flips = [(k, ds[k], glm[k]) for k in both if ds[k] != glm[k]]
    n2y = sum(1 for _, o, n in flips if not o and n)
    print(f"\n完成：GLM 判定 {len(glm)} 对 | 与 DS-v3 不一致 {len(flips)} "
          f"({len(flips)/max(len(both),1)*100:.2f}%；NO→YES {n2y}，"
          f"YES→NO {len(flips)-n2y}) | 未解析 {len(todo)}", flush=True)
    print(f"已写 {GLM_CACHE}", flush=True)


if __name__ == "__main__":
    main()
