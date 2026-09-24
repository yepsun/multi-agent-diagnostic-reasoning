#!/usr/bin/env python3
"""ER-Reason 判官补判（GLM-5.3-flash × v3 规则）→ judge_cache_glm_v3.json。

与主流程 judge_phase 的区别：as_completed 完成即写（每 100 条），
单次调用超时 60s（GLM 正常 1-6s），避免个别慢调用拖住整体写入；
断点续跑，3 轮重试未解析键。
"""
import json
import math
import os
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
DS_V3 = B / "judge_cache_dsflash_v3.json"
ER_DIR = B / "topn_erreason"
WORKERS = int(os.environ.get("GLM_WORKERS", "6"))
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


def er_rows():
    out = {}
    for f in ["ax1", "p", "mdt_synth"]:
        rows = [json.loads(l) for l in open(ER_DIR / f"{f}.jsonl") if l.strip()]
        out[f] = {r["case_id"]: r for r in rows}
    return out


def main():
    rows = er_rows()
    keys = set()
    for scheme in rows.values():
        for r in scheme.values():
            for c in r["top5"][:5]:
                keys.add(r["gold"][:150] + "||" + c[:150])
    cache = json.loads(GLM_CACHE.read_text())
    todo = sorted(k for k in keys if k not in cache)
    print(f"ER 唯一判定对 {len(keys)}；GLM 已有 {len(keys)-len(todo)}，"
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

    # 结果统计（GLM 口径）+ 与 DS-v3 对照
    ds = json.loads(DS_V3.read_text())

    def topk(r, k, cache_):
        for c in r["top5"][:k]:
            if cache_.get(r["gold"][:150] + "||" + c[:150]):
                return True
        return False

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(2 * sum(math.comb(n, i)
                           for i in range(min(b, c) + 1)) / 2 ** n, 1.0)

    names = {"ax1": "A×1", "p": "P", "mdt_synth": "MDT"}
    ids = list(rows["ax1"])
    for label, cache_ in [("GLM-v3", cache), ("DS-v3", ds)]:
        print(f"\n=== ER-Reason n={len(ids)} · 判官 {label} ===")
        for f in ["ax1", "p", "mdt_synth"]:
            line = []
            for k in (1, 3, 5):
                n = sum(1 for i in ids if topk(rows[f][i], k, cache_))
                line.append(f"top{k} {n/len(ids)*100:.1f}%")
            print(f"  {names[f]:4s} " + " | ".join(line))
    print("\n=== 配对 McNemar（GLM-v3）===")
    for k in (1, 3, 5):
        for a, b in [("mdt_synth", "ax1"), ("p", "ax1"), ("mdt_synth", "p")]:
            ao = bo = 0
            for i in ids:
                ha, hb = topk(rows[a][i], k, cache), topk(rows[b][i], k, cache)
                if ha and not hb:
                    ao += 1
                elif hb and not ha:
                    bo += 1
            print(f"  top-{k} {names[a]} vs {names[b]}: {ao}:{bo} "
                  f"p={mcnemar(ao,bo):.4f}")


if __name__ == "__main__":
    main()
