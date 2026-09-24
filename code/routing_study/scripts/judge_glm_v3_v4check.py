#!/usr/bin/env python3
"""补全判官矩阵缺失格子：GLM-5.3-flash × v3 规则 × 两批盲评样本。

- SET=v4check（默认）：第二批全新 100 例（无偏验证集），参照为 DS-v3
  （v3_verdicts.json）与 v4check 双人标注；
- SET=v1：第一批 100 例（边界对分层），参照为 DS-v2（llm_verdict）与
  双人标注；同批已有 GLM×v2（glm_judge_v2_100.json），可得 GLM 的
  v2→v3 同批对比。

输出 judge_validity/glm_judge_v3_<set>.json（断点续跑），并打印 κ 矩阵。
思考模式不可关，用 effort low + max_tokens 8192；严格解析。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import requests

from judge_study import cohens_kappa, STUDY_DIR  # noqa: E402
from judge_v3 import V3_PROMPT  # noqa: E402

SET = os.environ.get("SET", "v4check")
if SET == "v4check":
    SAMPLE = STUDY_DIR / "sample_v4check.json"
    ANN_A = STUDY_DIR / "v4check_annotations_a.json"
    ANN_B = STUDY_DIR / "v4check_annotations_b.json"
    DS_REF = None   # DS×v3 判定就在 sample 的 llm_verdict 字段（经核对
                    # 与 kappa_report_v4check.json 的 κ=0.66/0.76 一致）
    REF_NAME = "deepseek判官(v3)"
else:
    SAMPLE = STUDY_DIR / "sample.json"
    ANN_A = STUDY_DIR / "annotations_a.json"
    ANN_B = STUDY_DIR / "annotations_b.json"
    DS_REF = None                              # 用 sample 内 llm_verdict（DS × v2）
    REF_NAME = "deepseek判官(v2)"

OUT = STUDY_DIR / f"glm_judge_v3_{SET}.json"

KEY = [l.split("=", 1)[1].strip() for l in open(ROOT / "scripts" / ".env")
       if l.startswith("ZHIPU_API_KEY=")][0]


def glm_judge(gold, cand):
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "user",
                          "content": V3_PROMPT.format(gold=gold, pred=cand)}],
            "max_tokens": 8192, "temperature": 0.0,
            "thinking": {"type": "enabled", "reasoning_effort": "low"}}
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json=body, timeout=420)
    d = r.json()
    content = (d["choices"][0]["message"].get("content") or "")
    tokens = (d.get("usage") or {}).get("total_tokens", 0)
    v = content.strip().upper()
    if v.startswith("YES"):
        return True, tokens, None
    if v.startswith("NO"):
        return False, tokens, None
    return None, tokens, content.strip()[:60]


def main():
    sample = json.loads(SAMPLE.read_text())
    verdicts = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [t for t in sample if str(t["item_id"]) not in verdicts]
    print(f"GLM×v3×{SET}：已完成 {len(verdicts)}，待判 {len(todo)}", flush=True)

    parse_fail = 0
    tokens_total = 0
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(glm_judge, t["gold"], t["candidate"]): t
                for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                verdict, tokens, raw_tail = fut.result()
            except Exception as e:
                verdict, tokens, raw_tail = None, 0, f"EXC {e}"
            tokens_total += tokens
            if verdict is None:
                parse_fail += 1
                print(f"  [{i}] 解析失败: {raw_tail}", flush=True)
            else:
                verdicts[str(t["item_id"])] = verdict
            OUT.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1))
    print(f"完成：成功 {len(verdicts)} | 解析失败 {parse_fail} | "
          f"总 tokens {tokens_total}", flush=True)

    a = json.loads(ANN_A.read_text())
    b = json.loads(ANN_B.read_text())
    if DS_REF is not None:
        ds = json.loads(DS_REF.read_text())
        vl_map = {int(k): v for k, v in ds.items()}
    else:
        vl_map = {t["item_id"]: t["llm_verdict"] for t in sample}
    ids = [int(k) for k in verdicts
           if str(int(k)) in a and str(int(k)) in b and int(k) in vl_map]
    va = [a[str(i)] for i in ids]
    vb = [b[str(i)] for i in ids]
    vg = [verdicts[str(i)] for i in ids]
    vl = [vl_map[i] for i in ids]
    print(f"\n===== GLM×v3×{SET}（n={len(ids)}）=====")
    print(f"GLM-v3 vs 评者A:   {cohens_kappa(vg, va)}")
    print(f"GLM-v3 vs 评者B:   {cohens_kappa(vg, vb)}")
    print(f"GLM-v3 vs {REF_NAME}: {cohens_kappa(vg, vl)}")
    print(f"（参照）{REF_NAME} vs 评者A: {cohens_kappa(vl, va)}")
    print(f"（参照）{REF_NAME} vs 评者B: {cohens_kappa(vl, vb)}")


if __name__ == "__main__":
    main()
