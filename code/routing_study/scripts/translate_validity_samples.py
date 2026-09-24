import os as _os
#!/usr/bin/env python3
"""为 ER/MCR 人工盲评样本生成中文翻译（gold_zh / candidate_zh）。

协议要点：
- 与 batch-2（CPC translations.json）一致，翻译仅作语言辅助，评者仍独立做医学等价判断；
- 每条字符串独立翻译（不给出配对对象、不给 LLM judge 结论），不破坏盲评；
- 评者所见文本与裁判实际输入一致：先按 key_of 的规则截断到 150 字符；
- 走裁判同通道（GLM 官方端点 GLM-5.3-Flash），按字符串缓存，逐批落盘可断点续跑。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))

from caselevel_stats import (RESULTS, _judge_gate, _judge_event,  # noqa: E402
                             _post_hard_deadline, GLM_URL, GLM_JUDGE_KEY,
                             GLM_JUDGE_MODEL, GLM_TIMEOUT)
import requests  # noqa: E402

VALIDITY = RESULTS / "judge_validity"
T_CACHE = VALIDITY / "translations_er_mcr.json"
BATCH = 12

PROMPT = """请把下列编号的医学诊断条目逐条翻译为规范的中文医学术语。
要求：
- 忠实原意，保留所有限定词（分期、分型、部位、 laterality、并发症、"secondary to"/"due to" 等因果结构、括号内的编码或说明）；
- 英文缩写译出中文全称并保留缩写，如 "ARDS（急性呼吸窘迫综合征）"；
- 不确定时按最常见医学含义直译，不要加解释、不要合并或拆分条目；
- 只输出 JSON 对象（键为编号字符串，值为译文），不要输出任何其他文字。

条目：
"""


def chat(text, max_retries=4):
    body = {"model": GLM_JUDGE_MODEL, "temperature": 0,
            "messages": [{"role": "user", "content": text}]}
    for attempt in range(max_retries):
        _judge_gate()
        try:
            r = _post_hard_deadline(GLM_URL, 180,
                                    headers={"Authorization":
                                             f"Bearer {GLM_JUDGE_KEY}"},
                                    json=body, timeout=GLM_TIMEOUT)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            s, e = content.find("{"), content.rfind("}")
            obj = json.loads(content[s:e + 1])
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:  # noqa: BLE001
            _judge_event()
            time.sleep(min(20 * (attempt + 1), 90))
    return {}


def main():
    tc = json.loads(T_CACHE.read_text()) if T_CACHE.exists() else {}
    for ds in ("er", "mcr"):
        sample = json.loads((VALIDITY / f"sample_{ds}.json").read_text())
        # 与裁判输入对齐：截断到 key_of 的 150 字符
        for t in sample:
            t["gold"] = t["gold"][:150]
            t["candidate"] = t["candidate"][:150]
        todo = sorted({s[t] for t in ("gold", "candidate") for s in sample}
                      - set(tc))
        print(f"[{ds}] 需翻译 {len(todo)} 条（已缓存 {len(tc)}）")
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            numbered = "\n".join(f"{j + 1}. {s}"
                                 for j, s in enumerate(chunk))
            obj = chat(PROMPT + numbered)
            for j, s in enumerate(chunk):
                zh = obj.get(str(j + 1))
                if isinstance(zh, str) and zh.strip():
                    tc[s] = zh.strip()
            T_CACHE.write_text(json.dumps(tc, ensure_ascii=False, indent=1))
            time.sleep(1)
        for t in sample:
            t["gold_zh"] = tc.get(t["gold"], "")
            t["candidate_zh"] = tc.get(t["candidate"], "")
        (VALIDITY / f"sample_{ds}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=1))
        missing = sum(1 for t in sample if not t["gold_zh"]
                      or not t["candidate_zh"])
        print(f"[{ds}] 完成，缺译条目 {missing} → sample_{ds}.json")


if __name__ == "__main__":
    main()
