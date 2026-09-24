#!/usr/bin/env python3
"""离线重判分：用修复后的 _gold_text / judge 重算 static_routing_raw.jsonl 的对错。

不重跑任何模型生成调用；只重新调用语义判分器。
原文件备份为 static_routing_raw.v1.jsonl（首跑判分口径），重判结果原地覆写。
"""
import sys, os, json, shutil
from concurrent.futures import ThreadPoolExecutor

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

from case_extraction import preprocess_case_text  # noqa: F401  (与 runner 同一加载路径)
from run_static_routing import _gold_text, judge

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "static_routing_raw.jsonl")
BACKUP = os.path.join(HERE, "..", "results", "static_routing_raw.v1.jsonl")
DATASET = os.path.join(HERE, "..", "..", "data", "mgh_qa_dataset.json")


def main():
    if not os.path.exists(BACKUP):
        shutil.copy(RAW, BACKUP)
        print(f"backed up v1 verdicts -> {os.path.basename(BACKUP)}")

    cases = {str(c["case_id"]): c for c in json.load(open(DATASET))}
    rows = [json.loads(l) for l in open(RAW) if l.strip()]

    jobs = []  # (row_idx, field, pred, gold)
    for i, r in enumerate(rows):
        gold = _gold_text(cases[r["case_id"]])
        r["gold"] = gold
        for field, pred in (("a_majority", r["a_majority"]),
                            ("p_answer", r["p_answer"]),
                            ("p_qwen_answer", r["p_qwen_answer"])):
            jobs.append((i, field, pred, gold))

    def work(job):
        i, field, pred, gold = job
        return i, field, judge(gold, pred)

    changed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, field, verdict in pool.map(work, jobs):
            key = field.replace("_majority", "_correct").replace("_answer", "_correct")
            if rows[i][key] != verdict:
                changed += 1
            rows[i][key] = verdict
            print(f"[{rows[i]['case_id'][:40]}] {key}: {verdict}", flush=True)

    with open(RAW, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nrejudged {len(jobs)} verdicts, changed {changed}")


if __name__ == "__main__":
    main()
