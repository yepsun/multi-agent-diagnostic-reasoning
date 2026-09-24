#!/usr/bin/env python3
"""Judge v3：基于双人盲评错误分析修订判定规则，并与 v2 对比验证。

v3 新增条款（来自 19 条系统性偏差的模式归纳，非条目级拟合）：
- 3c 家族词兜底：更宽的疾病家族词若明确包含 gold 为其亚型 → 正确
  （如 lymphoma ↔ Hodgkin lymphoma、thyroiditis ↔ Hashimoto thyroiditis）
- 3d 复合诊断主成分：gold 为复合诊断时，答出其主要成分 → 正确
  （如 AIHA ↔ Evans 综合征、CMV 结肠炎 ↔ CMV 结肠炎+IBD）
- 3e 亚型粗化：gold 的相邻/未分化亚型 → 正确
  （如 borderline lepromatous ↔ lepromatous、basaloid SCC ↔ SCC）
- 4 保留：不同的疾病实体（angiolipoma ≠ lipoma 类）仍判错。

验证：v3 与 v2 在 100 条盲评样本上的判定对比 + 与人类 κ。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "routing_study" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_inference import call_llm  # noqa: E402
from judge_study import load_sample, cohens_kappa  # noqa: E402

V3_PROMPT = """You are a medical evaluation judge. A model produced a diagnosis for a clinical case.
Reference (correct) diagnosis: {gold}
Model's diagnosis: {pred}

Judging rules (apply in order):
1. CORRECT (YES) if the model's diagnosis names the same DISEASE as the reference —
   synonyms, abbreviations, and translations are acceptable.
2. The model adding extra findings, complications, etiologies, or secondary diagnoses
   does NOT make it wrong, as long as the reference disease is named as (part of) the
   main diagnosis.
3. Missing qualifiers in the reference (disease stage, severity, "in remission",
   anatomic subtype, etiologic form) do NOT make the model wrong.
3b. A BROADER disease-family or category term that clearly includes the reference
   disease as one of its subtypes or variants is CORRECT (e.g., "lymphoma" for
   "Hodgkin lymphoma"; "thyroiditis" for "Hashimoto thyroiditis"; "muscular
   dystrophy" for "limb-girdle muscular dystrophy type 2B").
3c. Naming the dominant component of a COMPOUND reference diagnosis is CORRECT
   (e.g., "autoimmune hemolytic anemia" for "Evans syndrome (AIHA +
   thrombocytopenia)"; "CMV colitis" for "CMV colitis and inflammatory bowel disease").
3d. A NEIGHBORING or less-specified subtype within the same disease entity is
   CORRECT (e.g., "borderline lepromatous leprosy" for "lepromatous leprosy";
   "basaloid squamous cell carcinoma" for "squamous cell carcinoma").
4. WRONG (NO) only if the model names a genuinely DIFFERENT disease entity as its
   main diagnosis — a distinct entity, not merely a broader category, a component
   of a compound diagnosis, or a neighboring subtype.
Answer with exactly one word: YES or NO."""


def v3_verdict(gold, cand):
    raw, _ = call_llm(V3_PROMPT.format(gold=gold, pred=cand), temperature=0.0,
                      max_tokens=10, timeout=60, provider="deepseek-flash",
                      disable_thinking=True)
    v = (raw or "").strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    raise RuntimeError(f"unparseable: {raw[:40]!r}")


def main():
    sample = load_sample()
    out_path = Path(ROOT / "routing_study" / "results" / "judge_validity"
                    / "v3_verdicts.json")
    verdicts = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [t for t in sample if str(t["item_id"]) not in verdicts]
    print(f"v3 判定：已完成 {len(verdicts)}，待判 {len(todo)}", flush=True)
    for t in todo:
        try:
            verdicts[str(t["item_id"])] = v3_verdict(t["gold"], t["candidate"])
        except Exception as e:
            print(f"  item {t['item_id']} 失败: {e}", flush=True)
    out_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1))

    llm2 = {t["item_id"]: t["llm_verdict"] for t in sample}
    a = json.loads((ROOT / "routing_study" / "results" / "judge_validity"
                    / "annotations_a.json").read_text())
    b = json.loads((ROOT / "routing_study" / "results" / "judge_validity"
                    / "annotations_b.json").read_text())
    ids = sorted(int(k) for k in a)
    va = [a[str(i)] for i in ids]
    vb = [b[str(i)] for i in ids]
    v2 = [llm2[i] for i in ids]
    v3 = [verdicts[str(i)] for i in ids]
    majority = [x or y for x, y in zip(va, vb)]

    flips = [(i, x, y) for i, (x, y) in enumerate(zip(v2, v3)) if x != y]
    print(f"\nv2→v3 翻转 {len(flips)} 条（全部应为 NO→YES 方向: "
          f"{sum(1 for _, x, _ in flips if not x)}/{len(flips)}）")
    print("\n===== 100 条盲评样本上的 κ 对比 =====")
    for tag, judge in (("v2 (现行)", v2), ("v3 (修订)", v3)):
        print(f"-- {tag} --")
        print(f"  vs 评者A:      {cohens_kappa(va, judge)}")
        print(f"  vs 评者B:      {cohens_kappa(vb, judge)}")
        print(f"  vs 多数票:     {cohens_kappa(majority, judge)}")
    print(f"  评者A vs B:    {cohens_kappa(va, vb)}")
    (ROOT / "routing_study" / "results" / "judge_validity" / "v3_report.json").write_text(
        json.dumps({"flips": len(flips), "flip_direction_no_to_yes":
                    sum(1 for _, x, _ in flips if not x),
                    "kappa": {"A_vs_v2": cohens_kappa(va, v2),
                              "A_vs_v3": cohens_kappa(va, v3),
                              "B_vs_v2": cohens_kappa(vb, v2),
                              "B_vs_v3": cohens_kappa(vb, v3),
                              "maj_vs_v2": cohens_kappa(majority, v2),
                              "maj_vs_v3": cohens_kappa(majority, v3)}},
                   ensure_ascii=False, indent=1))
    print("已写 v3_report.json / v3_verdicts.json", flush=True)


if __name__ == "__main__":
    main()
