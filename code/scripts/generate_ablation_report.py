#!/usr/bin/env python3
"""
Aggregate ablation results for groups 1-3 and generate a markdown report.

Supports any schemes found in result files.
"""

import json
import os
import glob
from collections import defaultdict
from datetime import datetime

RESULT_DIR = "../results/ablation"
OUTPUT_MD = "../results/ablation/ablation_report_groups_1_2_3.md"


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short_case_id(case_id, max_len=70):
    return case_id[:max_len] + "..." if len(case_id) > max_len else case_id


def main():
    # Load all result files for groups 1-3
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "group[123]_*_*.json")))

    grouped = defaultdict(dict)  # group -> scheme -> data
    for path in files:
        data = load_result(path)
        group = data.get("group")
        scheme = data.get("scheme")
        if group is None or scheme is None:
            continue
        grouped[group][scheme] = data

    if not grouped:
        print("No result files found")
        return

    # Determine scheme order: prefer A, P, B, then others alphabetically
    all_schemes = set()
    for schemes in grouped.values():
        all_schemes.update(schemes.keys())
    preferred_order = [s for s in ["A", "B"] if s in all_schemes]
    remaining = sorted(all_schemes - set(preferred_order))
    scheme_order = preferred_order + remaining

    # Build case_id -> group -> scheme -> result
    cases = defaultdict(lambda: defaultdict(dict))
    for group, schemes in grouped.items():
        for scheme, data in schemes.items():
            for r in data.get("results", []):
                cases[r["case_id"]][group][scheme] = r

    # Aggregate per-scheme totals
    scheme_totals = defaultdict(lambda: {"matches": 0, "total": 0, "llm": 0, "time": 0, "pubmed": 0})
    group_summaries = defaultdict(dict)

    for group, schemes in grouped.items():
        for scheme, data in schemes.items():
            summary = data.get("summary", {})
            group_summaries[group][scheme] = summary
            scheme_totals[scheme]["matches"] += summary.get("matches", 0)
            scheme_totals[scheme]["total"] += summary.get("total_cases", 0)
            scheme_totals[scheme]["llm"] += summary.get("total_llm_calls", 0)
            scheme_totals[scheme]["pubmed"] += summary.get("total_pubmed_queries", 0)
            scheme_totals[scheme]["time"] += summary.get("total_time", 0)

    # Build markdown
    lines = []
    scheme_names_str = ", ".join(f"Scheme {s}" for s in scheme_order)
    lines.append(f"# {scheme_names_str} 消融实验报告（Groups 1–3，共 30 例）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 实验设计")
    lines.append("")
    for scheme in scheme_order:
        desc = {
            "A": "单次 LLM 调用，纯 zero-shot CoT 提示，无检索。",
            "B": "自适应检索诊断：先评估置信度，低置信度时触发 PubMed 检索，经 NLI verifier 过滤后做最终诊断。",
        }.get(scheme, f"Scheme {scheme}")
        lines.append(f"- **Scheme {scheme}**：{desc}")
    lines.append("- 数据集：MGH CPC group 1、2、3，每组 10 例，共 30 例。")
    lines.append("")

    lines.append("## 总体结果")
    lines.append("")
    header = "| 方案 | 正确数 / 总数 | 准确率 | 总 LLM 调用 | 总 PubMed 查询 | 总耗时(s) | 每例平均耗时(s) |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (header.count("|") - 1)) + "|")
    for scheme in scheme_order:
        t = scheme_totals[scheme]
        acc = t["matches"] / t["total"] * 100 if t["total"] else 0
        avg_time = t["time"] / t["total"] if t["total"] else 0
        lines.append(
            f"| Scheme {scheme} | {t['matches']}/{t['total']} | {acc:.1f}% | "
            f"{t['llm']} | {t['pubmed']} | {t['time']:.1f} | {avg_time:.1f} |"
        )
    lines.append("")

    lines.append("## 分组结果")
    lines.append("")
    lines.append("| Group | " + " | ".join(f"Scheme {s}" for s in scheme_order) + " |")
    lines.append("|" + "|".join(["---"] * (len(scheme_order) + 1)) + "|")
    for group in sorted(group_summaries.keys()):
        summaries = group_summaries[group]
        row = [f"Group {group}"]
        for scheme in scheme_order:
            s = summaries.get(scheme, {})
            acc = s.get("accuracy", 0) * 100
            row.append(f"{s.get('matches', 0)}/{s.get('total_cases', 0)} ({acc:.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Pairwise agreement analysis for all scheme pairs
    lines.append("## 一致性分析")
    lines.append("")

    scheme_pairs = []
    for i, s1 in enumerate(scheme_order):
        for s2 in scheme_order[i + 1:]:
            scheme_pairs.append((s1, s2))

    if scheme_pairs:
        for s1, s2 in scheme_pairs:
            both_correct = 0
            both_wrong = 0
            s1_correct_only = 0
            s2_correct_only = 0
            disagreement_cases = []

            for case_id, group_data in cases.items():
                # Flatten across groups
                r1 = None
                r2 = None
                gold = None
                for group, schemes in group_data.items():
                    if s1 in schemes:
                        r1 = schemes[s1]
                        gold = schemes[s1].get("gold")
                    if s2 in schemes:
                        r2 = schemes[s2]
                        if not gold:
                            gold = schemes[s2].get("gold")

                if r1 is None or r2 is None or r1.get("error") or r2.get("error"):
                    continue

                m1 = r1.get("match", False)
                m2 = r2.get("match", False)
                if m1 and m2:
                    both_correct += 1
                elif not m1 and not m2:
                    both_wrong += 1
                elif m1:
                    s1_correct_only += 1
                else:
                    s2_correct_only += 1

                if m1 != m2:
                    disagreement_cases.append({
                        "case_id": case_id,
                        "gold": gold,
                        "s1_final": r1.get("final_diagnosis", "N/A"),
                        "s2_final": r2.get("final_diagnosis", "N/A"),
                        "s1_correct": m1,
                    })

            lines.append(f"### Scheme {s1} vs Scheme {s2}")
            lines.append("")
            lines.append(f"- 都正确：{both_correct}")
            lines.append(f"- 都错误：{both_wrong}")
            lines.append(f"- 仅 Scheme {s1} 正确：{s1_correct_only}")
            lines.append(f"- 仅 Scheme {s2} 正确：{s2_correct_only}")
            lines.append("")

            if disagreement_cases:
                lines.append("#### 分歧病例")
                lines.append("")
                for item in disagreement_cases:
                    winner = s1 if item["s1_correct"] else s2
                    lines.append(f"- **{short_case_id(item['case_id'])}")
                    lines.append(f"  - 金标准：{item['gold']}")
                    lines.append(f"  - 仅 Scheme {winner} 正确")
                    lines.append(f"  - {s1} 输出：{item['s1_final']}")
                    lines.append(f"  - {s2} 输出：{item['s2_final']}")
                    lines.append("")
    else:
        lines.append("只有一个方案，无需一致性分析。")
        lines.append("")

    # Per-case table
    lines.append("## 逐例结果")
    lines.append("")
    lines.append("| # | Group | Case ID | Gold Standard | " + " | ".join(f"{s} 结果" for s in scheme_order) + " |")
    lines.append("|" + "|".join(["---"] * (4 + len(scheme_order))) + "|")

    idx = 1
    for group in sorted(group_summaries.keys()):
        scheme_results = grouped[group]
        # Pair results by case_id for each scheme
        by_id = {s: {r["case_id"]: r for r in scheme_results[s].get("results", [])} for s in scheme_order if s in scheme_results}
        case_ids = set()
        for s in by_id.values():
            case_ids.update(s.keys())

        for case_id in sorted(case_ids):
            gold = ""
            for s in scheme_order:
                if s in by_id and case_id in by_id[s]:
                    gold = by_id[s][case_id].get("gold", "")
                    break

            marks = []
            for s in scheme_order:
                if s in by_id and case_id in by_id[s]:
                    r = by_id[s][case_id]
                    if r.get("error"):
                        marks.append("ERR")
                    elif r.get("match"):
                        marks.append("✓")
                    else:
                        marks.append("✗")
                else:
                    marks.append("-")

            gold_short = short_case_id(gold, max_len=50)
            cid_short = short_case_id(case_id, max_len=50)
            lines.append(f"| {idx} | {group} | {cid_short} | {gold_short} | " + " | ".join(marks) + " |")
            idx += 1
    lines.append("")

    # Wrong cases summary: cases where all available schemes are wrong
    lines.append("## 各方案均错误的病例")
    lines.append("")
    all_wrong = []
    for case_id, group_data in cases.items():
        for group, schemes in group_data.items():
            if all(not schemes.get(s, {}).get("match", False) and not schemes.get(s, {}).get("error")
                   for s in scheme_order if s in schemes):
                gold = ""
                finals = {}
                for s in scheme_order:
                    if s in schemes:
                        gold = schemes[s].get("gold", "")
                        finals[s] = schemes[s].get("final_diagnosis", "N/A")
                        break
                all_wrong.append({
                    "case_id": case_id,
                    "gold": gold,
                    "finals": finals,
                })
                break

    lines.append(f"共 {len(all_wrong)} 例：")
    lines.append("")
    for item in all_wrong:
        lines.append(f"- **{short_case_id(item['case_id'])}")
        lines.append(f"  - 金标准：{item['gold']}")
        for s, final in item["finals"].items():
            lines.append(f"  - Scheme {s} 输出：{final}")
        lines.append("")

    # Summary conclusion
    lines.append("## 结论")
    lines.append("")
    for scheme in scheme_order:
        t = scheme_totals[scheme]
        acc = t["matches"] / t["total"] * 100 if t["total"] else 0
        lines.append(
            f"- Scheme {scheme}：准确率 **{acc:.1f}%** ({t['matches']}/{t['total']})，"
            f"总 LLM 调用 {t['llm']}，总 PubMed 查询 {t['pubmed']}，总耗时 {t['time']:.1f}s。"
        )
    lines.append("")

    # Identify best scheme
    if scheme_order:
        best_scheme = max(scheme_order, key=lambda s: scheme_totals[s]["matches"] / max(scheme_totals[s]["total"], 1))
        best_acc = scheme_totals[best_scheme]["matches"] / max(scheme_totals[best_scheme]["total"], 1) * 100
        lines.append(f"- 在 30 例样本上，Scheme {best_scheme} 表现最好，准确率为 **{best_acc:.1f}%**。")
    lines.append("")

    # Write markdown
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {OUTPUT_MD}")
    for scheme in scheme_order:
        t = scheme_totals[scheme]
        acc = t["matches"] / t["total"] * 100 if t["total"] else 0
        print(f"  Scheme {scheme}: {acc:.1f}% ({t['matches']}/{t['total']})")


if __name__ == "__main__":
    main()
