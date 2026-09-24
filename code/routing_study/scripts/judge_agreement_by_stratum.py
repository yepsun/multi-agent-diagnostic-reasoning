#!/usr/bin/env python3
"""双判官分歧率按数据集与 ER 标签粒度分层。

缓存：GLM = judge_cache_glm_v3.json（53,511 对，主判官）、
DS-v3 = judge_cache_dsflash_v3.json（28,107 对，敏感性判官）；键均为
`gold[:150] + "||" + cand[:150]`，值 bool（True = 判为语义等价）。

缓存键不含 case_id，故逐例 JSONL 反向映射：对每个
(数据集, 策略, seed, 病例) 的 top-5 候选生成键，记录键 → 归属。
DS-v3 缓存冻结早于 ER-Reason seeds 2–5 跑批，因此 ER 只按 seed 1 归层；
CPC / MCR 为全 5 seeds。

用法：python routing_study/scripts/judge_agreement_by_stratum.py
输出：routing_study/results/judge_agreement_by_stratum.json + .md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
GLM_CACHE = B / "judge_cache_glm_v3.json"
DS_CACHE = B / "judge_cache_dsflash_v3.json"
OUT_JSON = B / "judge_agreement_by_stratum.json"
OUT_MD = B / "judge_agreement_by_stratum.md"

SEEDS = (1, 2, 3, 4, 5)
STRATEGIES = ("A×1", "P", "MDT")

# 与 recalc_erreason_judge.py:23-27 相同的症状级金标签正则
SYMPTOM = re.compile(
    r"unspecified|complains of|^pain|swelling|fever|hypoxia|syncope|dizziness|"
    r"nausea|vomiting|weakness|fatigue|fall,|suicidal|altered mental|headache|"
    r"bleeding|shortness of breath|chest pain|abdominal pain|back pain|rash|"
    r"edema|cough", re.I)


def key(gold: str, cand: str) -> str:
    return gold[:150] + "||" + cand[:150]


def load(p: Path) -> dict:
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


def run_path(dataset: str, strategy: str, seed: int) -> Path:
    if dataset == "CPC":
        stem = {"A×1": "Ax1", "P": "P", "MDT": "synthesis"}[strategy]
        return (B / "topn_seeds" / f"{stem}_s{seed}.jsonl" if strategy != "MDT"
                else (B / "topn_mdt/synthesis.jsonl" if seed == 1
                      else B / f"topn_mdt/s{seed}/synthesis.jsonl"))
    if dataset == "MCR":
        if seed == 1:
            return B / {"A×1": "topn_mcr/ax1.jsonl", "P": "topn_mcr/p.jsonl",
                        "MDT": "topn_mcr/mdt_synth.jsonl"}[strategy]
        if strategy == "MDT":
            return B / f"topn_mcr_seeds/s{seed}/mdt_synth.jsonl"
        return B / f"topn_mcr_seeds/{'Ax1' if strategy == 'A×1' else 'P'}_s{seed}.jsonl"
    stem = {"A×1": "ax1", "P": "p", "MDT": "mdt_synth"}[strategy]
    return (B / f"topn_erreason/{stem}.jsonl" if seed == 1
            else B / f"topn_erreason/s{seed}/{stem}.jsonl")


# DS-v3 缓存覆盖的运行：CPC/MCR 全 5 seeds × 3 策略；ER 仅 seed 1 × 3 策略
RUNS: list[tuple[str, str, int, Path]] = [
    ("CPC", strategy, seed, run_path("CPC", strategy, seed))
    for seed in SEEDS for strategy in STRATEGIES
] + [
    ("MCR", strategy, seed, run_path("MCR", strategy, seed))
    for seed in SEEDS for strategy in STRATEGIES
] + [
    ("ER", strategy, 1, run_path("ER", strategy, 1)) for strategy in STRATEGIES
]

# 逐例 JSONL 中每个键的归属：(数据集, seed, strategy, case_id)
membership: dict[str, set[tuple[str, int, str, str]]] = defaultdict(set)
run_keys: dict[tuple[str, str, int], set[str]] = {}
case_keys: dict[tuple[str, str, int, str], set[str]] = {}
run_n_cases: dict[tuple[str, str, int], int] = {}
for dataset, strategy, seed, path in RUNS:
    cases = load(path)
    run_n_cases[(dataset, strategy, seed)] = len(cases)
    ks: set[str] = set()
    for cid, rec in cases.items():
        cks = {key(rec["gold"], cand) for cand in rec["top5"][:5]}
        case_keys[(dataset, strategy, seed, cid)] = cks
        ks |= cks
        for k in cks:
            membership[k].add((dataset, seed, strategy, cid))
    run_keys[(dataset, strategy, seed)] = ks


def main() -> None:
    glm = json.loads(GLM_CACHE.read_text())
    ds = json.loads(DS_CACHE.read_text())
    inter = set(glm) & set(ds)
    covered = [k for k in inter if k in membership]
    ancillary = sorted(set(inter) - set(covered))

    def layer(keys) -> dict:
        dirs = Counter()
        n = 0
        for k in keys:
            if bool(glm[k]) != bool(ds[k]):
                n += 1
                dirs["glm_yes_ds_no" if glm[k] else "glm_no_ds_yes"] += 1
        return {
            "n_pairs": len(keys),
            "n_disagree": n,
            "disagree_rate": (n / len(keys)) if keys else None,
            # 方向命名沿用正文口径（deepseek-flash 判定 → GLM 判定）
            "directions": {
                "glm_yes_ds_no": dirs["glm_yes_ds_no"],
                "glm_no_ds_yes": dirs["glm_no_ds_yes"],
            },
        }

    overall = layer(inter)

    by_dataset: dict[str, dict] = {}
    for dataset in ("CPC", "MCR", "ER"):
        keys = [k for k in covered if any(m[0] == dataset for m in membership[k])]
        seeds = sorted({m[1] for k in keys for m in membership[k] if m[0] == dataset})
        block = layer(keys)
        block["seeds_covered"] = seeds
        block["n_cases"] = len({m[3] for k in keys for m in membership[k]
                                if m[0] == dataset})
        block["by_seed"] = {
            f"s{seed}": layer(sorted((run_keys[(dataset, "A×1", seed)]
                                      | run_keys[(dataset, "P", seed)]
                                      | run_keys[(dataset, "MDT", seed)]) & inter))
            for seed in seeds
        }
        block["by_strategy"] = {
            strategy: layer(sorted(
                set().union(*[run_keys[(dataset, strategy, seed)] for seed in seeds])
                & inter))
            for strategy in STRATEGIES
        }
        by_dataset[dataset] = block

    sub = {c["case_id"]: c for c in json.loads(
        (ROOT / "data" / "er_reason_subset.json").read_text())}
    er_strata: dict[str, dict] = {}
    for name, want_symptom in (("symptom", True), ("disease", False)):
        cases = [cid for cid in sub
                 if bool(SYMPTOM.search(sub[cid]["gold"])) is want_symptom]
        keys = sorted({k for cid in cases
                       for strategy in STRATEGIES
                       for k in case_keys[("ER", strategy, 1, cid)]} & inter)
        block = layer(keys)
        block["n_cases"] = len(cases)
        block["by_strategy"] = {
            strategy: layer(sorted(
                set().union(*[case_keys[("ER", strategy, 1, cid)] for cid in cases])
                & inter))
            for strategy in STRATEGIES
        }
        er_strata[name] = block

    # ER seeds 2–5 的键（DS 缓存未覆盖，只报告覆盖缺口）
    er_uncovered: dict[str, dict] = {}
    for seed in (2, 3, 4, 5):
        ks: set[str] = set()
        for strategy in STRATEGIES:
            for rec in load(run_path("ER", strategy, seed)).values():
                ks |= {key(rec["gold"], cand) for cand in rec["top5"][:5]}
        er_uncovered[f"s{seed}"] = {
            "n_keys": len(ks),
            "n_in_ds_cache": len(ks & set(ds)),
            "n_in_glm_cache": len(ks & set(glm)),
        }

    pub = {
        "evidence_bundle_overall": {"n_disagree": 1152, "n_pairs": 28107,
                                    "rate": 0.0410},
        "evidence_bundle_directions": {"glm_yes_ds_no": 991, "glm_no_ds_yes": 161},
        "evidence_bundle_er_s1": {
            "symptom": {"n_disagree": 112, "n_pairs": 2280},
            "disease": {"n_disagree": 156, "n_pairs": 2738},
        },
    }
    checks = {
        "overall_matches": (overall["n_disagree"] == 1152
                            and overall["n_pairs"] == 28107),
        "direction_matches": (overall["directions"]["glm_yes_ds_no"] == 991
                              and overall["directions"]["glm_no_ds_yes"] == 161),
        "er_symptom_matches": (er_strata["symptom"]["n_disagree"] == 112
                               and er_strata["symptom"]["n_pairs"] == 2280),
        "er_disease_matches": (er_strata["disease"]["n_disagree"] == 156
                               and er_strata["disease"]["n_pairs"] == 2738),
    }

    payload = {
        "caches": {"glm_v3": len(glm), "dsflash_v3": len(ds),
                   "intersection": len(inter)},
        "method": (
            "keys gold[:150]+'||'+cand[:150]; attributed to (dataset, seed, "
            "strategy, case) by regenerating them from the per-case top-5 "
            "JSONLs; DS-v3 covers CPC/MCR at all five seeds and ER at seed 1 "
            "only; direction labels follow the manuscript convention "
            "(deepseek-flash verdict -> GLM verdict)"),
        "n_covered_keys": len(covered),
        "n_ancillary_keys": len(ancillary),
        "overall": overall,
        "by_dataset": by_dataset,
        "ancillary": layer(ancillary),
        "er_strata": er_strata,
        "er_seeds_2_5_glm_only": er_uncovered,
        "publication_check": {**pub, **checks},
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"写出 {OUT_JSON.relative_to(ROOT)}")

    def cell(block: dict) -> str:
        return f"{block['n_disagree']}/{block['n_pairs']}"

    def rate(block: dict) -> str:
        return f"{block['disagree_rate'] * 100:.2f}%"

    def dirs(block: dict) -> str:
        return (f"{block['directions']['glm_yes_ds_no']} / "
                f"{block['directions']['glm_no_ds_yes']}")

    lines = [
        "# 双判官分歧率：按数据集与 ER 标签粒度分层",
        "",
        f"- GLM×v3 缓存 {len(glm):,} 对；DS-v3 缓存 {len(ds):,} 对；"
        f"交集 {len(inter):,} 对（DS 全部被 GLM 覆盖）。",
        "- 键 `gold[:150]+'||'+cand[:150]`；键→(数据集, seed, 策略, 病例) 由逐例 "
        "top-5 JSONL 反向重建。",
        f"- 可归入三数据集的键 {len(covered):,}；其余 {len(ancillary):,} 键属于"
        "辅助子研究/方法开发运行，单列为一块。",
        "- ER 仅按 seed 1 归层（DS-v3 缓存冻结早于 ER seeds 2–5 跑批）；"
        "CPC / MCR 覆盖全部 5 seeds。",
        "- 方向沿用正文口径（deepseek-flash 判定 → GLM 判定）：`GLM YES / DS NO` 与 "
        "`GLM NO / DS YES`。",
        "",
        "## 总体",
        "",
        "| 层 | 分歧对数 / 总对数 | 分歧率 | GLM YES/DS NO | GLM NO/DS YES |",
        "|---|---|---|---|---|",
        f"| 交集总体 | {cell(overall)} | {rate(overall)} | {dirs(overall)} |",
        f"| 三数据集可归属键 | {cell(layer(covered))} | {rate(layer(covered))} | "
        f"{dirs(layer(covered))} |",
        f"| 辅助子研究（未归入三数据集） | {cell(payload['ancillary'])} | "
        f"{rate(payload['ancillary'])} | {dirs(payload['ancillary'])} |",
        "",
        "复核 evidence_bundle.md（1,152 / 28,107 = 4.10%；991 / 161）："
        f"总数与分母 {'吻合' if checks['overall_matches'] else '不吻合'}，"
        f"方向拆分 {'吻合' if checks['direction_matches'] else '不吻合'}。",
        "",
        "## 按数据集分层",
        "",
        "| 数据集 | seeds | n 病例 | 分歧对数 / 总对数 | 分歧率 | GLM YES/DS NO | GLM NO/DS YES |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, block in by_dataset.items():
        seeds_txt = ",".join(f"s{s}" for s in block["seeds_covered"])
        lines.append(f"| {dataset} | {seeds_txt} | {block['n_cases']} | "
                     f"{cell(block)} | {rate(block)} | {dirs(block)} |")
    lines += ["", "### 各数据集逐 seed", "",
              "| 数据集 | seed | 分歧对数 / 总对数 | 分歧率 |", "|---|---|---|---|"]
    for dataset, block in by_dataset.items():
        for seed, sb in block["by_seed"].items():
            lines.append(f"| {dataset} | {seed} | {cell(sb)} | {rate(sb)} |")
    lines += ["", "### 各数据集逐策略", "",
              "| 数据集 | 策略 | 分歧对数 / 总对数 | 分歧率 |", "|---|---|---|---|"]
    for dataset, block in by_dataset.items():
        for strategy, sb in block["by_strategy"].items():
            lines.append(f"| {dataset} | {strategy} | {cell(sb)} | {rate(sb)} |")
    lines += ["",
              "（逐 seed / 逐策略行是各自独立的键集合；同一个 (gold, cand) 键可在多个 "
              "seed 或策略中重复出现，且分层只计一次，故行值不相加等于总体。）",
              ""]

    lines += ["", "## ER-Reason 按标签粒度（seed 1）", "",
              "| 粒度 | n 病例 | 分歧对数 / 总对数 | 分歧率 | GLM YES/DS NO | GLM NO/DS YES |",
              "|---|---|---|---|---|---|"]
    for name, block in er_strata.items():
        lines.append(f"| {name} | {block['n_cases']} | {cell(block)} | "
                     f"{rate(block)} | {dirs(block)} |")
    lines += ["",
              "复核已知 seed 1 结果（symptom 112/2280 = 4.91%、"
              "disease 156/2738 = 5.70%）："
              f"symptom {'吻合' if checks['er_symptom_matches'] else '不吻合'}，"
              f"disease {'吻合' if checks['er_disease_matches'] else '不吻合'}。",
              "两粒度层均由 seed 1 的金标准标签（SYMPTOM 正则，与 "
              "`recalc_erreason_judge.py:23-27` 相同）派生；ER 两层的分歧率均高于 "
              "CPC / MCR，疾病级略高于症状级。",
              "", "### ER 两粒度层逐策略", "",
              "| 粒度 | 策略 | 分歧对数 / 总对数 | 分歧率 |", "|---|---|---|---|"]
    for name, block in er_strata.items():
        for strategy, sb in block["by_strategy"].items():
            lines.append(f"| {name} | {strategy} | {cell(sb)} | {rate(sb)} |")

    lines += ["", "## ER seeds 2–5 的 DS 覆盖缺口", "",
              "| seed | 键数 | 在 DS 缓存中 | 在 GLM 缓存中 |", "|---|---|---|---|"]
    for seed, info in er_uncovered.items():
        lines.append(f"| {seed} | {info['n_keys']:,} | {info['n_in_ds_cache']:,} | "
                     f"{info['n_in_glm_cache']:,} |")
    lines += ["",
              "（ER seeds 2–5 的全部键都在 GLM 缓存中，只有约 1,600 键同时出现在 "
              "DS 缓存里——因为该键也出现在 seed 1 或辅助运行中；其余键没有敏感性"
              "判官判定，故 seeds 2–5 不参与分歧率分层。）",
              ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"写出 {OUT_MD.relative_to(ROOT)}")
    print(f"总体 {overall['n_disagree']}/{overall['n_pairs']} = "
          f"{overall['disagree_rate'] * 100:.2f}%；"
          f"方向 {dirs(overall)}")


if __name__ == "__main__":
    main()
