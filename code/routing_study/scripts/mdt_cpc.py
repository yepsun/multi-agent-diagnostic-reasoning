#!/usr/bin/env python3
"""MDT-同构：qwen3.8-flash 五角色独立 agent + 主持人汇总，87 例 CPC。

与 P（五视角同一上下文顺序扮演）的本质区别：五个角色 agent 相互隔离、
各自独立输出 top-5（含一行角色视角理由），最后由主持人读全病例 + 五份
意见做最终 top-5。检验"独立性"变量：隔离角色 vs 顺序角色（P）。

异构模式（MDT-异构，回应"同模型五角色共享盲点"质疑）：
- MDT_ROLE_PROVIDERS: 逗号分隔的 5 个 provider（按 ROLES 顺序），
  如 "qwen,deepseek-flash,qwen,deepseek-flash,qwen"。缺省全 qwen（行为与原版一致）。
- MDT_SYNTH_PROVIDER: 主持人 provider，缺省 qwen。
- MDT_OUTDIR: 输出目录名（routing_study/results/ 下），缺省 topn_mdt。
- MDT_SKIP_JUDGE=1: 跳过内置 deepseek-flash 判定阶段（异构实验改用
  judge_mdt_hetero_glm.py 的 GLM-v3 主判官口径）。

输出：routing_study/results/$MDT_OUTDIR/
  roles.jsonl（每 case×role 一行）、synthesis.jsonl、summary.json
对比基线：qwen3.8-flash A×1 / P（deepseek-flash 判官，topn_qwenmax/summary.json）
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
os.environ.setdefault("QWEN_MODEL", "qwen3.8-flash")

from run_inference import call_llm, call_llm_judge  # noqa: E402
from topn_cpc import load_done, append_row, MAX_WORKERS  # noqa: E402
from topn_cpc_promptv2_87 import load_merged  # noqa: E402
from webapp.prompts import extract_json  # noqa: E402
from webapp.clustering import same_disease  # noqa: E402

OUTDIR = (ROOT / "routing_study" / "results"
          / os.environ.get("MDT_OUTDIR", "topn_mdt"))
SEED = int(os.environ.get("MDT_SEED", "1"))
# s1 = 历史首轮（顶层目录）；s2+ 写入子目录，judge 缓存跨 seed 共享（内容寻址）
RUN_DIR = OUTDIR if SEED == 1 else OUTDIR / f"s{SEED}"
ROLES_PATH = RUN_DIR / "roles.jsonl"
SYNTH_PATH = RUN_DIR / "synthesis.jsonl"
SUMMARY_PATH = RUN_DIR / "summary.json"
JUDGE_CACHE = OUTDIR / "judge_cache_mdt.json"
SEED_CACHES = [ROOT / "routing_study" / "results" / "topn_dsflash" / "judge_cache_dsflash.json",
               ROOT / "routing_study" / "results" / "topn_qwenmax" / "judge_cache_qwenmax_dsflash.json",
               # 统一缓存（deepseek-flash 重判，rejudge_dsflash.py 产出）排最后，
               # 合并时其判定覆盖旧缓存
               ROOT / "routing_study" / "results" / "judge_cache_dsflash_unified.json"]
ROLE_WORKERS = 8
SYNTH_WORKERS = 4

ROLES = [
    ("Attending Internist",
     "overall clinical synthesis: which unifying diagnosis best explains the whole case, and which features are most discriminating"),
    ("Pathophysiologist",
     "underlying mechanism: what process could produce this constellation of findings, and hallmark histologic, laboratory, or molecular clues"),
    ("Imaging & Laboratory Specialist",
     "objective data: how to interpret the imaging, laboratory, and vital-sign patterns, and which patterns or paradoxes stand out"),
    ("Epidemiologist",
     "demographics, geography, travel, exposures, medications, occupation, and comorbidities — hidden risk factors in the narrative"),
    ("Skeptic / Challenger",
     "the strongest argument AGAINST the leading hypothesis: alternatives that explain more findings with fewer contradictions, and findings that remain unexplained"),
]

GUIDELINES = """Guidelines:
- A diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer.
- A striking organic finding may be an incidental companion finding.
- Combination diagnoses are allowed. For infections, name specific pathogens as separate items when clinically distinct."""

# per-role provider（按 ROLES 顺序）；缺省全 qwen，与原同构行为完全一致。
_raw_providers = [p.strip() for p in
                  os.environ.get("MDT_ROLE_PROVIDERS", "").split(",")
                  if p.strip()]
if _raw_providers and len(_raw_providers) != len(ROLES):
    raise SystemExit(
        f"MDT_ROLE_PROVIDERS 需为 {len(ROLES)} 个 provider，收到 {len(_raw_providers)}")
ROLE_PROVIDERS = _raw_providers or ["qwen"] * len(ROLES)
SYNTH_PROVIDER = os.environ.get("MDT_SYNTH_PROVIDER", "qwen")


def key_of(gold, cand):
    return f"{gold[:150]}||{cand[:150]}"


def parse_top5(raw):
    data = extract_json(raw) or {}
    items = data.get("top5") or []
    out = []
    for it in items[:5]:
        if not isinstance(it, dict):
            continue
        dx = str(it.get("diagnosis") or "").strip()
        if dx and dx.lower() not in [x["diagnosis"].lower() for x in out]:
            out.append({"diagnosis": dx,
                        "rationale": str(it.get("rationale") or "").strip()})
    return out


def call_role(role_title, role_brief, case_text, temperature=0.3,
              provider="qwen", disable_thinking=True):
    prompt = f"""You are the {role_title} on an MDT panel reviewing an MGH CPC case.

Your assigned lens: {role_brief}

Through this lens, list the 5 candidate diagnoses that best fit the case, ranked most to least likely FROM YOUR PERSPECTIVE. Include candidates another specialist might overlook if the findings support them.

{GUIDELINES}

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "...", "rationale": "one line through your lens"}}, ... exactly 5 items]}}

Case:
{case_text}
"""
    raw, usage = call_llm(prompt, temperature=temperature, max_tokens=4096,
                          timeout=300, provider=provider,
                          disable_thinking=disable_thinking)
    return parse_top5(raw), (usage or {}).get("total_tokens", 0)


def run_roles(cases):
    path = ROLES_PATH
    done = {(r["case_id"], r["role"]) for r in map(json.loads, open(path)) } if path.exists() else set()
    todo = [(c, title, brief, prov)
            for c in cases
            for (title, brief), prov in zip(ROLES, ROLE_PROVIDERS)
            if (c["case_id"], title) not in done]
    print(f"[角色] 已完成 {len(done)}，待跑 {len(todo)}（{len(cases)} 例 × 5 角色）",
          flush=True)
    with ThreadPoolExecutor(ROLE_WORKERS) as ex:
        futs = {ex.submit(call_role, t, b, c["text"], provider=prov): (c, t, prov)
                for c, t, b, prov in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c, role, prov = futs[fut]
            try:
                top5, tokens = fut.result()
            except Exception as e:
                print(f"[角色] {c['case_id'][:30]} {role} 失败: {e}", flush=True)
                continue
            append_row(path, {"case_id": c["case_id"], "role": role,
                              "provider": prov,
                              "top5": top5, "total_tokens": tokens})
            if i % 25 == 0 or i == len(todo):
                print(f"[角色] {i}/{len(todo)}", flush=True)


def run_synthesis(cases):
    path = SYNTH_PATH
    done = load_done(path)
    roles_by = {}
    for r in map(json.loads, open(ROLES_PATH)):
        roles_by.setdefault(r["case_id"], {})[r["role"]] = r["top5"]

    todo = []
    for c in cases:
        if c["case_id"] in done:
            continue
        opinions = roles_by.get(c["case_id"], {})
        if len(opinions) < len(ROLES):
            continue
        lines = []
        for title, _ in ROLES:
            lines.append(f"{title}:")
            for i, item in enumerate(opinions[title][:5], 1):
                line = f"  {i}. {item['diagnosis']}"
                if item.get("rationale"):
                    line += f" — {item['rationale']}"
                lines.append(line)
        todo.append((c, "\n".join(lines)))
    print(f"[汇总] 已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    def work(c, opinions_text):
        prompt = f"""You are the moderator of an MDT panel. Five specialists independently reviewed the MGH CPC case below, each through their own lens, without seeing each other's opinions. Their ranked candidate lists are given.

Integrate them into the final ranked top-5 for the case:
- Candidates supported by multiple specialists generally rise.
- A unique candidate with specific, case-grounded support must NOT be dropped merely because only one specialist listed it.
- Resolve conflicts by re-checking against the case text.
- Combination diagnoses are allowed.

Respond with ONLY a JSON object, no other text:
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, ... exactly 5 items]}}

Case:
{c['text']}

Panel opinions:
{opinions_text}
"""
        raw, usage = call_llm(prompt, temperature=0.3, max_tokens=4096,
                              timeout=300, provider=SYNTH_PROVIDER,
                              disable_thinking=True)
        data = extract_json(raw) or {}
        items = data.get("top5") or []
        top5 = []
        for it in items[:5]:
            dx = (str(it.get("diagnosis") or "").strip()
                  if isinstance(it, dict) else str(it or "").strip())
            if dx and dx.lower() not in [x.lower() for x in top5]:
                top5.append(dx)
        return {"case_id": c["case_id"], "gold": c["gold"], "top5": top5,
                "total_tokens": (usage or {}).get("total_tokens", 0)}

    with ThreadPoolExecutor(SYNTH_WORKERS) as ex:
        futs = {ex.submit(work, c, ops): c for c, ops in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            append_row(path, fut.result())
            if i % 10 == 0 or i == len(todo):
                print(f"[汇总] {i}/{len(todo)}", flush=True)


def judge_phase(rows):
    cache = {}
    for seed in SEED_CACHES:
        if seed.exists():
            cache.update(json.loads(seed.read_text()))
    if JUDGE_CACHE.exists():
        cache.update(json.loads(JUDGE_CACHE.read_text()))
    jobs = {}
    for row in rows:
        for cand in row["top5"][:5]:
            key = key_of(row["gold"], cand)
            if key not in cache:
                jobs[key] = (row["gold"], cand)
    print(f"[判定] 缓存 {len(cache)}，待判 {len(jobs)}", flush=True)
    unresolved = set(jobs.keys())
    for round_no in (1, 2, 3):
        if not unresolved:
            break
        items = jobs.items() if round_no == 1 else [(k, jobs[k]) for k in unresolved]
        nxt = []

        def work(item):
            key, (gold, cand) = item
            try:
                raw, _ = call_llm_judge(JUDGE_PROMPT_T(gold, cand), timeout=60,
                                        max_retries=2)
                v = (raw or "").strip().upper()
                if v.startswith("YES"):
                    return key, True
                if v.startswith("NO"):
                    return key, False
                return key, None
            except Exception:
                return key, None

        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for key, verdict in ex.map(work, list(items)):
                if verdict is None:
                    nxt.append(key)
                else:
                    cache[key] = verdict
        unresolved = set(nxt)
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[判定] 缓存 {len(cache)} 条，未解析 {len(unresolved)}", flush=True)

    def hits(row):
        return [bool(cache.get(key_of(row["gold"], c))) for c in row["top5"][:5]]

    return hits


def JUDGE_PROMPT_T(gold, cand):
    return f"""You are a medical evaluation judge. A model produced a diagnosis for a clinical case.
Reference (correct) diagnosis: {gold}
Model's diagnosis: {cand}

Judging rules (apply in order):
1. CORRECT (YES) if the model's diagnosis names the same DISEASE as the reference —
   synonyms, abbreviations, and translations are acceptable.
2. The model adding extra findings, complications, etiologies, or secondary diagnoses
   does NOT make it wrong, as long as the reference disease is named as (part of) the
   main diagnosis.
3. Missing qualifiers in the reference (disease stage, severity, "in remission",
   anatomic subtype, etiologic form) do NOT make the model wrong: e.g. "celiac disease"
   is correct for "celiac disease in histologic remission"; "tularemia" is correct for
   "ulceroglandular tularemia"; "acute pulmonary embolism" is correct for "acute massive
   pulmonary embolism with clot in transit".
4. WRONG (NO) only if the model names a DIFFERENT disease as its main diagnosis, or a
   generic category that never specifically names the reference disease.
Answer with exactly one word: YES or NO."""


def summarize_and_compare(rows, hits_fn):
    n = len(rows)
    stat = {"n": n, "top1": 0, "top3": 0, "top5": 0}
    details = []
    for row in rows:
        verdicts = hits_fn(row)
        hit = next((r for r, v in enumerate(verdicts, start=1) if v), None)
        if hit:
            if hit == 1:
                stat["top1"] += 1
            if hit <= 3:
                stat["top3"] += 1
            stat["top5"] += 1
        details.append({"case_id": row["case_id"], "top5": row["top5"],
                        "verdicts": verdicts, "hit": hit})
    for k in (1, 3, 5):
        stat[f"top{k}_acc"] = round(stat[f"top{k}"] / n, 3) if n else None
    print(f"MDT: n={n} top1 {stat['top1']} ({stat['top1_acc']}) "
          f"top3 {stat['top3']} ({stat['top3_acc']}) "
          f"top5 {stat['top5']} ({stat['top5_acc']})", flush=True)

    base = json.loads((ROOT / "routing_study" / "results" / "topn_qwenmax"
                       / "summary.json").read_text())["qwen3.8-flash_dsflash_judge"]
    print("\n== 对比（同 87 例、同判官 deepseek-flash）==", flush=True)
    print(f"{'':8s} {'top-1':>8s} {'top-3':>8s} {'top-5':>8s}")
    print(f"{'A×1':10s} {base['Ax1']['top1_acc']:>8.1%} {base['Ax1']['top3_acc']:>8.1%} {base['Ax1']['top5_acc']:>8.1%}")
    print(f"{'P':10s} {base['P']['top1_acc']:>8.1%} {base['P']['top3_acc']:>8.1%} {base['P']['top5_acc']:>8.1%}")
    print(f"{'MDT':10s} {stat['top1_acc']:>8.1%} {stat['top3_acc']:>8.1%} {stat['top5_acc']:>8.1%}")
    return {**stat, "details": details}


def diversity_stats(rows, roles_path):
    """agent 意见两两分歧度 + 最终 top5 中独有候选占比。"""
    roles_by = {}
    for r in map(json.loads, open(roles_path)):
        roles_by.setdefault(r["case_id"], []).append([x["diagnosis"] for x in r["top5"]])
    dis = []
    unique_kept = 0
    total_top5 = 0
    for row in rows:
        lists = roles_by.get(row["case_id"], [])
        if len(lists) < 2:
            continue
        # 两两不一致：1 - (同病交集/5)
        import itertools
        pair_dis = []
        for la, lb in itertools.combinations(lists, 2):
            same = sum(1 for a in la if any(same_disease(a, b) for b in lb))
            pair_dis.append(1 - same / 5)
        dis.append(sum(pair_dis) / len(pair_dis))
        # 统计每个最终 top5 候选由几个 agent 提出
        for cand in row["top5"][:5]:
            total_top5 += 1
            proponents = sum(1 for lst in lists
                             if any(same_disease(cand, x) for x in lst))
            if proponents == 1:
                unique_kept += 1
    if dis:
        print(f"角色意见平均两两分歧度: {sum(dis)/len(dis):.2f} "
              f"(0=完全一致, 1=完全不同) | 最终 top5 中仅 1 个 agent 提出的候选: "
              f"{unique_kept}/{total_top5}", flush=True)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_merged()
    role_cfg = ", ".join(f"{t}={p}" for (t, _), p in zip(ROLES, ROLE_PROVIDERS))
    print(f"数据集: {len(cases)} 例 | seed: s{SEED} | 输出: {RUN_DIR}\n"
          f"结构: 5 角色 agent 独立 → 主持人汇总 | 角色: {role_cfg} | "
          f"主持人: {SYNTH_PROVIDER}", flush=True)

    run_roles(cases)
    run_synthesis(cases)

    rows = list(load_done(SYNTH_PATH).values())
    if os.environ.get("MDT_SKIP_JUDGE"):
        diversity_stats(rows, ROLES_PATH)
        SUMMARY_PATH.write_text(json.dumps(
            {"n": len(rows), "judged": False,
             "role_providers": {t: p for (t, _), p in zip(ROLES, ROLE_PROVIDERS)},
             "synth_provider": SYNTH_PROVIDER},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已跳过内置判定（MDT_SKIP_JUDGE），写入 {SUMMARY_PATH}", flush=True)
        return
    hits_fn = judge_phase(rows)
    summary = summarize_and_compare(rows, hits_fn)
    diversity_stats(rows, ROLES_PATH)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
