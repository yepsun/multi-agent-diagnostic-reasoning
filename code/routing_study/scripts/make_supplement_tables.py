#!/usr/bin/env python3
"""Generate Supplementary Table S1 (per-seed accuracy) and Table S4 (BH
correction) numbers for paper/supplement.md.

S1: CPC (n=87), MCR (n=406) and ER-Reason (n=364), strategies A×1/P/MDT,
seeds 1-5, top-1/3/5 per-seed accuracy (%) under the primary judge
(GLM-5.3-flash × v3 rules, judge_cache_glm_v3.json). Mean ± SD across seeds is
recomputed as a sanity check against manuscript Table 1: the CPC/MCR values must
reproduce the case-level means of stats_caselevel.json, and the ER-Reason values
must reproduce the per-seed figures of topn_erreason/seed_summary.json.

S4: Benjamini-Hochberg q values across the 27 tests of the primary analysis
framework (3 datasets × 3 comparisons × 3 endpoints). CPC and MCR contribute
paired Wilcoxon p values from stats_caselevel.json; ER-Reason contributes the
case-level Wilcoxon p values of the five-seed run, read from
topn_erreason/seed_summary.json (produced by erreason_5seeds.py). The legacy
single-seed exact McNemar statistics for ER-Reason are still recomputed and
printed, but only as a superseded diagnostic — they are not part of S4.

Prerequisites (all local paths, none of them shipped in the public release):
  - routing_study/results/judge_cache_glm_v3.json (primary judge cache)
  - routing_study/results/stats_caselevel.json (CPC/MCR case-level stats)
  - routing_study/results/topn_erreason/seed_summary.json (ER 5-seed stats)
  - routing_study/results/topn_seeds/, topn_mdt/, topn_mcr/, topn_mcr_seeds/,
    topn_erreason/ (+ s2-s5) — the raw per-case jsonl outputs.
The ER-Reason outputs fall under the ER-Reason data use agreement and are
excluded from the public repository snapshot (see paper/dua_compliance_scan.md
§4 and paper/submission_checklist.md A3.2/A12.3); without them the script skips
the ER-Reason rows and prints a warning instead of failing.

Usage: python routing_study/scripts/make_supplement_tables.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "routing_study" / "results"
CACHE = B / "judge_cache_glm_v3.json"
ER_SUMMARY = B / "topn_erreason" / "seed_summary.json"

J = json.loads(CACHE.read_text())
key = lambda g, c: g[:150] + "||" + c[:150]


def load(p):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in open(p) if l.strip()}


runs = {}
for s in range(1, 6):
    runs[("cpc", "Ax1", s)] = load(B / f"topn_seeds/Ax1_s{s}.jsonl")
    runs[("cpc", "P", s)] = load(B / f"topn_seeds/P_s{s}.jsonl")
    runs[("cpc", "MDT", s)] = load(
        B / "topn_mdt/synthesis.jsonl" if s == 1
        else B / f"topn_mdt/s{s}/synthesis.jsonl")
    runs[("mcr", "Ax1", s)] = load(
        B / "topn_mcr/ax1.jsonl" if s == 1
        else B / f"topn_mcr_seeds/Ax1_s{s}.jsonl")
    runs[("mcr", "P", s)] = load(
        B / "topn_mcr/p.jsonl" if s == 1
        else B / f"topn_mcr_seeds/P_s{s}.jsonl")
    runs[("mcr", "MDT", s)] = load(
        B / "topn_mcr/mdt_synth.jsonl" if s == 1
        else B / f"topn_mcr_seeds/s{s}/mdt_synth.jsonl")

# ER-Reason seed 1 = topn_erreason/ top level, seeds 2-5 = topn_erreason/s{2..5}/
ER = B / "topn_erreason"
er_runs = {}
for s in range(1, 6):
    d = ER if s == 1 else ER / f"s{s}"
    for m, f in (("Ax1", "ax1"), ("P", "p"), ("MDT", "mdt_synth")):
        p = d / f"{f}.jsonl"
        if p.exists():
            er_runs[(m, s)] = load(p)
er_raw = len(er_runs) == 15
if er_raw:
    runs.update({("er", m, s): v for (m, s), v in er_runs.items()})
else:
    print(f"[warn] ER-Reason per-case outputs incomplete under {ER} "
          f"({len(er_runs)}/15 files); S1/S4 ER-Reason rows come from "
          f"{ER_SUMMARY.name} only")


def topk(flags, k):
    f = flags[:k]
    if not any(x is not None for x in f):
        return None
    return any(x is True for x in f)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n,
               1.0)


if not er_raw and not ER_SUMMARY.exists():
    sys.exit(f"[error] neither the ER-Reason raw outputs nor {ER_SUMMARY} "
             f"are available; cannot rebuild S1/S4")

# ------------------------------------------------------- verdict tabs
# hits[(ds, m, s, cid)] = list of verdicts for the top-5 candidates
hits = {}
missing = 0
for (ds, m, s), cases in runs.items():
    for cid, rec in cases.items():
        flags = []
        for c in rec["top5"][:5]:
            k = key(rec["gold"], c)
            if k in J:
                flags.append(bool(J[k]))
            else:
                missing += 1
                flags.append(None)
        hits[(ds, m, s, cid)] = flags
print(f"judge cache: {len(J)} pairs | missing verdicts in the tabulated runs: "
      f"{missing}")

# ------------------------------------------------------- ER from seed_summary
er_summ = json.loads(ER_SUMMARY.read_text()) if ER_SUMMARY.exists() else None
er_all = er_summ["strata"]["all"] if er_summ else None


def perc(v):
    return 100 * v


# ---------------------------------------------------------------- S1
print("\n===== S1: per-seed accuracy (%) =====")
s1 = {}
for ds, label, seeds in (("cpc", "CPC (n = 87)", range(1, 6)),
                         ("mcr", "MCR (n = 406)", range(1, 6))):
    ids = sorted(runs[(ds, "Ax1", 1)])
    print(f"\n--- {label} ---")
    for m in ("Ax1", "P", "MDT"):
        for k in (1, 3, 5):
            per = []
            for s in seeds:
                vals = [topk(hits[(ds, m, s, c)], k) for c in ids]
                vals = [v for v in vals if v is not None]
                per.append(100 * sum(vals) / len(vals))
            s1[(ds, m, k)] = per
            row = "  ".join(f"{v:.1f}" for v in per)
            print(f"{label} | {m} | top-{k} | {row} | "
                  f"{st.mean(per):.1f} ± {st.stdev(per):.1f}")

print(f"\n--- ER-Reason (n = 364) ---")
for m in ("Ax1", "P", "MDT"):
    for k in (1, 3, 5):
        if er_raw:
            ids = sorted(runs[("er", "Ax1", 1)])
            per = []
            for s in range(1, 6):
                vals = [topk(hits[("er", m, s, c)], k) for c in ids]
                vals = [v for v in vals if v is not None]
                per.append(100 * sum(vals) / len(vals))
        else:
            per = [perc(v) for v in er_all["per_seed"][m][str(k)]]
        s1[("er", m, k)] = per
        row = "  ".join(f"{v:.1f}" for v in per)
        print(f"ER-Reason (n = 364) | {m} | top-{k} | {row} | "
              f"{st.mean(per):.1f} ± {st.stdev(per):.1f}")

# ------------------------------------------- legacy ER exact McNemar (superseded)
if er_raw:
    er_ids = sorted(runs[("er", "Ax1", 1)])
    print("\n===== ER-Reason seed-1 exact McNemar "
          "(legacy, superseded by the case-level analysis) =====")
    for a, b in (("MDT", "Ax1"), ("MDT", "P"), ("P", "Ax1")):
        for k in (1, 3, 5):
            ao = bo = 0
            for c in er_ids:
                ha = topk(hits[("er", a, 1, c)], k)
                hb = topk(hits[("er", b, 1, c)], k)
                if ha and not hb:
                    ao += 1
                elif hb and not ha:
                    bo += 1
            print(f"ER | {a} vs {b} | top-{k} | {ao}:{bo} | "
                  f"p={exact_mcnemar(ao, bo):.6g}")

# ---------------------------------------------------------------- S4
stats = json.loads((B / "stats_caselevel.json").read_text())
COMP_LABEL = {"MDT_vs_Ax1": "MDT vs A×1", "MDT_vs_P": "MDT vs P",
              "P_vs_Ax1": "P vs A×1"}
tests = []  # (dataset, comparison, topk, p)
for ds_name, label in (("CPC87", "CPC"), ("MCR406", "MCR")):
    for k in ("1", "3", "5"):
        for comp, clab in COMP_LABEL.items():
            p = stats[ds_name]["topk"][k]["comparisons"][comp]["wilcoxon_p"]
            tests.append((label, clab, int(k), p))
for k in ("1", "3", "5"):
    for comp, clab in COMP_LABEL.items():
        p = er_all["topk"][k]["comparisons"][comp]["wilcoxon_p"]
        tests.append(("ER-Reason", clab, int(k), p))
assert len(tests) == 27, len(tests)

m = len(tests)
order = sorted(range(m), key=lambda i: tests[i][3])
q = [None] * m
prev = 1.0
for rank, i in reversed(list(enumerate(order, start=1))):
    val = min(tests[i][3] * m / rank, prev, 1.0)
    q[i] = val
    prev = val

print("\n===== S4: BH correction (27 tests, sorted by p) =====")
sig = 0
for rank, i in enumerate(order, start=1):
    ds, comp, k, p = tests[i]
    if q[i] < 0.05:
        sig += 1
    print(f"{rank:2d} | {ds} | {comp} | top-{k} | p={p:.6g} | q={q[i]:.4g}")
print(f"\nsignificant at q<0.05: {sig}/{27}; max q among significant: "
      f"{max(q[i] for i in range(27) if q[i] < 0.05):.4g}")

# ------------------------------------------------- consistency checks
# CPC/MCR per-seed means must equal the case-level means that feed Table 1;
# ER-Reason must equal seed_summary.json (only checkable from the raw outputs).
for ds, ds_name in (("cpc", "CPC87"), ("mcr", "MCR406")):
    for m_, comp in (("Ax1", "Ax1"), ("P", "P"), ("MDT", "MDT")):
        for k in (1, 3, 5):
            ref = stats[ds_name]["topk"][str(k)]["case_rate_mean"][comp]
            got = st.mean(s1[(ds, m_, k)]) / 100
            assert abs(got - ref) < 1e-9, (ds, m_, k, got, ref)
if er_raw:
    for m_ in ("Ax1", "P", "MDT"):
        for k in (1, 3, 5):
            ref = er_all["mean_sd"][m_][str(k)]["mean"]
            got = st.mean(s1[("er", m_, k)]) / 100
            assert abs(got - ref) < 1e-9, (m_, k, got, ref)
print("\n[check] S1 means match stats_caselevel.json (CPC/MCR)"
      + (" and seed_summary.json (ER-Reason)" if er_raw else ""))
