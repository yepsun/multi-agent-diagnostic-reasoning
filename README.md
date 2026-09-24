# When does a team of LLMs beat a single call? — Code and adjudication caches

Companion repository for the study of multi-agent diagnostic prompting strategies
(A×1 single call / P personas-in-one-context / MDT isolated roles + moderator,
plus matched control arms and factorial cells) on CPC, MedCaseReasoning, and
ER-Reason.

## Contents
- `code/` — full pipeline: inference drivers, prompts (byte-identical to the
  study, including the disclosed prompt-transcription defect and its control),
  LLM-judge implementation (v3 equivalence rules), statistics, and figure scripts.
- `data/judge_caches/` — the frozen primary judge cache
  (`judge_cache_glm_v3.json`, GLM-5.3-flash × v3 rules) and the sensitivity-judge
  cache, containing every adjudication used for the CPC and MedCaseReasoning
  analyses. **Entries derived from ER-Reason are excluded** under the ER-Reason
  data use agreement.
- `data/runs/` — per-case outputs for the CPC and MedCaseReasoning arms
  (case identifiers one-way hashed; ER-Reason per-case outputs are not included).

## Not included
- ER-Reason source data and per-case outputs (DUA; apply via PhysioNet).
- NEJM case texts (copyright; available from the journal).

## Judge reproduction
The judge scripts read `ZHIPU_API_KEY` from the environment and call the
GLM-5.3-flash endpoint; the cache key is `gold[:150] + "||" + cand[:150]`.
See `code/routing_study/scripts/caselevel_stats.py`.

## License
Code: MIT. Data files: released for verification of the manuscript's analyses.
