# 实际发出的提示词（逐字节）——CPC P 臂 / A×1 臂

生成时间：2026-09-19　|　生成方式：由 `routing_study/scripts/topn_cpc_promptv2.py` 与 `scripts/scheme_perspective.py` 的常量直接拼接（非手工抄写）。

## 1. 说明

本文件是补充材料用的可复现性声明。P 臂实际发送的文本由两段拼接而成，代码即：

```python
prompt = PERSPECTIVE_PROMPT.format(structured_case=case_text) + P_TOPN_SUFFIX
```

- `PERSPECTIVE_PROMPT`（`scripts/scheme_perspective.py`）唯一的占位符是 `{structured_case}`，经 `.format()` 替换为该例的结构化病例文本（`topn_cpc_promptv2_87.load_merged()` 的 `text` 字段）。

- `P_TOPN_SUFFIX`（`routing_study/scripts/topn_cpc_promptv2.py`）**是普通字符串，从未经过 `.format()`**，其 JSON 块写作字面双花括号 `{{"top5": [{{"rank": 1, ...}}]}}`。因此**实际发给模型的是双花括号**（下方 §2 为该文本的逐字节内容，其中 `<CASE_TEXT>` 处为病例文本占位符）。

- 对照：A×1 臂用的 `A_TOPN_PROMPT` 本身就要经 `.format(case_text=...)` 调用，其中的 `{{"top5": ...}}` 被折叠成单花括号，所以实际发出的是合法 JSON 示例（见 §4）。这正是同一模型族下 A×1 无空输出、P 有少量空输出的原因：qwen3.8-flash 会自行把双花括号改写成单花括号；deepseek-flash 会原样照抄。

- 两个模型族的所有臂都使用**同一份**提示词常量（本文件所述文本），未做任何按模型改写，因此 qwen 族与 deepseek 族严格可比。deepseek 臂仅在**解析层**加了兜底（主解析失败时 `{{`→`{`、`}}`→`}` 归一化后重解析），提示词与诊断内容均未改动。

- 字节指纹（sha256，UTF-8）：`PERSPECTIVE_PROMPT` = `156d210a7e7e09c5bb4d9fa3395736ebb798b942a94d93858232ebf4901a9c76`；`P_TOPN_SUFFIX` = `39fe609b8eed41f9c367964ec13c1e3569738276bf4a216d10898e5489ec2f2c`；拼接后（含 `<CASE_TEXT>` 占位符的完整发送文本）= `6ad81d84b8a2f29d0f4b6ff0075aaf7750beb6bf5e595a1e5fee79caa8b1ea0b`；`A_TOPN_PROMPT` = `49ccf4221d6f244639253ec933d23295638da75fcfec057efe9f651db6777b11`。

- 占位符说明：`<CASE_TEXT>` 在真实运行时被逐例的结构化病例文本替换；`PERSPECTIVE_PROMPT` 的占位符变量名是 `{structured_case}`。


## 2. 实际发送的 P 臂提示词（逐字节，占位符 = `<CASE_TEXT>`）

````text
You are an expert diagnostic team analyzing a complex medical case from multiple complementary perspectives. Each perspective contributes a brief analysis; you then synthesize them into a single final diagnosis.

## Case Presentation
<CASE_TEXT>

## Diagnostic Perspectives

Analyze the case from each of the following perspectives. Be concise (2-4 sentences each).

**1. Attending Physician Perspective**
What is the most coherent unifying diagnosis? Which clinical features are most discriminating?

**2. Pathophysiology Perspective**
What underlying mechanism could produce this constellation of findings? Are there hallmark laboratory, histologic, or molecular clues?

**3. Imaging and Laboratory Specialist Perspective**
How should the objective data (imaging, labs, vitals, procedures) be interpreted? What patterns or paradoxes stand out?

**4. Epidemiology and Exposure Perspective**
What role do demographics, geography, travel, diet, medications, toxins, occupational exposures, or comorbidities play? Are there hidden risk factors in the narrative?

**5. Skeptic / Challenger Perspective**
What is the strongest argument AGAINST the leading diagnosis? What alternative diagnoses could explain MORE findings with FEWER contradictions? What findings remain unexplained?

## Synthesis Task
Integrate the five perspectives above and provide:
- The single most likely diagnosis
- A ranked differential diagnosis (3-5 alternatives)
- Brief reasoning explaining how the multi-perspective analysis led to your choice

Respond in EXACTLY this format:
MOST_LIKELY_DIAGNOSIS: [final best diagnosis]
DIFFERENTIAL_DIAGNOSIS: [differential 1], [differential 2], [differential 3], [differential 4]
REASONING: [concise multi-perspective synthesis]
CONFIDENCE_SCORE: [0-100]


Guidelines:
- The diagnosis under question is not necessarily a structural/organic lesion: a diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer to determine.
- A striking organic finding (mass, lesion) may be an incidental companion finding; if two separate diagnostic lines exist, evaluate both and rank each by how well it explains the whole case.
- Combination diagnoses are allowed when they reflect the real process (e.g., "post-influenza bacterial superinfection pneumonia"). For infectious candidates, name specific pathogens as separate items when clinically distinct (e.g., mucormycosis vs aspergillosis).

## Output Requirement (final block)
After your five-perspective analysis, conclude with one JSON code block (the only JSON in your answer):
```json
{{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}
```
Exactly 5 items, ranked most to least likely. Nothing may follow the JSON block.
````


## 3. 「意图版本」对照（若 `P_TOPN_SUFFIX` 曾经过 `.format()`）

即把 `P_TOPN_SUFFIX` 中的 `{{`→`{`、`}}`→`}` 后的版本。唯一差异是那段 JSON 示例的花括号；其余文字逐字节相同：

````text
You are an expert diagnostic team analyzing a complex medical case from multiple complementary perspectives. Each perspective contributes a brief analysis; you then synthesize them into a single final diagnosis.

## Case Presentation
<CASE_TEXT>

## Diagnostic Perspectives

Analyze the case from each of the following perspectives. Be concise (2-4 sentences each).

**1. Attending Physician Perspective**
What is the most coherent unifying diagnosis? Which clinical features are most discriminating?

**2. Pathophysiology Perspective**
What underlying mechanism could produce this constellation of findings? Are there hallmark laboratory, histologic, or molecular clues?

**3. Imaging and Laboratory Specialist Perspective**
How should the objective data (imaging, labs, vitals, procedures) be interpreted? What patterns or paradoxes stand out?

**4. Epidemiology and Exposure Perspective**
What role do demographics, geography, travel, diet, medications, toxins, occupational exposures, or comorbidities play? Are there hidden risk factors in the narrative?

**5. Skeptic / Challenger Perspective**
What is the strongest argument AGAINST the leading diagnosis? What alternative diagnoses could explain MORE findings with FEWER contradictions? What findings remain unexplained?

## Synthesis Task
Integrate the five perspectives above and provide:
- The single most likely diagnosis
- A ranked differential diagnosis (3-5 alternatives)
- Brief reasoning explaining how the multi-perspective analysis led to your choice

Respond in EXACTLY this format:
MOST_LIKELY_DIAGNOSIS: [final best diagnosis]
DIFFERENTIAL_DIAGNOSIS: [differential 1], [differential 2], [differential 3], [differential 4]
REASONING: [concise multi-perspective synthesis]
CONFIDENCE_SCORE: [0-100]


Guidelines:
- The diagnosis under question is not necessarily a structural/organic lesion: a diagnosis already stated in the history (including a psychiatric diagnosis), or the main reason for the current admission, may itself be the answer to determine.
- A striking organic finding (mass, lesion) may be an incidental companion finding; if two separate diagnostic lines exist, evaluate both and rank each by how well it explains the whole case.
- Combination diagnoses are allowed when they reflect the real process (e.g., "post-influenza bacterial superinfection pneumonia"). For infectious candidates, name specific pathogens as separate items when clinically distinct (e.g., mucormycosis vs aspergillosis).

## Output Requirement (final block)
After your five-perspective analysis, conclude with one JSON code block (the only JSON in your answer):
```json
{"top5": [{"rank": 1, "diagnosis": "..."}, {"rank": 2, "diagnosis": "..."}, {"rank": 3, "diagnosis": "..."}, {"rank": 4, "diagnosis": "..."}, {"rank": 5, "diagnosis": "..."}]}
```
Exactly 5 items, ranked most to least likely. Nothing may follow the JSON block.
````


差异（逐行）如下，`-` 为实际发出、`+` 为意图版本：

```diff

- {{"top5": [{{"rank": 1, "diagnosis": "..."}}, {{"rank": 2, "diagnosis": "..."}}, {{"rank": 3, "diagnosis": "..."}}, {{"rank": 4, "diagnosis": "..."}}, {{"rank": 5, "diagnosis": "..."}}]}}
+ {"top5": [{"rank": 1, "diagnosis": "..."}, {"rank": 2, "diagnosis": "..."}, {"rank": 3, "diagnosis": "..."}, {"rank": 4, "diagnosis": "..."}, {"rank": 5, "diagnosis": "..."}]}
```


## 4. 对照：实际发送的 A×1 臂提示词尾部（经 `.format()`，单花括号）

```text
Respond with ONLY a JSON object, no other text:
{"top5": [{"rank": 1, "diagnosis": "..."}, {"rank": 2, "diagnosis": "..."}, {"rank": 3, "diagnosis": "..."}, {"rank": 4, "diagnosis": "..."}, {"rank": 5, "diagnosis": "..."}]}

Case:
<CASE_TEXT>

```
