from webapp.prompts import (
    A_WEB_PROMPT,
    EXPERT_GUIDELINES,
    EXPERT_ROLES,
    build_a_prompt,
    build_expert_prompt,
    build_moderator_prompt,
    extract_json,
    parse_assessment,
    parse_expert_top5,
)


GOOD = """{
  "primary_diagnosis": "多中心型Castleman病（iMCD）",
  "primary_diagnosis_en": "Multicentric Castleman disease",
  "confidence": 85,
  "key_findings": ["发热8个月", "胸锁关节骨破坏"],
  "key_negatives": ["M蛋白阴性"],
  "differential_diagnoses": [
    {"diagnosis": "POEMS综合征", "diagnosis_en": "POEMS syndrome",
     "supporting": "骨破坏、色素沉着", "refuting": "M蛋白阴性", "next_test": "血游离轻链"},
    {"diagnosis": "结核", "diagnosis_en": "Tuberculosis",
     "supporting": "T-SPOT阳性", "refuting": "无结核中毒症状", "next_test": "病灶活检mNGS"}
  ],
  "reasoning_summary": "以淋巴结肿大+炎症指标升高为主线。",
  "next_steps": ["淋巴结活检", "骨髓穿刺"]
}"""

EXPERT = ('{"top5": ['
          '{"rank": 1, "diagnosis": "iMCD", "rationale": "multi-system inflammation"},'
          '{"rank": 2, "diagnosis": "POEMS综合征", "rationale": "bone lesions"}]}')


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json(GOOD)["confidence"] == 85

    def test_fenced_block_after_prose(self):
        text = "分析过程……综合判断如下：\n```json\n" + GOOD + "\n```\n以上。"
        assert extract_json(text)["primary_diagnosis"].startswith("多中心")

    def test_last_fenced_block_wins(self):
        text = "```json\n{\"a\": 1}\n```\n更正：\n```json\n" + GOOD + "\n```"
        assert "primary_diagnosis" in extract_json(text)

    def test_prose_then_bare_object(self):
        text = "结论：\n" + GOOD + "\n（完）"
        assert extract_json(text)["confidence"] == 85

    def test_garbage_returns_none(self):
        assert extract_json("没有JSON") is None
        assert extract_json("") is None
        assert extract_json("{broken") is None


class TestParseAssessment:
    def test_full_parse(self):
        a = parse_assessment(GOOD)
        assert a["primary_diagnosis"] == "多中心型Castleman病（iMCD）"
        assert a["confidence"] == 85
        assert len(a["differential_diagnoses"]) == 2
        assert a["differential_diagnoses"][0]["next_test"] == "血游离轻链"
        assert a["key_negatives"] == ["M蛋白阴性"]
        assert a["raw"] == GOOD.strip()

    def test_confidence_clamped_and_typed(self):
        a = parse_assessment('{"primary_diagnosis": "X", "confidence": 120}')
        assert a["confidence"] == 100
        b = parse_assessment('{"primary_diagnosis": "X", "confidence": "abc"}')
        assert b["confidence"] is None
        c = parse_assessment('{"primary_diagnosis": "X", "confidence": 3.6}')
        assert c["confidence"] == 4

    def test_missing_fields_degrade(self):
        a = parse_assessment('{"primary_diagnosis": "流感"}')
        assert a["differential_diagnoses"] == []
        assert a["confidence"] is None

    def test_empty_response(self):
        a = parse_assessment("模型没按要求输出")
        assert a["primary_diagnosis"] == ""
        assert a["raw"] == "模型没按要求输出"


class TestPromptBuilders:
    def test_a_prompt_contains_case_and_schema(self):
        p = build_a_prompt("患者男性，55岁，发热8个月。")
        assert "患者男性，55岁" in p
        assert "primary_diagnosis_en" in p
        assert "differential_diagnoses" in p
        assert "分析要求" in p
        assert A_WEB_PROMPT.startswith("你是一名经验丰富的内科会诊专家")

    def test_expert_prompt_contains_role_case_and_top5(self):
        title, brief = EXPERT_ROLES[0]
        case = "A 55-year-old man with fever and lymphadenopathy."
        p = build_expert_prompt(title, brief, case)
        assert title in p                       # role identity
        assert brief in p                       # role lens
        assert case in p                        # case appended
        assert '"top5"' in p                    # top5 schema requested
        assert EXPERT_GUIDELINES in p
        assert "{case_text}" not in p

    def test_expert_roles_shape(self):
        assert len(EXPERT_ROLES) == 5
        titles = [t for t, _ in EXPERT_ROLES]
        assert "Attending Internist" in titles
        assert "Skeptic / Challenger" in titles

    def test_moderator_prompt_contains_case_opinions_and_schema(self):
        case = "55岁男性，发热待查。"
        opinion = "Attending Internist:\n  1. iMCD — multi-system inflammation"
        p = build_moderator_prompt(case, opinion)
        assert case in p
        assert opinion in p
        assert "primary_diagnosis_en" in p
        assert "differential_diagnoses" in p
        assert p.rstrip().endswith("只输出一个JSON代码块，其后不再输出任何文字。")
        # no unsubstituted placeholder from the old P prompt
        assert "{structured_case}" not in p


class TestParseExpertTop5:
    def test_basic_parse(self):
        out = parse_expert_top5(EXPERT)
        assert out == [
            {"diagnosis": "iMCD", "rationale": "multi-system inflammation"},
            {"diagnosis": "POEMS综合征", "rationale": "bone lesions"},
        ]

    def test_fenced_response(self):
        out = parse_expert_top5("分析如下：\n```json\n" + EXPERT + "\n```\n以上。")
        assert out[0]["diagnosis"] == "iMCD"

    def test_dedup_by_diagnosis_case_insensitive(self):
        raw = ('{"top5": ['
               '{"diagnosis": "iMCD", "rationale": "a"},'
               '{"diagnosis": "IMCD", "rationale": "b"},'
               '{"diagnosis": "POEMS", "rationale": "c"}]}')
        out = parse_expert_top5(raw)
        assert [x["diagnosis"] for x in out] == ["iMCD", "POEMS"]

    def test_max_five(self):
        items = ", ".join(
            '{"rank": %d, "diagnosis": "D%d", "rationale": "r"}' % (i, i)
            for i in range(1, 9))
        out = parse_expert_top5('{"top5": [%s]}' % items)
        assert len(out) == 5
        assert [x["diagnosis"] for x in out] == ["D1", "D2", "D3", "D4", "D5"]

    def test_malformed_returns_empty(self):
        assert parse_expert_top5("没有JSON") == []
        assert parse_expert_top5("") == []
        assert parse_expert_top5('{"top5": "not a list"}') == []
        assert parse_expert_top5('{"top5": [1, 2]}') == []
