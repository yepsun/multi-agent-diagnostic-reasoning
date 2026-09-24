from webapp.clustering import (
    aggregate_a,
    cluster_diagnoses,
    normalize_name,
    same_disease,
)


def sample(primary, alts, conf=80):
    return {
        "primary_diagnosis": primary,
        "primary_diagnosis_en": "",
        "confidence": conf,
        "key_findings": [], "key_negatives": [],
        "differential_diagnoses": [
            {"diagnosis": a, "diagnosis_en": "", "supporting": "s",
             "refuting": "r", "next_test": "t"} for a in alts
        ],
        "reasoning_summary": "", "next_steps": [], "raw": "",
    }


class TestNormalizeAndSame:
    def test_imcd_variants_merge(self):
        assert same_disease("iMCD", "多中心型Castleman病")
        assert same_disease("Castleman病（多中心型，HHV-8阴性）", "多中心型Castleman病（iMCD）")
        assert normalize_name("iMCD") == "castleman"

    def test_different_diseases_do_not_merge(self):
        assert not same_disease("POEMS综合征", "多中心型Castleman病")
        assert not same_disease("结核感染", "结节病")

    def test_generic_head_word_does_not_merge(self):
        assert not same_disease("贫血", "再生障碍性贫血")
        assert not same_disease("淋巴瘤", "T细胞淋巴瘤")

    def test_containment_merges(self):
        assert same_disease("IgG4相关疾病", "IgG4相关性疾病（累及泪腺）")

    def test_empty_strings(self):
        assert not same_disease("", "x")
        assert not same_disease(None, None)


class TestCluster:
    def test_rep_is_most_frequent_spelling(self):
        names = ["iMCD", "多中心型Castleman病", "iMCD", "结节病", "iMCD"]
        clusters = cluster_diagnoses(names)
        by_rep = {c["rep"]: c for c in clusters}
        assert by_rep["iMCD"]["count"] == 4  # 3×iMCD + 多中心型Castleman病
        assert by_rep["结节病"]["count"] == 1

    def test_order_preserved(self):
        clusters = cluster_diagnoses(["结节病", "iMCD", "结节病"])
        assert clusters[0]["rep"] == "结节病"
        assert clusters[1]["rep"] == "iMCD"


class TestAggregateA:
    def test_consensus_with_minority(self):
        samples = [sample("iMCD", ["POEMS综合征", "结核"]),
                   sample("多中心型Castleman病", ["POEMS综合征", "淋巴瘤"]),
                   sample("Castleman病（多中心型）", ["结节病"]),
                   sample("iMCD", ["POEMS综合征"]),
                   sample("AITL", ["POEMS综合征"])]
        agg = aggregate_a(samples)
        assert agg["consensus"]["diagnosis"] == "iMCD"
        assert agg["consensus"]["votes"] == 4
        assert agg["agreement"] == 0.8
        assert {"rep": "AITL", "count": 1} in agg["minority"]
        # top5: consensus first, POEMS ranked highest among alternates
        assert agg["top5"][0]["diagnosis"] == "iMCD"
        assert agg["top5"][1]["diagnosis"] == "POEMS综合征"
        assert agg["top5"][1]["votes"] == 4
        assert len(agg["top5"]) <= 5
        # alternates same-disease as consensus are excluded
        assert all(not same_disease(t["diagnosis"], "iMCD") for t in agg["top5"][1:])

    def test_unanimous(self):
        samples = [sample("流感", ["普通感冒"]) for _ in range(5)]
        agg = aggregate_a(samples)
        assert agg["consensus"]["votes"] == 5
        assert agg["agreement"] == 1.0
        assert agg["minority"] == []

    def test_no_valid_primaries(self):
        agg = aggregate_a([sample("", []), sample("", [])])
        assert agg["consensus"] is None
        assert agg["top5"] == []

    def test_five_samples_real_case_pattern(self):
        # The 32-bed case: 4/5 disease-level agreement despite string variety
        names = ["Castleman病（多中心型，HHV-8阴性，iMCD）", "iMCD",
                 "多中心Castleman病", "iMCD（特发性）", "AITL"]
        samples = [sample(n, ["IgG4相关疾病", "POEMS综合征"]) for n in names]
        agg = aggregate_a(samples)
        assert agg["consensus"]["votes"] == 4
        assert agg["agreement"] == 0.8
