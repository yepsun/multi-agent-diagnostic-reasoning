import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from webapp import app as app_module
from webapp.app import app, _JOBS
from webapp.prompts import EXPERT_ROLES

A_MARKER = "鉴别诊断分析"              # appears only in the A_WEB prompt
MODERATOR_MARKER = "moderator of an MDT panel"  # appears only in the moderator prompt

GOOD_A = ('{"primary_diagnosis": "iMCD", "primary_diagnosis_en": '
          '"Multicentric Castleman disease", "confidence": 85,'
          '"differential_diagnoses": ['
          '{"diagnosis": "POEMS综合征", "diagnosis_en": "POEMS syndrome",'
          '"supporting": "骨破坏", "refuting": "M蛋白阴性", "next_test": "轻链"}],'
          '"reasoning_summary": "x", "next_steps": ["活检"]}')

GOOD_MOD = ('```json\n{"primary_diagnosis": "多中心型Castleman病", '
            '"primary_diagnosis_en": "Multicentric Castleman disease", '
            '"confidence": 80, "key_findings": ["发热8个月"], '
            '"key_negatives": ["M蛋白阴性"], '
            '"differential_diagnoses": ['
            '{"diagnosis": "POEMS综合征", "diagnosis_en": "POEMS syndrome",'
            '"supporting": "s", "refuting": "r", "next_test": "t"},'
            '{"diagnosis": "结核", "diagnosis_en": "Tuberculosis",'
            '"supporting": "s", "refuting": "r", "next_test": "t"},'
            '{"diagnosis": "淋巴瘤", "diagnosis_en": "Lymphoma",'
            '"supporting": "s", "refuting": "r", "next_test": "t"},'
            '{"diagnosis": "结节病", "diagnosis_en": "Sarcoidosis",'
            '"supporting": "s", "refuting": "r", "next_test": "t"}], '
            '"reasoning_summary": "y", "next_steps": ["淋巴结活检"]}\n```')

_EXPERT_DX = ["iMCD", "POEMS综合征", "结核", "淋巴瘤", "结节病"]
GOOD_EXPERT = json.dumps(
    {"top5": [{"rank": i + 1, "diagnosis": dx, "rationale": "r%d" % i}
              for i, dx in enumerate(_EXPERT_DX)]}, ensure_ascii=False)

CASE = "患者男性，55岁，反复咳嗽1年，发热8个月，淋巴结肿大，骨破坏待查。" * 2


def _kind(prompt: str) -> str:
    if '"top5"' in prompt:
        return "expert"
    if MODERATOR_MARKER in prompt:
        return "moderator"
    return "a"


def _fake_call_llm(prompt, **kwargs):
    kind = _kind(prompt)
    if kind == "expert":
        return GOOD_EXPERT, {}
    if kind == "moderator":
        return GOOD_MOD, {}
    return GOOD_A, {}


@pytest.fixture()
def client(monkeypatch):
    _JOBS.clear()
    # no real LLM, no network
    monkeypatch.setattr(app_module.ri, "call_llm", _fake_call_llm)
    monkeypatch.setattr(app_module.ri, "call_llm_judge",
                        lambda prompt, **kw: ("YES", {}))
    monkeypatch.setattr(app_module, "_llama_healthy", lambda: False)
    with TestClient(app) as c:
        yield c


def _run(client, modes=None, **data):
    payload = {"text": CASE}
    if modes is not None:
        payload["modes"] = modes
    payload.update(data)
    return client.post("/api/jobs", data=payload)


def _wait_done(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error", "stopped"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


class TestJobModes:
    def test_a_only_job(self, client, monkeypatch):
        calls = []

        def rec(prompt, **kw):
            calls.append(_kind(prompt))
            return _fake_call_llm(prompt, **kw)

        monkeypatch.setattr(app_module.ri, "call_llm", rec)
        job_id = _run(client, modes="a").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done", body["error"]
        assert body["modes"] == ["a"]
        assert len(body["a_samples"]) == 1
        assert body["p"] is None
        assert body["p_experts_done"] == 0
        assert body["agreement"] is None
        # only the A call ran: no expert, no moderator
        assert calls == ["a"]

    def test_p_only_job(self, client, monkeypatch):
        calls = []

        def rec(prompt, **kw):
            calls.append(_kind(prompt))
            return _fake_call_llm(prompt, **kw)

        monkeypatch.setattr(app_module.ri, "call_llm", rec)
        job_id = _run(client, modes="p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done", body["error"]
        assert body["modes"] == ["p"]
        assert body["a_samples"] == []
        assert body["p"] is not None
        assert body["p"]["primary_diagnosis"] == "多中心型Castleman病"
        assert body["p"]["key_findings"] == ["发热8个月"]
        assert body["p"]["next_steps"] == ["淋巴结活检"]
        assert body["p_experts_done"] == 5
        # aggregate_a over zero samples yields a null consensus, not null itself
        assert body["consensus"]["consensus"] is None
        assert body["agreement"] is None
        assert calls.count("expert") == 5
        assert calls.count("moderator") == 1
        assert calls.count("a") == 0

    def test_both_modes_job(self, client):
        job_id = _run(client, modes="a,p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done", body["error"]
        assert body["modes"] == ["a", "p"]
        assert len(body["a_samples"]) == 1
        assert body["p"] is not None
        assert body["p_experts_done"] == 5
        assert body["consensus"]["consensus"]["votes"] == 1
        assert body["consensus"]["consensus"]["diagnosis"] == "iMCD"
        assert body["consensus"]["top5"][1]["diagnosis"] == "POEMS综合征"
        # iMCD vs 多中心型Castleman病 → same disease → agree (fake judge YES)
        assert body["agreement"] == "agree"
        assert body["agreement_method"] == "llm"

    def test_default_modes_is_both(self, client):
        job_id = _run(client).json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["modes"] == ["a", "p"]
        assert body["p"] is not None and len(body["a_samples"]) == 1

    def test_modes_canonicalized_deduped(self, client):
        job_id = _run(client, modes=" p , a , p ").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["modes"] == ["a", "p"]

    def test_modes_validation(self, client):
        # httpx drops truely empty form values, so the "" path is unit-tested
        # below; whitespace / unknown-token selections go through the API.
        assert app_module._parse_modes("") is None
        for raw in ("   ", "x", "a,x", "p,q", ","):
            r = _run(client, modes=raw)
            assert r.status_code == 400, raw
            assert r.json()["detail"] == "请至少选择一种模式（A / P）"

    def test_expert_prompts_issued_for_every_role(self, client, monkeypatch):
        seen = []
        lock = threading.Lock()

        def rec(prompt, **kw):
            kind = _kind(prompt)
            if kind == "expert":
                with lock:
                    seen.append(prompt)
            return _fake_call_llm(prompt, **kw)

        monkeypatch.setattr(app_module.ri, "call_llm", rec)
        job_id = _run(client, modes="p").json()["job_id"]
        _wait_done(client, job_id)
        assert len(seen) == 5
        for title, _ in EXPERT_ROLES:
            assert any(title in p for p in seen), title

    def test_progressive_snapshots(self, client, monkeypatch):
        def slow(prompt, **kw):
            time.sleep(0.4)
            return _fake_call_llm(prompt, **kw)

        monkeypatch.setattr(app_module.ri, "call_llm", slow)
        job_id = _run(client, modes="a,p").json()["job_id"]
        time.sleep(0.55)  # A done (0.4s), experts still running
        body = client.get(f"/api/jobs/{job_id}").json()
        assert body["status"] == "running"
        assert len(body["a_samples"]) == 1
        assert body["p"] is None
        _wait_done(client, job_id)

    def test_validation_errors(self, client):
        assert client.post("/api/jobs", data={"text": "太短"}).status_code == 400
        assert client.post("/api/jobs", data={}).status_code == 400
        assert client.get("/api/jobs/nope").status_code == 404

    def test_docx_upload(self, client):
        from webapp.tests.test_docx_extract import _make_docx
        data = _make_docx([CASE[:60], CASE[60:]])
        r = client.post("/api/jobs",
                        files={"file": ("case.docx", data,
                                        "application/vnd.openxmlformats-"
                                        "officedocument.wordprocessingml.document")})
        assert r.status_code == 200
        body = _wait_done(client, r.json()["job_id"])
        assert body["status"] == "done"
        assert body["source_name"] == "case.docx"

    def test_llm_failure_marks_job_error(self, client, monkeypatch):
        def boom(prompt, **kwargs):
            raise RuntimeError("llama server down")
        monkeypatch.setattr(app_module.ri, "call_llm", boom)
        job_id = _run(client, modes="p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "error"
        assert "llama server down" in body["error"]


class TestStop:
    def test_stop_aborts_experts_and_keeps_finished_a(self, client, monkeypatch):
        state = {"experts": 0, "moderator": 0}
        lock = threading.Lock()

        def cancellable(prompt, stop_event=None, **kwargs):
            kind = _kind(prompt)
            if kind == "a":
                return GOOD_A, {}
            if kind == "expert":
                with lock:
                    state["experts"] += 1
                if stop_event is not None and stop_event.wait(timeout=5):
                    raise app_module.ri.LLMCancelledError("推理已被停止")
                return GOOD_EXPERT, {}
            with lock:
                state["moderator"] += 1
            return GOOD_MOD, {}

        monkeypatch.setattr(app_module.ri, "call_llm", cancellable)
        job_id = _run(client, modes="a,p").json()["job_id"]

        # wait until A is done and all five experts are in flight
        deadline = time.time() + 5
        while time.time() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if len(body["a_samples"]) == 1 and state["experts"] >= 5:
                break
            time.sleep(0.02)
        assert len(body["a_samples"]) == 1 and state["experts"] == 5

        assert client.post(f"/api/jobs/{job_id}/stop").status_code == 200

        deadline = time.time() + 5
        while time.time() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] == "stopped":
                break
            time.sleep(0.02)
        assert body["status"] == "stopped"
        assert body["stop_requested"] is True
        assert len(body["a_samples"]) == 1     # finished A result kept
        assert body["p"] is None               # experts aborted, discarded
        assert state["moderator"] == 0         # moderator never started
        time.sleep(0.2)
        assert state["experts"] == 5           # no further expert calls started

    def test_stop_unknown_job_404(self, client):
        assert client.post("/api/jobs/nope/stop").status_code == 404

    def test_stop_on_finished_job_is_noop(self, client):
        job_id = _run(client, modes="a").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done"
        r = client.post(f"/api/jobs/{job_id}/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "done"
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"


class TestAgreementJudge:
    def test_judge_receives_both_primaries_and_verdicts(self, client, monkeypatch):
        seen = {}

        def fake_judge(prompt, **kw):
            seen["prompt"] = prompt
            return "YES", {}

        monkeypatch.setattr(app_module.ri, "call_llm_judge", fake_judge)
        job_id = _run(client, modes="a,p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["agreement"] == "agree"
        assert body["agreement_method"] == "llm"
        assert "诊断A" in seen["prompt"] and "诊断B" in seen["prompt"]
        assert "iMCD" in seen["prompt"] and "多中心型Castleman病" in seen["prompt"]

    def test_judge_no_forces_disagree_despite_matching_strings(
            self, client, monkeypatch):
        monkeypatch.setattr(app_module.ri, "call_llm_judge",
                            lambda prompt, **kw: ("NO", {}))
        job_id = _run(client, modes="a,p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["agreement"] == "disagree"
        assert body["agreement_method"] == "llm"

    def test_single_mode_never_calls_judge(self, client, monkeypatch):
        called = []
        monkeypatch.setattr(app_module.ri, "call_llm_judge",
                            lambda prompt, **kw: called.append(prompt) or ("YES", {}))
        body = _wait_done(client, _run(client, modes="a").json()["job_id"])
        assert body["agreement"] is None
        body = _wait_done(client, _run(client, modes="p").json()["job_id"])
        assert body["agreement"] is None
        assert called == []

    def test_judge_called_once_per_job(self, client, monkeypatch):
        calls = []

        def fake_judge(prompt, **kw):
            calls.append(prompt)
            return "YES", {}

        monkeypatch.setattr(app_module.ri, "call_llm_judge", fake_judge)
        job_id = _run(client, modes="a,p").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done"
        assert len(calls) == 1  # consensus label never changed → single judge call

    def test_judge_failure_falls_back_to_string_heuristic(
            self, client, monkeypatch):
        def boom(prompt, **kw):
            raise RuntimeError("judge down")

        monkeypatch.setattr(app_module.ri, "call_llm_judge", boom)
        job_id = _run(client, modes="a,p").json()["job_id"]
        body = _wait_done(client, job_id)
        # iMCD vs 多中心型Castleman病: string heuristic says same disease
        assert body["agreement"] == "agree"
        assert body["agreement_method"] == "string"


class TestProvider:
    def test_forced_modes(self, monkeypatch):
        monkeypatch.setattr(app_module.config, "PROVIDER_MODE", "qwen")
        assert app_module._pick_provider() == "qwen"
        monkeypatch.setattr(app_module.config, "PROVIDER_MODE", "llamacpp")
        assert app_module._pick_provider() == "llamacpp"

    def test_auto_follows_llama_health(self, monkeypatch):
        monkeypatch.setattr(app_module.config, "PROVIDER_MODE", "auto")
        monkeypatch.setattr(app_module, "_llama_healthy", lambda: True)
        assert app_module._pick_provider() == "llamacpp"
        monkeypatch.setattr(app_module, "_llama_healthy", lambda: False)
        assert app_module._pick_provider() == "qwen"

    def test_health_reports_effective_qwen_when_llama_down(self, client):
        body = client.get("/api/health").json()
        assert body["llm"]["reachable"] is False
        assert body["llm"]["effective"] == "qwen"
        assert body["llm"]["model"] == app_module.ri.QWEN_MODEL

    def test_job_uses_cloud_provider_with_thinking_disabled(self, client, monkeypatch):
        seen = []

        def rec(prompt, provider=None, disable_thinking=None, **kw):
            seen.append((provider, disable_thinking))
            return _fake_call_llm(prompt, **kw)

        monkeypatch.setattr(app_module.ri, "call_llm", rec)
        job_id = _run(client, modes="a").json()["job_id"]
        body = _wait_done(client, job_id)
        assert body["status"] == "done"
        assert seen == [("qwen", True)]


class TestLiterature:
    def test_returns_articles(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "search_pubmed",
                            lambda q, retmax=None: [
                                {"pmid": "111", "title": "T", "journal": "J",
                                 "pubdate": "2020", "authors": "A",
                                 "url": "u", "abstract": "abs"}])
        r = client.get("/api/literature", params={"query": "castleman disease"})
        assert r.status_code == 200
        body = r.json()
        assert body["articles"][0]["pmid"] == "111"

    def test_short_query_400(self, client):
        assert client.get("/api/literature", params={"query": "a"}).status_code == 400

    def test_upstream_failure_502(self, client, monkeypatch):
        def boom(q, retmax=None):
            raise RuntimeError("timeout")
        monkeypatch.setattr(app_module, "search_pubmed", boom)
        r = client.get("/api/literature", params={"query": "anemia"})
        assert r.status_code == 502


class TestHealth:
    def test_unreachable_llm_reports_false(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("connection refused")
        monkeypatch.setattr(app_module._requests, "get", boom)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["llm"]["reachable"] is False

    def test_reachable_llm_reports_pretty_model(self, client, monkeypatch):
        class FakeResp:
            ok = True
            def json(self):
                return {"data": [{"id": "D:\\\\llm\\\\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"}]}
        monkeypatch.setattr(app_module._requests, "get",
                            lambda *args, **kwargs: FakeResp())
        body = client.get("/api/health").json()
        assert body["llm"]["reachable"] is True
        assert body["llm"]["model"] == "Qwen3.8-Flash-Next-UD-IQ4_XS"

    def test_index_page_served(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestAudit:
    def test_job_and_query_audited_without_case_text(self, client, monkeypatch, tmp_path):
        log = tmp_path / "audit.jsonl"
        monkeypatch.setattr(app_module.config, "AUDIT_LOG", log)
        monkeypatch.setattr(app_module, "search_pubmed", lambda q, retmax=None: [])
        job_id = _run(client, modes="a,p").json()["job_id"]
        _wait_done(client, job_id)
        client.get("/api/literature", params={"query": "castleman"})
        lines = [json.loads(l) for l in log.read_text().splitlines()]
        events = [l["event"] for l in lines]
        assert "job_created" in events and "literature_query" in events
        assert all(CASE[:20] not in json.dumps(l) for l in lines)
