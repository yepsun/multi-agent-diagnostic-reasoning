"""Streaming + cancellation tests for the llamacpp provider path."""
import json
import threading

import pytest

from webapp import config as _config  # noqa: F401  (puts scripts/ on sys.path)
import run_inference as ri


class FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _sse(*deltas, usage=None):
    lines = []
    for d in deltas:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]}))
    if usage:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return lines


def _sse_bytes(*deltas):
    """Realistic SSE lines as the wire delivers them: UTF-8 bytes."""
    lines = []
    for d in deltas:
        payload = json.dumps({"choices": [{"delta": {"content": d}}]},
                             ensure_ascii=False)
        lines.append(("data: " + payload).encode("utf-8"))
    lines.append(b"data: [DONE]")
    return lines


class TestLlamacppStreaming:
    def _patch_post(self, monkeypatch, resp):
        calls = {}

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            calls.update(url=url, payload=json, stream=stream)
            return resp

        monkeypatch.setattr(ri.requests, "post", fake_post)
        return calls

    def test_stream_assembles_text_and_usage(self, monkeypatch):
        resp = FakeStreamResp(_sse("你好", '{"a":1}', usage={"total_tokens": 42}))
        calls = self._patch_post(monkeypatch, resp)
        ev = threading.Event()  # unset: streaming enabled, nothing cancelled
        text, usage = ri._call_llamacpp("p", 0.5, 100, 5, stop_event=ev)
        assert text == '你好{"a":1}'
        assert usage["total_tokens"] == 42
        assert calls["payload"]["stream"] is True
        assert calls["payload"]["stream_options"] == {"include_usage": True}
        assert resp.closed  # closed in finally

    def test_nonstream_path_unchanged_without_stop_event(self, monkeypatch):
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "整段返回"}}],
                        "usage": {"total_tokens": 7}}

        calls = self._patch_post(monkeypatch, FakeResp())
        text, usage = ri._call_llamacpp("p", 0.5, 100, 5)
        assert text == "整段返回"
        assert usage["total_tokens"] == 7
        assert "stream" not in calls["payload"]

    def test_stream_decodes_utf8_bytes_lines(self, monkeypatch):
        # iter_lines yields raw bytes; Chinese must survive as proper UTF-8
        resp = FakeStreamResp(_sse_bytes("多中心型Castleman病", "，你好"))
        self._patch_post(monkeypatch, resp)
        ev = threading.Event()
        text, _ = ri._call_llamacpp("p", 0.5, 100, 5, stop_event=ev)
        assert text == "多中心型Castleman病，你好"

    def test_stop_event_set_before_request_never_posts(self, monkeypatch):
        ev = threading.Event()
        ev.set()

        def unexpected(*args, **kwargs):
            raise AssertionError("must not POST when already cancelled")

        monkeypatch.setattr(ri.requests, "post", unexpected)
        with pytest.raises(ri.LLMCancelledError):
            ri._call_llamacpp("p", 0.5, 100, 5, stop_event=ev)

    def test_stop_event_midstream_aborts_and_closes(self, monkeypatch):
        ev = threading.Event()
        lines = _sse("部分", "输出")

        def gen():
            yield lines[0]
            ev.set()          # stop pressed between chunks
            yield lines[1]    # would still arrive over the wire

        class MidStreamResp(FakeStreamResp):
            def iter_lines(self, decode_unicode=True):
                return gen()

        resp = MidStreamResp(lines)
        self._patch_post(monkeypatch, resp)
        with pytest.raises(ri.LLMCancelledError):
            ri._call_llamacpp("p", 0.5, 100, 5, stop_event=ev)
        assert resp.closed

    def test_call_llm_propagates_cancellation(self, monkeypatch):
        def fake_llamacpp(*args, **kwargs):
            raise ri.LLMCancelledError("推理已被停止")

        monkeypatch.setattr(ri, "_call_llamacpp", fake_llamacpp)
        with pytest.raises(ri.LLMCancelledError):
            ri.call_llm("p", provider="llamacpp", max_retries=3)


class TestQwenStreaming:
    def _patch_post(self, monkeypatch, resp):
        calls = {}

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            calls.update(url=url, payload=json, stream=stream, timeout=timeout)
            return resp

        monkeypatch.setattr(ri.requests, "post", fake_post)
        return calls

    def test_stream_payload_and_assembly(self, monkeypatch):
        resp = FakeStreamResp(_sse("你好", "，world"))
        calls = self._patch_post(monkeypatch, resp)
        ev = threading.Event()
        text, _ = ri._call_qwen("p", 0.3, 100, 30, disable_thinking=True,
                                stop_event=ev)
        assert text == "你好，world"
        assert calls["url"].startswith(ri.QWEN_API_BASE)
        assert calls["payload"]["stream"] is True
        assert calls["payload"]["stream_options"] == {"include_usage": True}
        assert calls["payload"]["enable_thinking"] is False
        assert calls["timeout"] == (10, 30)  # fast connect fail, long read

    def test_stop_before_request_never_posts(self, monkeypatch):
        ev = threading.Event()
        ev.set()

        def unexpected(*args, **kwargs):
            raise AssertionError("must not POST when already cancelled")

        monkeypatch.setattr(ri.requests, "post", unexpected)
        with pytest.raises(ri.LLMCancelledError):
            ri._call_qwen("p", 0.3, 100, 30, stop_event=ev)

    def test_stop_midstream_aborts_and_closes(self, monkeypatch):
        ev = threading.Event()
        lines = _sse("部分", "输出")

        def gen():
            yield lines[0]
            ev.set()
            yield lines[1]

        class MidStreamResp(FakeStreamResp):
            def iter_lines(self, decode_unicode=True):
                return gen()

        resp = MidStreamResp(lines)
        self._patch_post(monkeypatch, resp)
        with pytest.raises(ri.LLMCancelledError):
            ri._call_qwen("p", 0.3, 100, 30, stop_event=ev)
        assert resp.closed

    def test_no_stop_event_keeps_nonstream(self, monkeypatch):
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "x"}}], "usage": {}}

        calls = self._patch_post(monkeypatch, FakeResp())
        text, _ = ri._call_qwen("p", 0.3, 100, 30)
        assert text == "x"
        assert "stream" not in calls["payload"]
