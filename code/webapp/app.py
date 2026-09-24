"""MDT 内网辅助诊断 Web 服务（FastAPI）。

用户可为每个作业选择 A / P 两种模式（可单选或同时选）：
- A：单次贪心调用（temperature 0.0），一次结构化评估。
- P：5 个互相隔离的专家并行推理（各出 top-5），再由主持人汇总为结构化评估。
本地 llama.cpp 推理；文献检索仅向 PubMed 发送诊断关键词，病例文本不出内网。

启动（项目根目录）：
    .venv/bin/uvicorn webapp.app:app --host 0.0.0.0 --port 8000
"""
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests as _requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import config
from .clustering import aggregate_a, same_disease
from .docx_extract import extract_any
from .prompts import (EXPERT_ROLES, build_a_prompt, build_expert_prompt,
                      build_moderator_prompt, parse_assessment,
                      parse_expert_top5)
from .pubmed import search_pubmed

# env defaults must be set before run_inference is imported (config does it)
import run_inference as ri  # noqa: E402

app = FastAPI(title="MDT 辅助诊断", docs_url=None, redoc_url=None)

_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()

_LLAMA_HEALTH = {"ts": 0.0, "ok": False}
_LLAMA_HEALTH_LOCK = threading.Lock()


def _llama_healthy() -> bool:
    """Probe the llama.cpp server, cached for 60s to stay off the hot path."""
    now = time.time()
    with _LLAMA_HEALTH_LOCK:
        if now - _LLAMA_HEALTH["ts"] < 60:
            return _LLAMA_HEALTH["ok"]
    try:
        r = _requests.get(f"{ri.LLAMACPP_API_BASE}/models", timeout=3)
        ok = r.ok
    except Exception:
        ok = False
    with _LLAMA_HEALTH_LOCK:
        _LLAMA_HEALTH.update(ts=now, ok=ok)
    return ok


def _pick_provider() -> str:
    """Resolve the inference transport for a job.

    "auto" prefers the intranet llama.cpp server (privacy: case text stays
    local) and falls back to the DashScope cloud API only when it is
    unreachable. "llamacpp" / "qwen" force one transport.
    """
    mode = config.PROVIDER_MODE
    if mode in ("llamacpp", "qwen"):
        return mode
    return "llamacpp" if _llama_healthy() else "qwen"

PUBLIC_FIELDS = ("job_id", "status", "created_at", "source_name", "input_chars",
                 "modes", "p", "p_experts_done", "a_samples", "consensus",
                 "agreement", "error", "stop_requested", "agreement_method")

_MODES_ORDER = ("a", "p")


def _parse_modes(raw: Optional[str]) -> Optional[list]:
    """Parse the comma-separated mode selection, or None when invalid.

    Accepts any subset of {"a", "p"} (case-insensitive, whitespace stripped,
    duplicates ignored) and returns it in canonical order. Unknown tokens or
    an empty selection are invalid.
    """
    tokens = [t.strip().lower() for t in (raw or "").split(",") if t.strip()]
    if not tokens or any(t not in _MODES_ORDER for t in tokens):
        return None
    return [m for m in _MODES_ORDER if m in tokens]

JUDGE_PROMPT = """你是医学诊断判定助手。判断下面两个诊断结论描述的主要疾病是否相同。

判定规则（按顺序执行）：
1. 两者命名了同一种疾病 → YES（分期、分型、活动度、严重度等限定词的差异，以及一方多写或少写限定词，均不影响）。
2. 一方额外提及并发症、合并症或受累器官（如"狼疮性肾炎""继发性噬血细胞综合征"）→ 不影响判定。
3. 仅当主要疾病本身不同 → NO。

诊断A：{a}
诊断B：{b}

只回答 YES 或 NO。"""


def _judge_agreement(a: str, b: str) -> Optional[bool]:
    """Semantic verdict for the banner via the independent judge channel.

    Only the two primary-diagnosis strings leave the intranet (no case
    text). Returns None when the judge is unavailable.
    """
    try:
        raw, _ = ri.call_llm_judge(JUDGE_PROMPT.format(a=a, b=b),
                                   timeout=60, max_retries=1)
        v = (raw or "").strip().upper()
        if v.startswith("YES"):
            return True
        if v.startswith("NO"):
            return False
    except Exception:
        pass
    return None


def _audit(event: dict) -> None:
    """Append an audit record; case text is never written to the log."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    try:
        with open(config.AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _prune_jobs(keep: int = 50) -> None:
    with _JOBS_LOCK:
        excess = len(_JOBS) - keep
        if excess <= 0:
            return
        finished = sorted((j["created_at"], jid) for jid, j in _JOBS.items()
                          if j["status"] in ("done", "error", "stopped"))
        for _, jid in finished[:excess]:
            del _JOBS[jid]


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    mode = config.PROVIDER_MODE
    base = ri.LLAMACPP_API_BASE
    reachable = False
    model = ""
    try:
        r = _requests.get(f"{base}/models", timeout=4)
        reachable = r.ok
        if reachable:
            try:
                model = r.json()["data"][0]["id"]
            except Exception:
                model = ""
    except Exception:
        reachable = False
    # llama.cpp returns the full GGUF path; show a compact model name
    model = re.split(r"[\\/]+", model)[-1]
    model = re.sub(r"(-\d+-of-\d+)?\.gguf$", "", model, flags=re.IGNORECASE)

    if mode in ("qwen", "llamacpp"):
        effective = mode
    else:  # auto
        effective = "llamacpp" if reachable else "qwen"
    if effective == "qwen":
        model = ri.QWEN_MODEL
    with _LLAMA_HEALTH_LOCK:
        _LLAMA_HEALTH.update(ts=time.time(), ok=reachable)
    return {"llm": {"reachable": reachable, "model": model, "base": base,
                    "mode": mode, "effective": effective}}


@app.post("/api/jobs")
async def create_job(request: Request,
                     text: Optional[str] = Form(None),
                     file: Optional[UploadFile] = File(None),
                     modes: str = Form("a,p")):
    if file is not None and file.filename:
        data = await file.read()
        if not data:
            raise HTTPException(400, "上传文件为空")
        try:
            content = extract_any(data, file.filename)
        except Exception:
            raise HTTPException(400, "文件解析失败（仅支持 .docx 与纯文本）")
        source = file.filename
    elif text and text.strip():
        content = text.strip()
        source = "pasted"
    else:
        raise HTTPException(400, "请粘贴病例文本或上传 docx 文件")

    if len(content) < 50:
        raise HTTPException(400, "病例文本太短（至少50字符）")
    selected = _parse_modes(modes)
    if selected is None:
        raise HTTPException(400, "请至少选择一种模式（A / P）")
    content = content[:config.MAX_INPUT_CHARS]

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_name": source,
        "input_chars": len(content),
        "modes": selected,
        "p": None,
        "p_experts_done": 0,
        "a_samples": [],
        "consensus": None,
        "agreement": None,
        "agreement_method": None,
        "judge_key": None,
        "error": None,
        "stop_requested": False,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    _prune_jobs()
    threading.Thread(target=_execute_job, args=(job_id, content), daemon=True).start()
    _audit({"event": "job_created", "job_id": job_id,
            "ip": request.client.host if request.client else "",
            "chars": len(content), "source": source})
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    return {k: job.get(k) for k in PUBLIC_FIELDS}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str, request: Request):
    """Request early stop: keep finished inference, skip remaining calls.

    The LLM call in flight cannot be interrupted; it completes and is kept,
    then the loop breaks before starting any further call.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    if job["status"] in ("running", "queued"):
        job["stop_requested"] = True
        ev = job.get("stop_event")
        if ev is not None:
            ev.set()  # aborts the in-flight streamed generation immediately
        _audit({"event": "job_stop_requested", "job_id": job_id,
                "ip": request.client.host if request.client else ""})
    return {"job_id": job_id, "status": job["status"]}


@app.get("/api/literature")
def literature(request: Request, query: str):
    query = (query or "").strip()[:200]
    if len(query) < 2:
        raise HTTPException(400, "检索词太短（至少2个字符）")
    try:
        articles = search_pubmed(query)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"PubMed 检索失败：{e}")
    _audit({"event": "literature_query",
            "ip": request.client.host if request.client else "",
            "query": query, "hits": len(articles)})
    return {"query": query, "articles": articles}


def _update_derived(job: dict) -> None:
    """Recompute consensus and the A/P agreement signal after each sample.

    The verdict banner is judged semantically by an LLM (the two primary
    diagnosis strings can be clinically identical despite different
    spellings, qualifiers, or extra complications). The judge runs only
    when the consensus label changes; on judge failure the string
    heuristic (same_disease) is the fallback and is flagged via
    agreement_method.
    """
    agg = aggregate_a(job["a_samples"])
    job["consensus"] = agg
    p_primary = (job.get("p") or {}).get("primary_diagnosis") or ""
    if not (agg.get("consensus") and p_primary):
        job["agreement"] = None
        return

    key = (agg["consensus"]["diagnosis"], p_primary)
    if job.get("judge_key") == key:
        return  # consensus label unchanged since the last judgement
    job["judge_key"] = key

    verdict = _judge_agreement(agg["consensus"]["diagnosis"], p_primary)
    if verdict is None:
        verdict = same_disease(agg["consensus"]["diagnosis"], p_primary)
        job["agreement_method"] = "string"
    else:
        job["agreement_method"] = "llm"
    job["agreement"] = "agree" if verdict else "disagree"


def _run_a(job: dict, case_text: str, stop_event: threading.Event,
           provider: str) -> None:
    """Exactly one greedy A call; append the parsed assessment."""
    raw, _ = ri.call_llm(build_a_prompt(case_text),
                         temperature=config.A_TEMPERATURE,
                         max_tokens=config.MAX_TOKENS,
                         timeout=config.LLM_TIMEOUT, provider=provider,
                         disable_thinking=(provider == "qwen"),
                         stop_event=stop_event)
    job["a_samples"].append(parse_assessment(raw))
    _update_derived(job)


def _format_expert_opinions(experts: list) -> str:
    """Render the five expert top-5 lists for the moderator prompt."""
    lines = []
    for (title, _), top5 in zip(EXPERT_ROLES, experts):
        lines.append(f"{title}:")
        for i, item in enumerate((top5 or [])[:5], 1):
            line = f"  {i}. {item['diagnosis']}"
            if item.get("rationale"):
                line += f" — {item['rationale']}"
            lines.append(line)
    return "\n".join(lines)


def _run_p(job: dict, case_text: str, stop_event: threading.Event,
           provider: str) -> None:
    """5 isolated expert calls in parallel, then one moderator synthesis."""
    experts: list = [None] * len(EXPERT_ROLES)
    lock = threading.Lock()

    def work(idx: int, title: str, brief: str) -> None:
        if stop_event.is_set():
            return
        try:
            raw, _ = ri.call_llm(
                build_expert_prompt(title, brief, case_text),
                temperature=config.EXPERT_TEMPERATURE,
                max_tokens=config.EXPERT_MAX_TOKENS,
                timeout=config.LLM_TIMEOUT, provider=provider,
                disable_thinking=(provider == "qwen"),
                stop_event=stop_event)
        except ri.LLMCancelledError:
            return  # stop requested: this expert contributes nothing
        experts[idx] = parse_expert_top5(raw)
        with lock:
            job["p_experts_done"] = job.get("p_experts_done", 0) + 1

    with ThreadPoolExecutor(max_workers=len(EXPERT_ROLES)) as ex:
        futs = [ex.submit(work, i, t, b)
                for i, (t, b) in enumerate(EXPERT_ROLES)]
        for fut in as_completed(futs):
            fut.result()  # propagate a non-cancellation failure to the job

    if stop_event.is_set():
        return  # skip the moderator when a stop was requested

    raw, _ = ri.call_llm(
        build_moderator_prompt(case_text, _format_expert_opinions(experts)),
        temperature=config.MODERATOR_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        timeout=config.LLM_TIMEOUT, provider=provider,
        disable_thinking=(provider == "qwen"),
        stop_event=stop_event)
    job["p"] = parse_assessment(raw)
    _update_derived(job)


def _execute_job(job_id: str, case_text: str) -> None:
    """Run only the selected modes, in canonical order: A then P.

    A stop request aborts immediately: the streamed connection of the
    in-flight call is closed (its partial output is discarded) and no
    further calls start. Results already collected are kept, and the job
    ends as "stopped".
    """
    job = _JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    stop_event = threading.Event()
    job["stop_event"] = stop_event
    provider = _pick_provider()
    job["provider"] = provider
    modes = job.get("modes") or []
    try:
        if "a" in modes and not job.get("stop_requested"):
            _run_a(job, case_text, stop_event, provider)

        if "p" in modes and not job.get("stop_requested"):
            _run_p(job, case_text, stop_event, provider)

        job["status"] = "stopped" if job.get("stop_requested") else "done"
    except ri.LLMCancelledError:
        job["status"] = "stopped"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    _audit({"event": "job_finished", "job_id": job_id, "status": job["status"],
            "provider": job.get("provider")})
