#!/usr/bin/env python3
"""
Medical Diagnosis Inference - Core utilities and shared components.

This module provides:
- Budget tracking for API calls
- LLM provider abstraction (DeepSeek, Qwen/DashScope, OpenAI, Anthropic, llama.cpp)
- Output parsing utilities
- Harness contradiction detection

Used by Scheme A and Scheme B.
"""

import os
import json
import re
import time
import random
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Load .env file before other imports that might use environment variables
from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

from case_extraction import preprocess_case_text, format_structured_case
from retrieval_module import RetrievalOrchestrator

# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = os.environ.get("DATASET_PATH", "../data/mgh_qa_dataset.json")
_default_output = "../results/inference_results.jsonl"
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", _default_output)

# LLM provider selection: "deepseek-flash", "qwen", "openai", "anthropic", or "llamacpp"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek-flash").lower()


class LLMCancelledError(RuntimeError):
    """Raised when a call is aborted via its stop_event (llamacpp/qwen)."""


def _consume_openai_sse(resp, stop_event) -> Tuple[str, Dict]:
    """Read an OpenAI-compatible SSE stream, honoring stop_event.

    Lines are decoded manually: iter_lines(decode_unicode=True) is
    unreliable here (latin-1 mojibake, or raw bytes when no charset is
    given, which silently matches nothing). Closing the response on cancel
    also signals the server to stop generating (llama.cpp honours
    disconnects).
    """
    parts: List[str] = []
    usage: Dict = {}
    try:
        for raw_line in resp.iter_lines():
            if stop_event.is_set():
                raise LLMCancelledError("推理已被停止")
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            line = raw_line
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                parts.append(piece)
    finally:
        resp.close()
    return "".join(parts), usage

# Semantic-judge provider: decoupled from LLM_PROVIDER so scheme inference can
# run on a local llama.cpp server while the lightweight YES/NO semantic judge
# still uses DeepSeek (avoids doubling local inference time). Only "deepseek-flash"
# (default) routes to the dedicated judge path; any other value falls back to
# the normal call_llm dispatch (old behavior).
LLM_JUDGE_PROVIDER = os.environ.get("LLM_JUDGE_PROVIDER", "deepseek-flash").lower()

# DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("DEEP_SEEK_API", ""))
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")  # deepseek-flash = DeepSeek V4.1-Flash 的 serving alias（论文记为 deepseek-v4.1-flash）

# Qwen (DashScope OpenAI-compatible endpoint)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
QWEN_API_BASE = os.environ.get("QWEN_API_BASE",
                               "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-flash")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Local llama.cpp server (OpenAI-compatible API)
LLAMACPP_API_BASE = os.environ.get("LLAMACPP_API_BASE", "http://127.0.0.1:8080/v1")

# ---------------------------------------------------------------------------
# ER 合规路由（2026-09-24）：凡涉及 ER-Reason 数据的 API 调用一律经 OpenRouter，
# 路由层开启零数据保留（provider.zdr = true，仅选用不留存请求/响应的端点），
# 并以 data_collection = "deny" 禁止端点将数据用于训练或其他收集用途。
# 由 ER 驱动脚本设置环境变量 OPENROUTER_ER=1 启用；非 ER 流程不受影响。
OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_ER = os.environ.get("OPENROUTER_ER", "0") == "1"
OPENROUTER_MODEL_QWEN = os.environ.get("OPENROUTER_MODEL_QWEN", "qwen/qwen3.8-flash")
OPENROUTER_MODEL_DEEPSEEK = os.environ.get("OPENROUTER_MODEL_DEEPSEEK", "deepseek/deepseek-flash")


def _apply_er_openrouter(payload: dict, model_key: str):
    """ER 调用经 OpenRouter：改写端点/鉴权/模型名，并注入 zdr 与 data_collection 参数。"""
    payload = dict(payload)
    payload["model"] = {"qwen": OPENROUTER_MODEL_QWEN, "deepseek": OPENROUTER_MODEL_DEEPSEEK}[model_key]
    payload["provider"] = {**(payload.get("provider") or {}), "zdr": True}
    payload["data_collection"] = "deny"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    return payload, headers, f"{OPENROUTER_API_BASE}/chat/completions"
LLAMACPP_MODEL = os.environ.get("LLAMACPP_MODEL", "local-model")

# -----------------------------------------------------------------------------
# Retry / error-classification configuration
# -----------------------------------------------------------------------------
# Maximum attempts for retryable errors (429 / 5xx / network). Override via env.
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", "1.0"))
LLM_RETRY_MAX_DELAY = float(os.environ.get("LLM_RETRY_MAX_DELAY", "30.0"))

# ---- 全局自适应限流熔断（跨线程共享；2026-09-21 针对服务端限流风暴加装）----
# 问题：限流期间各 worker 独立做 1-4s 小退避后立即重试，形成 ~4-8 req/s 的
# 持续重试风暴，把限流窗口一次次打满（实测一夜爬行 6.8h）。对策：任何
# retryable 错误升级"全局冷却窗"，所有线程发送前在同一闸门排队，让配额窗口
# 真正歇过来；连续成功则逐级降级。
_THROTTLE_LOCK = threading.Lock()
_THROTTLE = {"pause_until": 0.0, "level": 0, "success_streak": 0}
THROTTLE_PAUSE_BASE = float(os.environ.get("THROTTLE_PAUSE_BASE", "60"))
THROTTLE_PAUSE_MAX = float(os.environ.get("THROTTLE_PAUSE_MAX", "900"))


def _throttle_gate(provider):
    """发送前闸门：处于全局冷却期则（所有线程一起）等待到窗口结束。"""
    while True:
        with _THROTTLE_LOCK:
            wait = _THROTTLE["pause_until"] - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 5.0) + random.uniform(0, 0.5))


def _throttle_event(provider, retry_after=None):
    """记录一次限流症状（429/5xx/连接类超时），升级全局冷却窗。"""
    with _THROTTLE_LOCK:
        _THROTTLE["success_streak"] = 0
        _THROTTLE["level"] = min(_THROTTLE["level"] + 1, 6)
        pause = min(THROTTLE_PAUSE_BASE * (2 ** (_THROTTLE["level"] - 1)),
                    THROTTLE_PAUSE_MAX)
        if retry_after:
            pause = max(pause, float(retry_after))
        pause += random.uniform(0, 5)
        _THROTTLE["pause_until"] = time.time() + pause
        level = _THROTTLE["level"]
    print(f"  [Throttle] {provider} 限流症状 → 全局冷却 {pause:.0f}s "
          f"(level {level})", flush=True)


def _throttle_success():
    """连续成功则逐级降级，恢复吞吐。"""
    with _THROTTLE_LOCK:
        _THROTTLE["success_streak"] += 1
        if _THROTTLE["success_streak"] >= 20 and _THROTTLE["level"] > 0:
            _THROTTLE["level"] -= 1
            _THROTTLE["success_streak"] = 0

# -----------------------------------------------------------------------------
# Cost estimation (USD per 1M tokens). Used by BudgetTracker for usage tracking.
# Values are approximate list prices; override via env to match your plan.
# -----------------------------------------------------------------------------
DEEPSEEK_INPUT_PRICE_PER_MTOK = float(os.environ.get("DEEPSEEK_INPUT_PRICE_PER_MTOK", "0.27"))
DEEPSEEK_OUTPUT_PRICE_PER_MTOK = float(os.environ.get("DEEPSEEK_OUTPUT_PRICE_PER_MTOK", "1.10"))
OPENAI_INPUT_PRICE_PER_MTOK = float(os.environ.get("OPENAI_INPUT_PRICE_PER_MTOK", "2.50"))
OPENAI_OUTPUT_PRICE_PER_MTOK = float(os.environ.get("OPENAI_OUTPUT_PRICE_PER_MTOK", "10.00"))
# Qwen (DashScope) list prices in USD per 1M tokens. qwen3.8-flash pricing is
# CNY-denominated; these are placeholders — override via env with the current
# list price (converted to USD) so cost tracking stays meaningful.
QWEN_INPUT_PRICE_PER_MTOK = float(os.environ.get("QWEN_INPUT_PRICE_PER_MTOK", "0.05"))
QWEN_OUTPUT_PRICE_PER_MTOK = float(os.environ.get("QWEN_OUTPUT_PRICE_PER_MTOK", "0.40"))


def get_current_model() -> str:
    """Return the active model name based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "qwen":
        if os.environ.get("QWEN_BACKEND", "api").strip().lower() == "local":
            return LLAMACPP_MODEL
        return QWEN_MODEL
    elif LLM_PROVIDER == "anthropic":
        return ANTHROPIC_MODEL
    elif LLM_PROVIDER == "llamacpp":
        return LLAMACPP_MODEL
    return DEEPSEEK_MODEL


def resolve_provider(provider: Optional[str] = None) -> str:
    """Resolve a logical provider name to a concrete transport.

    "qwen" is a logical name: QWEN_BACKEND=api (default) routes it to the
    DashScope API, QWEN_BACKEND=local routes it to the intranet llama.cpp
    server that serves the same qwen3.8-flash model family. "qwen-api" and
    "qwen-local" are explicit escapes that ignore QWEN_BACKEND, handy when
    one script needs both transports side by side.
    """
    p = (provider or LLM_PROVIDER).lower()
    if p == "qwen":
        backend = os.environ.get("QWEN_BACKEND", "api").strip().lower()
        p = "llamacpp" if backend == "local" else "qwen"
    elif p == "qwen-api":
        p = "qwen"
    elif p == "qwen-local":
        p = "llamacpp"
    return p


# =============================================================================
# Budget Tracking
# =============================================================================

class BudgetTracker:
    """Track and enforce budget limits across the inference pipeline.

    In addition to call / query / time limits, this can enforce cumulative
    token and cost budgets, and records usage (input/output tokens, cost) from
    every completed LLM call via record_usage().
    """

    def __init__(self, max_llm_calls=30, max_pubmed_queries=50, max_total_seconds=300,
                 max_total_tokens=None, max_cost=None,
                 input_price_per_mtok=None, output_price_per_mtok=None):
        """
        Args:
            max_llm_calls: Max LLM API calls before stopping.
            max_pubmed_queries: Max PubMed queries before stopping.
            max_total_seconds: Max wall-clock seconds before stopping.
            max_total_tokens: Optional max cumulative tokens (input+output)
                before stopping new calls. None = unlimited (default).
            max_cost: Optional max cumulative USD cost before stopping new
                calls. None = unlimited (default).
            input_price_per_mtok / output_price_per_mtok: Optional price
                overrides (USD per 1M tokens). Defaults to module-level pricing
                for the active provider.
        """
        self.max_llm_calls = max_llm_calls
        self.max_pubmed_queries = max_pubmed_queries
        self.max_total_seconds = max_total_seconds
        self.max_total_tokens = max_total_tokens
        self.max_cost = max_cost
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok
        self.llm_calls = 0
        self.pubmed_queries = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        # RLock so spend_* methods can safely call check_budget() (which also
        # acquires the lock) without deadlocking.
        self.lock = threading.RLock()
        self._cancelled = False

    def check_budget(self, operation=""):
        """Check if budget is exhausted. Returns True if OK, False if exceeded."""
        with self.lock:
            if self._cancelled:
                return False
            elapsed = time.time() - self.start_time
            if self.llm_calls >= self.max_llm_calls:
                print(f"[Budget] LLM call budget exhausted ({self.llm_calls}/{self.max_llm_calls})")
                return False
            if self.pubmed_queries >= self.max_pubmed_queries:
                print(f"[Budget] PubMed query budget exhausted ({self.pubmed_queries}/{self.max_pubmed_queries})")
                return False
            if elapsed >= self.max_total_seconds:
                print(f"[Budget] Time budget exhausted ({elapsed:.1f}s/{self.max_total_seconds}s)")
                return False
            if self.max_total_tokens is not None:
                total_used = self.total_input_tokens + self.total_output_tokens
                if total_used >= self.max_total_tokens:
                    print(f"[Budget] Token budget exhausted ({total_used}/{self.max_total_tokens})")
                    return False
            if self.max_cost is not None and self.total_cost >= self.max_cost:
                print(f"[Budget] Cost budget exhausted (${self.total_cost:.4f}/${self.max_cost:.4f})")
                return False
            return True

    def spend_llm_call(self, count=1):
        with self.lock:
            self.llm_calls += count
            return self.check_budget()

    def spend_pubmed_query(self, count=1):
        with self.lock:
            self.pubmed_queries += count
            return self.check_budget()

    def record_usage(self, usage: Dict):
        """Record token usage / cost from a completed LLM call.

        Args:
            usage: The usage dict returned by call_llm (prompt_tokens,
                completion_tokens, total_tokens, optionally cost).
        """
        with self.lock:
            try:
                in_t = int(usage.get("prompt_tokens", 0) or 0)
            except (TypeError, ValueError):
                in_t = 0
            try:
                out_t = int(usage.get("completion_tokens", 0) or 0)
            except (TypeError, ValueError):
                out_t = 0
            self.total_input_tokens += in_t
            self.total_output_tokens += out_t
            try:
                cost = float(usage.get("cost") or 0.0)
            except (TypeError, ValueError):
                cost = 0.0
            if cost <= 0.0:
                cost = self._estimate_cost(in_t, out_t)
            self.total_cost += cost

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for the given token counts (list-price estimate)."""
        if self.input_price_per_mtok is not None:
            p_in = self.input_price_per_mtok
        elif LLM_PROVIDER == "openai":
            p_in = OPENAI_INPUT_PRICE_PER_MTOK
        elif LLM_PROVIDER == "deepseek-flash":
            p_in = DEEPSEEK_INPUT_PRICE_PER_MTOK
        elif LLM_PROVIDER == "qwen":
            p_in = QWEN_INPUT_PRICE_PER_MTOK
        else:
            p_in = 0.0
        if self.output_price_per_mtok is not None:
            p_out = self.output_price_per_mtok
        elif LLM_PROVIDER == "openai":
            p_out = OPENAI_OUTPUT_PRICE_PER_MTOK
        elif LLM_PROVIDER == "deepseek-flash":
            p_out = DEEPSEEK_OUTPUT_PRICE_PER_MTOK
        elif LLM_PROVIDER == "qwen":
            p_out = QWEN_OUTPUT_PRICE_PER_MTOK
        else:
            p_out = 0.0
        return (input_tokens * p_in + output_tokens * p_out) / 1_000_000.0

    def get_status(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "llm_calls": self.llm_calls,
                "max_llm_calls": self.max_llm_calls,
                "pubmed_queries": self.pubmed_queries,
                "max_pubmed_queries": self.max_pubmed_queries,
                "elapsed_seconds": round(elapsed, 1),
                "max_total_seconds": self.max_total_seconds,
                "remaining_llm_calls": self.max_llm_calls - self.llm_calls,
                "remaining_pubmed_queries": self.max_pubmed_queries - self.pubmed_queries,
                "remaining_seconds": round(self.max_total_seconds - elapsed, 1),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost, 6),
                "max_total_tokens": self.max_total_tokens,
                "max_cost": self.max_cost,
            }

    def get_remaining_llm_calls(self):
        with self.lock:
            return self.max_llm_calls - self.llm_calls

    def get_remaining_pubmed_queries(self):
        with self.lock:
            return self.max_pubmed_queries - self.pubmed_queries

    def cancel(self):
        with self.lock:
            self._cancelled = True


# =============================================================================
# LLM calling
# =============================================================================

def _classify_llm_error(e: Exception) -> str:
    """Classify an LLM request error.

    Returns:
        "retryable": transient — HTTP 429, 5xx, network/connection/timeout.
        "fatal": permanent — 401/403, quota/billing exhaustion, bad request.
        "unknown": anything else (treated as fatal; no retries).
    """
    body = ""
    status = None
    if isinstance(e, requests.exceptions.HTTPError):
        status = e.response.status_code if e.response is not None else None
        try:
            body = e.response.text or "" if e.response is not None else ""
        except Exception:
            body = ""
    elif isinstance(e, requests.exceptions.RequestException):
        status = None

    body_lower = body.lower()
    # Billing/quota/credential exhaustion is permanent — retrying will not help.
    # (OpenAI reports quota exhaustion as HTTP 429, so this check must come
    # before the generic 429 -> retryable branch.)
    permanent_markers = (
        "insufficient_quota", "insufficient balance", "billing",
        "quota", "no balance", "payment required", "access_terminated",
        "invalid_api_key", "authentication_error", "permission_denied",
    )
    if any(marker in body_lower for marker in permanent_markers):
        return "fatal"

    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return "retryable"
    if isinstance(e, requests.exceptions.HTTPError):
        if status is not None and (status == 429 or 500 <= status < 600):
            return "retryable"
        return "fatal"
    if isinstance(e, requests.exceptions.RequestException):
        return "unknown"
    return "unknown"


def _friendly_error_reason(e: Exception, category: str) -> str:
    """Return a human-readable reason for a failed LLM request."""
    if category == "retryable":
        return "transient error (exhausted retries)"
    if isinstance(e, requests.exceptions.HTTPError):
        status = e.response.status_code if e.response is not None else None
        if status == 401:
            return "authentication failed — check API key"
        if status == 403:
            return "forbidden — API key lacks permission"
        if status == 402:
            return "payment required — billing/quota exhausted"
        if status == 429:
            return "rate limited"
        if status is not None and 500 <= status < 600:
            return "server error"
        return f"HTTP {status}"
    if isinstance(e, requests.exceptions.Timeout):
        return "request timed out"
    if isinstance(e, requests.exceptions.ConnectionError):
        return "network/connection error"
    return str(e)[:200]


def _estimate_cost(usage: Dict, provider: Optional[str] = None) -> float:
    """Estimate USD cost of a call from usage token counts.

    Uses module-level list prices (see Configuration section). Returns 0.0 for
    providers without configured pricing. This is an estimate, not a bill.
    """
    try:
        in_t = int(usage.get("prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        in_t = 0
    try:
        out_t = int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        out_t = 0
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "deepseek-flash":
        p_in, p_out = DEEPSEEK_INPUT_PRICE_PER_MTOK, DEEPSEEK_OUTPUT_PRICE_PER_MTOK
    elif provider == "openai":
        p_in, p_out = OPENAI_INPUT_PRICE_PER_MTOK, OPENAI_OUTPUT_PRICE_PER_MTOK
    elif provider == "qwen":
        p_in, p_out = QWEN_INPUT_PRICE_PER_MTOK, QWEN_OUTPUT_PRICE_PER_MTOK
    else:
        p_in, p_out = 0.0, 0.0
    return (in_t * p_in + out_t * p_out) / 1_000_000.0


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2048,
             budget_tracker: Optional[BudgetTracker] = None,
             timeout: int = 120,
             extract_reasoning: bool = True,
             disable_thinking: bool = False,
             use_json_mode: bool = False,
             max_retries: Optional[int] = None,
             enable_thinking: bool = False,
             provider: Optional[str] = None,
             stop_event: Optional[threading.Event] = None,
             er_privacy: Optional[bool] = None) -> Tuple[str, Dict]:
    # er_privacy: None=跟随进程开关 OPENROUTER_ER；True/False=调用级强制（判分类调用必须显式 False）
    """Call the configured LLM provider with the given prompt.

    Args:
        prompt: The prompt to send. Always the first positional argument.
        extract_reasoning: If True (default), try to extract the final structured
                          answer from reasoning_content when content is empty.
                          Set to False to get the full reasoning text for
                          multi-section outputs.
        disable_thinking: If True, disable thinking mode for this call.
                          Use for utility calls where structured output is needed.
        use_json_mode: If True, request structured JSON output from DeepSeek via
                       response_format={"type": "json_object"}. The returned text
                       is then a JSON object (parse it with parse_llm_output).
                       Default False keeps existing plain-text behavior, so all
                       current callers are unaffected.
        max_retries: Max retries for retryable errors (429 / 5xx / network).
                     Defaults to LLM_MAX_RETRIES (env-configurable, 3).
        provider: Per-call provider override ("deepseek-flash"/"qwen"/...). None uses LLM_PROVIDER.

    Returns:
        (raw_text, usage_dict). On error returns ("", usage_with_zeros) so the
        pipeline can continue. usage_dict includes prompt_tokens,
        completion_tokens, total_tokens and an estimated "cost" (USD).

    Retry policy:
        - Retryable (429, 5xx, connection/timeout): exponential backoff, up to
          max_retries attempts.
        - Fatal (401/403, quota/billing exhaustion, bad request): fail fast with
          a clear error message; no retries.
    """
    if budget_tracker and not budget_tracker.spend_llm_call():
        return "[BUDGET_EXHAUSTED]", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}

    provider = resolve_provider(provider)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}

    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    # Try to call the LLM with retries for transient failures
    start = time.time()
    attempt = 0
    while True:
        _throttle_gate(provider)
        try:
            route_er = OPENROUTER_ER if er_privacy is None else er_privacy
            if provider == "deepseek-flash":
                raw, usage = _call_deepseek(prompt, temperature, max_tokens, timeout,
                                            extract_reasoning, disable_thinking, use_json_mode,
                                            route_er=route_er)
            elif provider == "qwen":
                raw, usage = _call_qwen(prompt, temperature, max_tokens, timeout,
                                        extract_reasoning, disable_thinking, use_json_mode,
                                        stop_event=stop_event, route_er=route_er)
            elif provider == "openai":
                raw, usage = _call_openai(prompt, temperature, max_tokens, timeout)
            elif provider == "anthropic":
                raw, usage = _call_anthropic(prompt, temperature, max_tokens, timeout)
            elif provider == "llamacpp":
                raw, usage = _call_llamacpp(prompt, temperature, max_tokens, timeout,
                                            enable_thinking=enable_thinking,
                                            stop_event=stop_event)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            usage["cost"] = _estimate_cost(usage, provider)
            if budget_tracker:
                budget_tracker.record_usage(usage)
            _throttle_success()
            break
        except LLMCancelledError:
            # user cancellation must propagate: never retried, never swallowed
            raise
        except Exception as e:
            category = _classify_llm_error(e)
            if category == "retryable" and attempt < max_retries:
                # 限流症状升级全局冷却；Retry-After 优先
                ra = None
                try:
                    if isinstance(e, requests.exceptions.HTTPError) and \
                            e.response is not None and \
                            e.response.status_code == 429:
                        ra = e.response.headers.get("Retry-After")
                        ra = float(ra) if ra and str(ra).replace(".", "").isdigit() else None
                except Exception:
                    ra = None
                _throttle_event(provider, retry_after=ra)
                time.sleep(random.uniform(0.5, 1.5))
                attempt += 1
                continue

            reason = _friendly_error_reason(e, category)
            print(f"[LLM Error] {provider}: {e} ({reason})", flush=True)
            # Return empty response on error so pipeline can continue
            raw = ""
            break

    elapsed = time.time() - start
    print(f"  [LLM] {provider} call took {elapsed:.1f}s, tokens: {usage.get('total_tokens', 0)}", flush=True)
    return raw, usage


_HARD_LEAKS = {"n": 0}


def post_with_deadline(url, total_s, **kw):
    """requests.post with an enforced total deadline (trickle-safe).

    服务端限流时可能以字节级缓速让 socket 读超时反复重置，requests 的
    per-read timeout 因此失效：worker 会无限阻塞在 SSL read（2026-09-21
    实测，MCR 采样两度卡死均为此因）。这里用守护线程执行请求，超过总
    时限即放弃并抛 Timeout，让上层重试换新连接；被放弃的线程会在后台
    阻塞直至进程退出（设泄漏上限防 FD 耗尽）。
    """
    box = {}

    def _run():
        try:
            box["resp"] = requests.post(url, **kw)
        except BaseException as e:  # noqa: BLE001 - 原样透传给调用方
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(total_s)
    if th.is_alive():
        _HARD_LEAKS["n"] += 1
        if _HARD_LEAKS["n"] > 128:
            raise RuntimeError("hard-deadline 泄漏线程超上限，疑似持续限流")
        raise requests.exceptions.Timeout(
            f"hard deadline {total_s}s exceeded (trickle-safe)")
    if "err" in box:
        raise box["err"]
    return box["resp"]

def _call_deepseek(prompt, temperature, max_tokens, timeout, extract_reasoning=True, disable_thinking=False, use_json_mode=False, route_er=False):
    """Call DeepSeek API.

    use_json_mode: if True, request structured JSON output via
    response_format={"type": "json_object"}. DeepSeek requires the word "json"
    to appear in the prompt for JSON mode; it is appended if absent.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # JSON mode: ask the model to return a structured JSON object.
    if use_json_mode:
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond with a valid JSON object only."
        payload["response_format"] = {"type": "json_object"}
        payload["messages"] = [{"role": "user", "content": prompt}]
        # Reasoning models put the answer in reasoning_content and leave
        # content empty when thinking is enabled. For structured JSON output we
        # must disable thinking so the JSON lands in content where it can be
        # parsed. (JSON mode + thinking chains are incompatible.)
        payload["thinking"] = {"type": "disabled"}

    # Disable thinking mode for utility calls (structured output needed)
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}

    url = f"{DEEPSEEK_API_BASE}/chat/completions"
    if route_er:  # ER 合规路由：OpenRouter + zdr + data_collection deny
        payload, headers, url = _apply_er_openrouter(payload, "deepseek")
    resp = post_with_deadline(
        url,
        max(timeout * 2, 600),
        headers=headers,
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    text = message.get("content", "")

    # Fallback: if content is empty but reasoning_content exists
    if not text and "reasoning_content" in message:
        reasoning = message.get("reasoning_content", "")
        print(f"  [DEBUG] Content empty, reasoning_content length: {len(reasoning)}")
        if extract_reasoning:
            # Try to extract the final structured answer from the END of
            # reasoning. The thinking-mode prompts instruct the model to
            # conclude with a final answer block, so the LAST
            # "MOST_LIKELY_DIAGNOSIS:" header marks its start. If no block
            # exists, return "" (a counted miss) rather than mid-thinking
            # rambling text.
            extracted = _extract_final_answer_from_reasoning_tail(reasoning)
            if extracted:
                text = extracted
                print(f"  [DEBUG] Extracted structured answer from reasoning, length: {len(text)}")
            else:
                text = ""
                print(f"  [DEBUG] No final answer block in reasoning; returning empty (counted miss)")
        else:
            text = reasoning
            print(f"  [DEBUG] Using full reasoning_content, length: {len(text)}")

    usage = data.get("usage", {})
    return text, usage


def _extract_final_answer_from_reasoning(reasoning: str) -> str:
    """Try to extract the final structured answer from reasoning content.

    The model often ends its reasoning with the final answer in the requested format.
    Look for the last occurrence of structured field markers.
    """
    if not reasoning:
        return ""

    # Look for the last section that contains structured fields
    structured_markers = [
        "MOST_LIKELY_DIAGNOSIS:",
        "REFLECTION_SUMMARY:",
        "FINAL DIAGNOSIS:",
        "DIAGNOSIS:",
        "CANDIDATE_LIST:",
        "GAP_ANALYSIS:",
        "ROUND_SUMMARY:",
        "QUERIES:",
        "UPDATED_CANDIDATES:",
        "TOP_TWO:",
        "OUTPUT_LIST:",
        "CANDIDATE_SCORES:",
        "SELECTED_DIAGNOSIS:",
        "CONFIDENCE_SCORE:",
        "CLEANED_CANDIDATES:",
        "MISSED_DIAGNOSES:",
        "FILTERED_CANDIDATES:",
        "DISCARDED:",
    ]

    last_marker_pos = -1
    for marker in structured_markers:
        pos = reasoning.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos

    if last_marker_pos >= 0:
        candidate = reasoning[last_marker_pos:]
        return candidate.strip()

    return ""


def _extract_final_answer_from_reasoning_tail(reasoning: str) -> str:
    """Extract the FINAL answer block from the END of reasoning_content.

    deepseek-flash with thinking ON puts everything (deliberation + answer)
    in reasoning_content while content stays empty. The model writes TENTATIVE
    structured markers mid-thinking and keeps reasoning after them, so the old
    "last marker anywhere" logic returns mid-thinking slices. The thinking-mode
    prompts instruct the model to conclude with a final answer block as the
    very last section, so the LAST "MOST_LIKELY_DIAGNOSIS:" header marks the
    start of the true final block: return everything from there to the end of
    reasoning_content.

    If the model rambled after its final block (e.g. a trailing remark after
    CONFIDENCE_SCORE), the tail is trimmed at the last CONFIDENCE_SCORE.

    Returns "" when no answer block header is present so callers fall back to
    "Unable to determine diagnosis" instead of returning rambling mid-thinking
    text.
    """
    if not reasoning:
        return ""

    header = "MOST_LIKELY_DIAGNOSIS:"
    pos = reasoning.rfind(header)
    if pos < 0:
        # No answer-block header. Fall back to scanning the tail of
        # reasoning_content for a concluding-diagnosis sentence: the model may
        # burn its whole token budget deliberating and only end with a prose
        # conclusion like `The final diagnosis in CPC likely "X"`. Return ""
        # when nothing matches so the caller falls back to "Unable to determine
        # diagnosis" (a counted miss, never a garbage meta-commentary line).
        return _extract_prose_conclusion(reasoning)

    tail = reasoning[pos:].strip()

    # Trim anything the model wrote after its final answer block. Keep through
    # the last CONFIDENCE_SCORE value (which ends the block).
    score_matches = list(re.finditer(r"CONFIDENCE_SCORE\s*:\s*\d+", tail,
                                     re.IGNORECASE))
    if score_matches:
        return tail[:score_matches[-1].end()].rstrip()
    return tail


# Conclusion-zone width: the model's concluding sentence lives in the last few
# hundred chars of reasoning_content.
CONCLUSION_ZONE_CHARS = 800

# Concluding-diagnosis phrase patterns (prose fallback). The model sometimes
# spends its entire token budget deliberating and never writes the answer
# block, ending instead with a prose conclusion such as:
#   The final diagnosis in CPC likely "Hypothyroidism due to Hashimoto's
#   thyroiditis"
#   Most likely diagnosis: Sarcoidosis
#   The most likely diagnosis is Influenza A virus infection
_PROSE_DOUBLE_QUOTED_RE = re.compile(
    r"(?:final diagnosis|most likely diagnosis|final answer|diagnosis)\b"
    r"[^\"\n]{0,90}?\"([^\"]{3,140})\"",
    re.IGNORECASE | re.DOTALL,
)
# Single-quoted variant. Kept separate from the double-quoted pattern so an
# apostrophe inside a quoted diagnosis (e.g. "Hashimoto's thyroiditis") is not
# mistaken for a closing quote.
_PROSE_SINGLE_QUOTED_RE = re.compile(
    r"(?:final diagnosis|most likely diagnosis|final answer|diagnosis)\b"
    r"[^'\n]{0,90}?'([^']{3,140})'",
    re.IGNORECASE | re.DOTALL,
)
_PROSE_COLON_RE = re.compile(
    r"(?:most likely diagnosis|final diagnosis|diagnosis)\s*:\s*([^\n]{3,160})",
    re.IGNORECASE,
)
_PROSE_IS_RE = re.compile(
    r"(?:the most likely diagnosis|the final diagnosis|the diagnosis|"
    r"most likely diagnosis|final diagnosis)"
    r"\s+(?:is|was|will be|probably is|likely is)\s+([^\n]{3,160})",
    re.IGNORECASE,
)
_PROSE_FINAL_RE = re.compile(
    r"the final diagnosis(?:\s+in CPC)?\s+(?:is|was|will be|likely|probably)\s+"
    r"([A-Z][^\n]{3,160})",
    re.IGNORECASE,
)

# Leading meta-commentary that is never a real diagnosis.
_PROSE_BLACKLIST_PREFIXES = (
    "we need", "let's", "let us", "question", "actually", "think",
    "i think", "the question", "i need",
)


def _clean_prose_diagnosis(phrase: str) -> str:
    """Normalize a captured prose-conclusion phrase into a diagnosis string."""
    phrase = phrase.strip()
    if len(phrase) >= 2 and phrase[0] in "\"'" and phrase[-1] == phrase[0]:
        phrase = phrase[1:-1].strip()
    # Stop at the end of the first sentence.
    phrase = re.split(r"[.;]", phrase, maxsplit=1)[0].strip()
    # Strip trailing punctuation and leading articles/verbs/hedges.
    phrase = re.sub(
        r"^(?:is |was |will be |maybe |probably |likely |the |a |an |"
        r"that is |which is )+",
        "", phrase, flags=re.I,
    )
    return phrase.strip()


def _looks_like_diagnosis(phrase: str) -> bool:
    low = phrase.strip().lower()
    if len(low) < 3:
        return False
    return not low.startswith(_PROSE_BLACKLIST_PREFIXES)


def _extract_prose_conclusion(reasoning: str) -> str:
    """Scan the tail of reasoning_content for a concluding-diagnosis sentence.

    Used when the model never emitted a MOST_LIKELY_DIAGNOSIS: answer block
    (it spent its whole token budget deliberating). Only the last
    CONCLUSION_ZONE_CHARS are examined. Quoted/italicized phrases are preferred
    (most reliable), then the colon form, then the "is" form, then
    "The final diagnosis (in CPC) <verb> X".
    """
    tail = reasoning[-CONCLUSION_ZONE_CHARS:]

    for pattern in (_PROSE_DOUBLE_QUOTED_RE, _PROSE_SINGLE_QUOTED_RE,
                    _PROSE_COLON_RE, _PROSE_IS_RE, _PROSE_FINAL_RE):
        matches = list(pattern.finditer(tail))
        if not matches:
            continue
        cleaned = _clean_prose_diagnosis(matches[-1].group(1))
        if _looks_like_diagnosis(cleaned):
            return cleaned
    return ""


def call_llm_judge(prompt: str, temperature: float = 0.0, max_tokens: int = 10,
                   timeout: int = 120,
                   max_retries: Optional[int] = None) -> Tuple[str, Dict]:
    """Call the semantic-judge LLM, decoupled from LLM_PROVIDER.

    The judge always uses the DeepSeek API (LLM_JUDGE_PROVIDER == "deepseek-flash",
    the default) so that scheme inference on a local llama.cpp server is not
    slowed by a local judge call. disable_thinking=True is applied on the
    DeepSeek path: deepseek-flash is a reasoner and the YES/NO verdict must
    land in content, not reasoning_content.

    If LLM_JUDGE_PROVIDER is explicitly set to another provider, this falls
    back to the normal call_llm dispatch (old behavior preserved).

    Returns:
        (raw_text, usage_dict), same contract as call_llm. On error returns
        ("", usage_with_zeros) so callers can continue.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost": 0.0}

    if LLM_JUDGE_PROVIDER != "deepseek-flash":
        # Explicit non-DeepSeek judge provider: use the normal dispatch so the
        # judge follows LLM_PROVIDER (old behavior).
        return call_llm(prompt, temperature=temperature, max_tokens=max_tokens,
                        timeout=timeout, disable_thinking=True)

    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    # Dedicated DeepSeek judge path: no budget_tracker, no LLM_PROVIDER
    # routing. Retry only transient errors (429 / 5xx / network); fatal errors
    # fail fast and return "" so the caller can continue.
    attempt = 0
    while True:
        try:
            raw, usage = _call_deepseek(
                prompt, temperature, max_tokens, timeout,
                extract_reasoning=True, disable_thinking=True,
                route_er=False)  # 裁判只传诊断字符串，恒直连官方端点
            usage["cost"] = _estimate_cost(usage, "deepseek-flash")
            return raw, usage
        except Exception as e:
            category = _classify_llm_error(e)
            if category == "retryable" and attempt < max_retries:
                delay = min(LLM_RETRY_BASE_DELAY * (2 ** attempt),
                            LLM_RETRY_MAX_DELAY)
                delay += random.uniform(0, 0.5)
                print(
                    f"  [Retry] judge deepseek-flash retryable error "
                    f"(attempt {attempt + 1}/{max_retries}): {e} — "
                    f"waiting {delay:.1f}s")
                time.sleep(delay)
                attempt += 1
                continue

            reason = _friendly_error_reason(e, category)
            print(f"[LLM Error] judge deepseek-flash: {e} ({reason})")
            return "", usage


def _call_qwen(prompt, temperature, max_tokens, timeout, extract_reasoning=True,
               disable_thinking=False, use_json_mode=False, stop_event=None, route_er=False):
    """Call Qwen via the DashScope OpenAI-compatible endpoint.

    qwen3.8-flash is a hybrid reasoner: thinking is ON by default and the
    answer lands in reasoning_content with content empty. Non-streaming calls
    disable thinking via the DashScope-specific top-level "enable_thinking"
    flag (verified: with enable_thinking=false the reply has no
    reasoning_content and the answer appears in content).

    use_json_mode: if True, request structured JSON output via
    response_format={"type": "json_object"}. Thinking is force-disabled on
    this path so the JSON lands in content where it can be parsed.
    """
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": not disable_thinking,
    }

    if use_json_mode:
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond with a valid JSON object only."
        payload["response_format"] = {"type": "json_object"}
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["enable_thinking"] = False

    if stop_event is not None:
        # streaming branch so a stop request aborts the generation
        # immediately (same SSE handling as the llamacpp path)
        if stop_event.is_set():
            raise LLMCancelledError("推理已被停止")
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        q_url = f"{QWEN_API_BASE}/chat/completions"
        if route_er:  # ER 合规路由：OpenRouter + zdr + data_collection deny
            payload, headers, q_url = _apply_er_openrouter(payload, "qwen")
        resp = requests.post(
            q_url,
            headers=headers,
            json=payload,
            timeout=(10, timeout),
            stream=True,
        )
        resp.raise_for_status()
        return _consume_openai_sse(resp, stop_event)

    url = f"{QWEN_API_BASE}/chat/completions"
    if route_er:  # ER 合规路由：OpenRouter + zdr + data_collection deny
        payload, headers, url = _apply_er_openrouter(payload, "qwen")
    resp = post_with_deadline(
        url,
        max(timeout * 2, 600),
        headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    text = message.get("content", "")

    # Fallback: if content is empty but reasoning_content exists
    if not text and message.get("reasoning_content"):
        reasoning = message.get("reasoning_content", "")
        print(f"  [DEBUG] Qwen content empty, reasoning_content length: {len(reasoning)}")
        if extract_reasoning:
            extracted = _extract_final_answer_from_reasoning_tail(reasoning)
            text = extracted
        else:
            text = reasoning

    usage = data.get("usage", {})
    return text, usage


def _call_openai(prompt, temperature, max_tokens, timeout):
    """Call OpenAI API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


def _call_anthropic(prompt, temperature, max_tokens, timeout):
    """Call Anthropic API."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content", "")
    # Reasoning models (e.g. Qwen with deepseek-reasoner compat) put output in reasoning_content.
    if not text:
        text = msg.get("reasoning_content", "")
    usage = data.get("usage", {})
    return text, usage


# llama.cpp servers typically run with a single slot: concurrent requests are
# rejected with 503 instead of queued. Serialize all calls process-wide so
# parallel callers (e.g. Scheme B self-consistency threads) queue up here.
_LLAMACPP_LOCK = threading.Lock()


def _call_llamacpp(prompt, temperature, max_tokens, timeout, enable_thinking=False,
                   stop_event=None):
    """Call local llama.cpp server.

    When *stop_event* is given the request runs as SSE streaming and the
    stream is checked between chunks: once the event is set the connection
    is closed (which also cancels generation server-side) and
    LLMCancelledError is raised. Without it, behavior is unchanged
    non-streaming.
    """
    with _LLAMACPP_LOCK:
        return _call_llamacpp_locked(prompt, temperature, max_tokens, timeout,
                                     enable_thinking=enable_thinking,
                                     stop_event=stop_event)


def _call_llamacpp_locked(prompt, temperature, max_tokens, timeout, enable_thinking=False,
                          stop_event=None):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLAMACPP_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        # Qwen3-style hybrid models run in no-think mode by default on
        # llama.cpp; enable the thinking branch via chat template kwargs.
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    if stop_event is not None:
        if stop_event.is_set():
            raise LLMCancelledError("推理已被停止")
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        resp = requests.post(
            f"{LLAMACPP_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, timeout),
            stream=True,
        )
        resp.raise_for_status()
        return _consume_openai_sse(resp, stop_event)

    resp = requests.post(
        f"{LLAMACPP_API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        # (connect, read): fail fast when the server is unreachable, but keep
        # the long budget for actual generation
        timeout=(10, timeout)
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


def call_llm_with_retry(prompt, temperature=0.3, max_tokens=2048,
                        budget_tracker=None, max_retries=2):
    """Call LLM with simple retry logic."""
    for attempt in range(max_retries + 1):
        try:
            return call_llm(prompt, temperature, max_tokens, budget_tracker)
        except LLMCancelledError:
            raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [Retry] LLM call failed (attempt {attempt+1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [Fail] LLM call failed after {max_retries+1} attempts: {e}")
                raise
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# =============================================================================
# Output parsing
# =============================================================================

def _find_json_object(text: str) -> Optional[dict]:
    """Locate and parse the first JSON object embedded in arbitrary text.

    Tries the whole text first (covers DeepSeek JSON mode output), then
    progressively extracts balanced-brace substrings (covers JSON wrapped in
    markdown fences or surrounded by commentary). Returns None if no valid JSON
    object is found.
    """
    if not text:
        return None

    clean = text.strip()
    # Strip markdown code fences so fenced JSON parses directly.
    clean = re.sub(r'^```[a-zA-Z0-9_\-]*\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean).strip()

    # Candidate 1: the whole (cleaned) text is a JSON object.
    if clean.startswith('{') and clean.endswith('}'):
        try:
            obj = json.loads(clean)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass

    # Candidate 2: scan for balanced-brace JSON objects inside the text.
    stack = []
    start = -1
    for i, ch in enumerate(clean):
        if ch == '{':
            if not stack:
                start = i
            stack.append(i)
        elif ch == '}' and stack:
            open_pos = stack.pop()
            if not stack:
                candidate = clean[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (ValueError, TypeError):
                    continue
    return None


def _parse_json_fields(raw_text: str) -> Optional[Dict]:
    """Try to parse structured fields from a JSON-formatted LLM response.

    Recognizes a broad set of field names and maps them onto the
    parse_llm_output result contract (most_likely_diagnosis,
    differential_diagnosis, reasoning) plus forward-compat aliases
    (final_diagnosis, ranked_diagnoses).

    Returns None when no valid JSON object is found, so the caller can fall
    back to the regex-based parser.
    """
    obj = _find_json_object(raw_text)
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None

    result = {}

    # --- final diagnosis ---
    diagnosis = (
        obj.get("final_diagnosis")
        or obj.get("most_likely_diagnosis")
        or obj.get("diagnosis")
        or obj.get("final_dx")
        or obj.get("leading_diagnosis")
        or ""
    )
    result["most_likely_diagnosis"] = str(diagnosis).strip()

    # --- ranked / differential list ---
    ranked = (
        obj.get("ranked_diagnoses")
        if obj.get("ranked_diagnoses") is not None
        else obj.get("differential_diagnosis")
        if obj.get("differential_diagnosis") is not None
        else obj.get("differential")
        if obj.get("differential") is not None
        else obj.get("candidates")
    )
    diffs = []
    if isinstance(ranked, list):
        for item in ranked:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    diffs.append(item)
            elif isinstance(item, dict):
                d = item.get("diagnosis") or item.get("name") or item.get("label") or item.get("item") or ""
                if d:
                    diffs.append(str(d).strip())
    elif isinstance(ranked, str):
        diffs = [d.strip().rstrip('.').rstrip(',') for d in re.split(r'[,;]|\d+\.', ranked) if d.strip()]
    # De-duplicate preserving order.
    seen = set()
    unique_diffs = []
    for d in diffs:
        key = d.lower()
        if d and key not in seen:
            seen.add(key)
            unique_diffs.append(d)
    result["differential_diagnosis"] = unique_diffs

    # --- reasoning ---
    reasoning = (
        obj.get("reasoning")
        or obj.get("explanation")
        or obj.get("rationale")
        or obj.get("analysis")
        or ""
    )
    result["reasoning"] = str(reasoning).strip()

    # --- optional confidence (informational) ---
    conf = obj.get("confidence_score")
    if conf is None:
        conf = obj.get("confidence")
    if conf is not None:
        try:
            result["confidence_score"] = float(conf)
        except (TypeError, ValueError):
            pass

    # --- forward-compat aliases used by scheme runners ---
    result["final_diagnosis"] = result["most_likely_diagnosis"]
    ranked_list = [result["most_likely_diagnosis"]] if result["most_likely_diagnosis"] else []
    seen = {r.lower() for r in ranked_list}
    for d in result["differential_diagnosis"]:
        if d and d.lower() not in seen:
            seen.add(d.lower())
            ranked_list.append(d)
    result["ranked_diagnoses"] = ranked_list[:10]

    return result


def parse_llm_output(raw_text: str) -> Dict:
    """Parse LLM output into structured fields.

    JSON-first: if the response is (or embeds) a JSON object, fields are read
    from it (final_diagnosis / most_likely_diagnosis, ranked_diagnoses /
    differential_diagnosis, reasoning, ...). Otherwise the legacy regex parser
    handles the text format:

      MOST_LIKELY_DIAGNOSIS: <diagnosis>
      DIFFERENTIAL_DIAGNOSIS: <d1>, <d2>, ...
      REASONING: <reasoning text>

    Robust to: markdown bold markers, multi-line diagnoses,
    alternate field names, extra commentary, and markdown code blocks.

    Returns a dict with keys: most_likely_diagnosis, differential_diagnosis,
    reasoning, raw, plus aliases final_diagnosis and ranked_diagnoses.
    """
    result = {
        "most_likely_diagnosis": "",
        "differential_diagnosis": [],
        "reasoning": "",
        "raw": raw_text,
        "final_diagnosis": "",
        "ranked_diagnoses": [],
    }
    if not raw_text:
        return result

    # --- Preferred path: structured JSON ---
    json_parsed = _parse_json_fields(raw_text)
    if json_parsed is not None:
        result.update(json_parsed)
        result["raw"] = raw_text
        # If JSON mode gave no reasoning, try the regex reasoning fallback.
        if not result.get("reasoning"):
            _fill_reasoning_from_regex(result, _strip_markup(raw_text))
        return result

    clean = _strip_markup(raw_text)

    # --- MOST_LIKELY_DIAGNOSIS ---
    # Try multiple patterns, from most specific to most lenient
    diag_patterns = [
        # Standard: capture until next field (DIFFERENTIAL, REASONING, CHANGES, CONFIDENCE, etc.)
        r'(?:MOST_LIKELY_DIAGNOSIS|MOST LIKELY DIAGNOSIS|DIAGNOSIS)\s*:\s*(.+?)(?=\n\s*(?:DIFFERENTIAL|REASONING|CHANGES|CONFIDENCE|REFLECTION|UNCERTAINTY|SHOULD|NOTE|IMPORTANT|\*\*|$))',
        # Multi-line: field name on one line, value on next
        r'(?:MOST_LIKELY_DIAGNOSIS|MOST LIKELY DIAGNOSIS|DIAGNOSIS)\s*:\s*\n\s*(.+?)(?:\n(?:[A-Z]|\*\*)|$)',
        # Fallback: just "Diagnosis:" followed by text (single line)
        r'(?:^|\n)\s*Diagnosis\s*:\s*\n?\s*([A-Z][^\n]{5,200})',
    ]
    for pat in diag_patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            diag = m.group(1).strip().rstrip('.').rstrip(',')
            if diag and len(diag) > 1:
                result["most_likely_diagnosis"] = diag
                break

    # --- DIFFERENTIAL_DIAGNOSIS ---
    diff_patterns = [
        r'(?:DIFFERENTIAL_DIAGNOSIS|DIFFERENTIAL DIAGNOSIS|DIFFERENTIAL)\s*:\s*(.+?)(?:\n(?:[A-Z]|\*\*)|$)',
        r'(?:DIFFERENTIAL_DIAGNOSIS|DIFFERENTIAL DIAGNOSIS)\s*:\s*\n\s*(.+?)(?:\n(?:[A-Z]|\*\*)|$)',
        # Fallback: look for differential list after diagnosis
        r'(?:Differential|DDx|D/D)\s*[:\s]*\n?\s*([A-Z].{10,300}?)(?:\n\n|\n[A-Z]|$)',
    ]
    for pat in diff_patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            diff_text = m.group(1).strip().rstrip('.')
            if diff_text and diff_text.lower() not in ('none', 'n/a', ''):
                # Split by comma, semicolon, or numbered list
                diffs = [d.strip().rstrip('.').rstrip(',') for d in re.split(r'[,;]|\d+\.', diff_text) if d.strip()]
                result["differential_diagnosis"] = diffs
                break

    # --- REASONING ---
    _fill_reasoning_from_regex(result, clean)

    # --- forward-compat aliases used by scheme runners ---
    result["final_diagnosis"] = result["most_likely_diagnosis"]
    ranked_list = [result["most_likely_diagnosis"]] if result["most_likely_diagnosis"] else []
    seen = {r.lower() for r in ranked_list}
    for d in result["differential_diagnosis"]:
        if d and d.lower() not in seen:
            seen.add(d.lower())
            ranked_list.append(d)
    result["ranked_diagnoses"] = ranked_list[:10]

    return result


def _strip_markup(raw_text: str) -> str:
    """Strip markdown code fences and bold markers for regex matching."""
    clean = raw_text.strip()
    if clean.startswith('```'):
        # Remove opening code fence
        clean = clean.split('\n', 1)[1] if '\n' in clean else clean[3:]
    if clean.endswith('```'):
        clean = clean[:-3].strip()
    # Strip markdown bold markers for easier matching
    clean = clean.replace('**', '')
    return clean


def _fill_reasoning_from_regex(result: Dict, clean: str) -> None:
    """Extract the REASONING field (or fallback) into result['reasoning']."""
    # Match from REASONING to end or next section header
    m = re.search(r'REASONING\s*:\s*(.+)', clean, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        # Cut off at next section header if present
        next_header = re.search(r'\n(?:[A-Z][A-Z_]+|CHANGES_MADE|CONFIDENCE)\s*:', reasoning)
        if next_header:
            reasoning = reasoning[:next_header.start()]
        result["reasoning"] = reasoning.strip()
    else:
        # Fallback: try to capture any paragraph after diagnosis/differential
        # Look for a sentence that starts with common reasoning words
        reasoning_fallback = re.search(
            r'(?:The patient|Given|Based on|Considering|This is|In this case|Therefore|However|Additionally)\s*[^\n]{20,500}',
            clean, re.IGNORECASE
        )
        if reasoning_fallback:
            result["reasoning"] = reasoning_fallback.group(0).strip()


# =============================================================================
# Harness: Contradiction detection and revision
# =============================================================================

def run_harness_check(case_text: str, original: str, differential: List[str],
                      candidate_literature: Dict, budget_tracker: Optional[BudgetTracker] = None) -> Dict:
    """Run the harness to check for contradictions in the diagnosis.

    Returns dict with:
        - verdict: "KEEP", "REVISE", or "ERROR"
        - final_diagnosis: the diagnosis to use
        - contradictions: list of contradiction strings
    """
    if not original:
        return {"verdict": "ERROR", "final_diagnosis": original, "contradictions": []}

    # Build differential text
    differential_text = ", ".join(differential[:5]) if differential else "None"

    # Build harness prompt
    harness_prompt = f"""You are a rigorous medical reviewer checking a clinician's diagnosis for internal contradictions with the case findings.

Case: {case_text[:10000]}

Clinician's diagnosis: {original}
Clinician's differential: {differential_text}

Your task: Identify any contradictions between the diagnosis and the case findings.
A contradiction is a case finding that is INCOMPATIBLE with the diagnosis (not just unusual).

Respond in EXACTLY this format:
VERDICT: [PASS or FAIL]
CONTRADICTIONS: [If FAIL, list contradictions. If PASS, write "None"]
EXPLANATION: [One sentence explaining your reasoning]
"""

    try:
        raw_harness, usage_harness = call_llm(harness_prompt, budget_tracker=budget_tracker)

        # Parse harness output
        verdict = "ERROR"
        contradictions = []

        m = re.search(r'VERDICT[:\s]+(PASS|FAIL)', raw_harness, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()

        m = re.search(r'CONTRADICTIONS[:\s]+(.+?)(?:\n|$)', raw_harness, re.IGNORECASE)
        if m:
            contra_text = m.group(1).strip()
            if contra_text.lower() != "none":
                contradictions = [c.strip() for c in contra_text.split(";") if c.strip()]

        print(f"  Harness verdict: {verdict}")
        if contradictions:
            print(f"  Contradictions: {contradictions}")

        # If FAIL with contradictions, request revision
        if verdict == "FAIL" and contradictions:
            print("[Harness] Hard contradiction found — requesting single revision...")

            contradiction_lines = "\n".join([f"{i+1}. {c}" for i, c in enumerate(contradictions)])
            revision_prompt = f"""You are a senior attending. A reviewer flagged contradictions in your CPC diagnosis. Review them CRITICALLY — the reviewer may be WRONG, but if the contradiction is valid you MUST revise.

Case: {case_text[:12000]}

Your initial diagnosis: {original}
Your differential: {differential_text}

Reviewer's claimed contradictions:
{contradiction_lines}

## RULES FOR REVISION

You should REVISE your diagnosis when:
1. At least one contradiction cites an EXPLICIT case finding (lab value, imaging result, biopsy, physical exam sign) that directly contradicts your diagnosis
2. The cited finding is ACTUALLY in the case (the reviewer may have hallucinated or misinterpreted)
3. There is NO reasonable explanation for why this finding could coexist with your diagnosis
4. The alternative diagnosis explains MORE findings with FEWER contradictions

You should KEEP your diagnosis when:
- The reviewer's contradiction is based on what "typically" happens (CPC cases are ATYPICAL)
- The reviewer proposes an alternative that you already considered and ranked lower for good reasons
- The reviewer's contradiction is circumstantial, not explicit
- You are UNSURE → KEEP

## SPECIAL RULE FOR CONTRADICTIONS INVOLVING KEY DIAGNOSTIC FINDINGS
If the reviewer identifies a key diagnostic finding that is INCOMPATIBLE with your diagnosis, this is strong evidence for revision. Examples:
- Echocardiographic findings of HOCM (asymmetric hypertrophy, SAM, outflow tract gradient) argue against cardiac amyloidosis
- Normal initial potassium argues against severe hypokalemia as the cause of cardiac arrest
- A negative toxicology screen for acetaminophen argues against acetaminophen toxicity
- Superficial anal fissures without deep ulceration/lymphadenopathy argue against LGV proctitis
- Severe hypercalcemia with normal phosphorus and sclerotic bone lesions argues against simple milk-alkali syndrome

## SPECIAL RULE FOR UPTODATE KNOWLEDGE
If UpToDate differential knowledge was provided in the literature section:
- UpToDate represents authoritative clinical knowledge — weight it heavily when it provides specific differential guidance
- If UpToDate highlights a key differentiating feature that strongly favors an alternative diagnosis → this is strong evidence for revision
- If UpToDate confirms your diagnosis as the most likely based on the case findings → this supports keeping your diagnosis

## IMPORTANT
If you decide to REVISE, you MAY propose a diagnosis that was NOT in your initial differential, IF it explains the case better than your current diagnosis. Be decisive — do not keep a diagnosis that is clearly contradicted by objective findings.

Respond in EXACTLY this format:
VERDICT: [REVISE or KEEP]
REVISED_DIAGNOSIS: [If REVISE, new diagnosis. If KEEP, write "None"]
REASON: [One sentence]
"""
            try:
                raw_rev, usage_rev = call_llm(revision_prompt)

                # Robust parsing: try multiple patterns for VERDICT
                rev_verdict = None
                verdict_patterns = [
                    r'VERDICT:\s*(REVISE|KEEP)',
                    r'VERDICT\s*[:\-]?\s*(REVISE|KEEP)',
                    r'\*\*VERDICT:\*\*\s*(REVISE|KEEP)',
                    r'VERDICT\s+(REVISE|KEEP)',
                ]
                for pattern in verdict_patterns:
                    rev_verdict = re.search(pattern, raw_rev, re.IGNORECASE)
                    if rev_verdict:
                        break

                if rev_verdict:
                    rev_verdict_str = rev_verdict.group(1).upper()
                    if rev_verdict_str == "REVISE":
                        # Extract revised diagnosis
                        rev_diag = None
                        diag_patterns = [
                            r'REVISED_DIAGNOSIS[:\s]+(.+?)(?:\n|$)',
                            r'REVISED_DIAGNOSIS\s*[:\-]?\s*(.+)',
                        ]
                        for pattern in diag_patterns:
                            rev_diag = re.search(pattern, raw_rev, re.IGNORECASE)
                            if rev_diag:
                                break

                        if rev_diag:
                            new_diag = rev_diag.group(1).strip()
                            if new_diag.lower() not in ["none", "n/a", ""]:
                                print(f"  [Harness] Diagnosis revised to: {new_diag}")
                                return {
                                    "verdict": "REVISE",
                                    "final_diagnosis": new_diag,
                                    "contradictions": contradictions
                                }

                print("  [Harness] Keeping original diagnosis")
                return {
                    "verdict": "KEEP",
                    "final_diagnosis": original,
                    "contradictions": contradictions
                }

            except Exception as e:
                print(f"  [Harness] Revision failed: {e}, keeping original")
                return {
                    "verdict": "KEEP",
                    "final_diagnosis": original,
                    "contradictions": contradictions
                }

        return {
            "verdict": verdict,
            "final_diagnosis": original,
            "contradictions": contradictions
        }

    except Exception as e:
        print(f"  [Harness] Error: {e}")
        return {
            "verdict": "ERROR",
            "final_diagnosis": original,
            "contradictions": []
        }

