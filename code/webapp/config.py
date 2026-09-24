"""Shared config for the intranet MDT webapp.

Must be imported BEFORE run_inference / scheme_perspective: module-level
constants there read environment variables at import time, so the llama.cpp
endpoint default is set here first.
"""
import os
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATIC_DIR = WEBAPP_DIR / "static"
JOBS_DIR = WEBAPP_DIR / "jobs"
LOGS_DIR = WEBAPP_DIR / "logs"
AUDIT_LOG = LOGS_DIR / "audit.jsonl"

# Local llama.cpp server (Qwen3.8-Flash-Next IQ4_XS GGUF, 32K context).
os.environ.setdefault("LLAMACPP_API_BASE", "http://10.88.128.90:8081/v1")

sys.path.insert(0, str(SCRIPTS_DIR))

A_TEMPERATURE = 0.0            # fixed single greedy A call
EXPERT_TEMPERATURE = 0.3       # 5 isolated expert calls
MODERATOR_TEMPERATURE = 0.3    # host synthesis
MAX_TOKENS = 4096
EXPERT_MAX_TOKENS = 4096
LLM_TIMEOUT = 1800    # local hardware: a 4k-token generation takes minutes

MAX_INPUT_CHARS = 20000  # keep prompt + generation inside the 32K window

PUBMED_RETMAX = 6

# Inference backend selection: "auto" (default) prefers the intranet
# llama.cpp server and falls back to the DashScope cloud API when it is
# unreachable; "llamacpp" / "qwen" force one transport.
PROVIDER_MODE = os.environ.get("MDT_PROVIDER_MODE", "auto")
