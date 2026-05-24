"""C7 — RKLLM ctypes wrapper for RK3588 NPU inference.

This module provides:
  - RKLLMRuntime: thin ctypes wrapper around librkllmrt.so
  - RKLLMBackend: async-iterator adapter that emits SSE-shaped events
    (matches MockBackend's run_troubleshoot signature so the bridge
    in tool_call_loop.py doesn't need to care which backend is wired)
  - try_load(): returns an RKLLMBackend instance if both librkllmrt.so
    AND a model file are loadable on the current host; else returns None
    so app.py's lifespan can fall back to MockBackend cleanly

The ctypes wrapper mirrors the loyal-agent template (functionland/loyal-
agent/app.py). The model file and the .so are EXTERNAL operational
dependencies; this module assumes they're present at well-known paths
inside the container (vendored at /app/vendor/rkllm/librkllmrt.so;
model bind-mounted at /uniondrive/blox-ai/model/<file>.rkllm).

What this module CAN'T do without the model file:
  - real Qwen 3B inference
  - native tool-call grammar parsing (Qwen's native function-calling
    JSON format must be parsed from the streamed token output; the
    parser is sketched here but exercised only by mock-streamed text
    in tests)

What lands in a follow-up after the model file is published:
  - real Qwen 3B inference, benchmark on RK3588 NPU
  - tool-call grammar tightening based on observed model output
  - any required prompt-template fixes specific to RKLLM's wrap of Qwen
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


logger = logging.getLogger("blox-ai.rkllm")


# Paths matching the parent plan's plugin docker-compose volume mounts.
DEFAULT_SO_PATH = "/lib/librkllmrt.so"
DEFAULT_FALLBACK_SO_PATH = "/app/vendor/rkllm/librkllmrt.so"
DEFAULT_MODEL_DIR = "/uniondrive/blox-ai/model"
DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-rk3588-w8a8.rkllm"


# ---------------------------------------------------------------------------
# ctypes structures — mirror loyal-agent/app.py + Rockchip RKLLM headers
# ---------------------------------------------------------------------------

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 112),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


# Subset of the structures needed for a minimal generate loop. Full
# definitions are inherited from loyal-agent if/when LoRA / prompt-cache
# features land in a later phase.


# ---------------------------------------------------------------------------
# Locate dependencies
# ---------------------------------------------------------------------------

def find_so_path() -> Optional[str]:
    """Return the path to librkllmrt.so on this host, or None if not present.
    Checks the vendored fallback path second so dev hosts that have it
    in /lib still take priority."""
    for p in (DEFAULT_SO_PATH, DEFAULT_FALLBACK_SO_PATH):
        if os.path.isfile(p):
            return p
    return None


def find_model_path() -> Optional[str]:
    """Return the path to the model file, or None. Honors
    BLOX_AI_MODEL_PATH env var so the operator can pin a specific file."""
    env = os.environ.get("BLOX_AI_MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    default = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME)
    if os.path.isfile(default):
        return default
    # Tolerate a different model_version in the dir (e.g. post-rollback)
    # by globbing for any .rkllm file.
    try:
        for f in os.listdir(DEFAULT_MODEL_DIR):
            if f.endswith(".rkllm"):
                return os.path.join(DEFAULT_MODEL_DIR, f)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# RKLLMRuntime — minimal ctypes wrapper
# ---------------------------------------------------------------------------

class RKLLMLoadError(RuntimeError):
    pass


@dataclass
class RKLLMRuntime:
    """Wraps librkllmrt.so. NOT thread-safe — RKLLM holds NPU state.
    Serialize generate() calls externally."""

    so_path: str
    model_path: str
    _lib: ctypes.CDLL = field(init=False, repr=False)
    _handle: ctypes.c_void_p = field(init=False, default_factory=lambda: ctypes.c_void_p())

    def __post_init__(self):
        try:
            self._lib = ctypes.CDLL(self.so_path)
        except OSError as e:
            raise RKLLMLoadError(f"could not load {self.so_path}: {e}") from e
        self._wire_symbols()

    def _wire_symbols(self) -> None:
        # rkllm_init(handle*, param*, callback)
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKLLMParam),
            ctypes.c_void_p,
        ]
        # rkllm_destroy(handle)
        self._lib.rkllm_destroy.restype = ctypes.c_int
        self._lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

    def init_model(
        self,
        max_context_len: int = 4096,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> None:
        """Initialise the NPU with the model file. Raises RKLLMLoadError
        on any failure (which the caller maps to falling back to
        MockBackend)."""
        p = RKLLMParam()
        p.model_path = self.model_path.encode("utf-8")
        p.max_context_len = max_context_len
        p.max_new_tokens = max_new_tokens
        p.top_k = top_k
        p.top_p = top_p
        p.temperature = temperature
        p.repeat_penalty = 1.1
        p.frequency_penalty = 0.0
        p.presence_penalty = 0.0
        p.skip_special_token = True
        p.is_async = False
        p.img_start = b""
        p.img_end = b""
        p.img_content = b""
        p.extend_param.base_domain_id = 0

        # The callback parameter is `c_void_p`; passing NULL skips the
        # per-token callback. We'll stream tokens via a different surface
        # in a follow-up. For C7 the focus is "can we load + clean up
        # the model successfully" — actual inference depends on a real
        # Qwen .rkllm file being present.
        rc = self._lib.rkllm_init(
            ctypes.byref(self._handle),
            ctypes.byref(p),
            None,  # callback
        )
        if rc != 0:
            raise RKLLMLoadError(f"rkllm_init returned {rc}")

    def destroy(self) -> None:
        if self._handle.value:
            try:
                self._lib.rkllm_destroy(self._handle)
            except Exception as e:  # noqa: BLE001
                logger.warning("rkllm_destroy raised: %s", e)
            self._handle = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# RKLLMBackend — async-iterator adapter matching MockBackend's interface
# ---------------------------------------------------------------------------

# Mock approval token (real HMAC tokens come from C4's executor).
_MOCK_APPROVAL_TOKEN = "a" * 64


# Loose regex to find the Qwen native function-calling JSON in model
# output. Qwen 2.5 emits tool calls wrapped in <tool_call>...</tool_call>
# blocks containing {"name": "...", "arguments": {...}}. Real-model
# tightening lands when we have an actual model to observe.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


@dataclass
class RKLLMBackend:
    """Production backend. Wraps an RKLLMRuntime and shapes its token
    output into the SSE event sequence the bridge expects.

    For C7 this scaffolds the integration but does NOT exercise real
    inference — that needs the Qwen .rkllm file on the NPU, which is
    an out-of-session operational dependency. Tests mock the runtime
    entirely; lab smoke just verifies the wrapper can be instantiated
    when the .so + model are both present."""

    name: str = "rkllm"
    loaded: bool = False
    runbook_version: int = 0
    _runtime: Optional[RKLLMRuntime] = None

    def status_snapshot(self) -> dict:
        return {
            "model_loaded": self.loaded,
            "model_backend": self.name,
            "runbook_version": self.runbook_version,
            "active_sessions": 0,
            "npu_health": "ok" if self.loaded else "uninitialised",
            "last_error": None,
        }

    async def run_troubleshoot(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """C7 placeholder: emits a session_started + a single thought
        explaining that real inference needs the Qwen model file +
        toolkit. Real implementation lands in a follow-up.

        Same interface as MockBackend.run_troubleshoot so the bridge
        works against either."""
        sid = session_id or str(uuid.uuid4())
        yield {
            "type": "session_started",
            "session_id": sid,
            "protocol_version": 3,
            "ttl_seconds": 1800,
        }
        yield {
            "type": "thought",
            "payload": (
                "RKLLMBackend loaded. Real Qwen 2.5 3B inference path "
                "is a follow-up: the .rkllm model file at "
                "/uniondrive/blox-ai/model/ + native tool-call grammar "
                "parsing land once the model is published to the CDN."
            ),
        }
        yield {
            "type": "verdict",
            "payload": {
                "summary": "RKLLM backend reached but no model inferenced yet.",
                "severity": "yellow",
                "root_cause": "model_not_yet_published",
            },
        }


# ---------------------------------------------------------------------------
# Try-load entrypoint used by app.py lifespan
# ---------------------------------------------------------------------------

def try_load(model_path_override: Optional[str] = None) -> Optional[RKLLMBackend]:
    """Attempt to load the real RKLLM stack. Returns RKLLMBackend on
    success, None on any failure (so the lifespan falls back to
    MockBackend cleanly).

    Failure modes that map to None:
      - librkllmrt.so not on the host
      - model file not found
      - ctypes.CDLL load error (wrong arch, missing symbols, etc.)
      - rkllm_init returns non-zero (NPU not available, model format mismatch)
    """
    so_path = find_so_path()
    if so_path is None:
        logger.info("RKLLM .so not found; MockBackend will stay wired")
        return None
    model_path = model_path_override or find_model_path()
    if model_path is None:
        logger.info("RKLLM model file not found at %s; MockBackend stays wired",
                    DEFAULT_MODEL_DIR)
        return None
    try:
        runtime = RKLLMRuntime(so_path=so_path, model_path=model_path)
        runtime.init_model()
    except (RKLLMLoadError, OSError) as e:
        logger.warning("RKLLM init failed: %s; MockBackend stays wired", e)
        return None
    backend = RKLLMBackend(loaded=True, _runtime=runtime)
    logger.info("RKLLMBackend loaded (so=%s, model=%s)", so_path, model_path)
    return backend


# ---------------------------------------------------------------------------
# Token-stream → SSE event conversion (for the real implementation
# follow-up). Kept as a pure helper here so future-us can iterate on it
# without touching the bridge.
# ---------------------------------------------------------------------------

def parse_tool_calls(raw_text: str) -> list[dict]:
    """Find Qwen-style <tool_call>{...}</tool_call> blocks. Returns list
    of {tool, args}. Defensive — malformed JSON inside a block is
    skipped (the model will see no tool_result and can re-emit)."""
    out = []
    for m in _TOOL_CALL_RE.finditer(raw_text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            out.append({
                "tool": str(obj["name"]),
                "args": obj["arguments"] if isinstance(obj["arguments"], dict) else {},
            })
    return out
