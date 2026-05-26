"""RKLLM ctypes wrapper + production backend for RK3588 NPU inference.

Three layers:
  - RKLLMRuntime: ctypes wrapper around librkllmrt.so. Owns the handle,
    the C callback that fans tokens out to a thread-safe queue per
    in-flight inference, and the bare-metal `generate()` call that
    blocks until the model finishes.
  - RKLLMBackend: async-iterator adapter that translates one
    /troubleshoot request into the sequence of SSE event dicts the
    bridge expects. Drives the tool-call loop (model emits tool_call ->
    we run executor -> append tool_response to context -> model
    continues) up to MAX_TURNS.
  - try_load(): factory; returns None on any failure so app.py's
    lifespan falls back to MockBackend cleanly.

The chat template + tool-call grammar is Qwen 2.5's native
function-calling format. Qwen emits tool calls wrapped in
<tool_call>{"name":"...","arguments":{...}}</tool_call>; we append
results back as <tool_response>{...}</tool_response>. The runbook +
tool definitions come in via system prompt at session start.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import queue
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional


logger = logging.getLogger("blox-ai.rkllm")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_SO_PATH = "/lib/librkllmrt.so"
DEFAULT_FALLBACK_SO_PATH = "/app/vendor/rkllm/librkllmrt.so"
DEFAULT_MODEL_DIR = "/uniondrive/blox-ai/model"
DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-rk3588-w8a8.rkllm"


# ---------------------------------------------------------------------------
# ctypes structures
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


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ]


# Enum-ish constants matching the C header
RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3


# ---------------------------------------------------------------------------
# Helpers — locate libraries + model
# ---------------------------------------------------------------------------

def find_so_path() -> Optional[str]:
    for p in (DEFAULT_SO_PATH, DEFAULT_FALLBACK_SO_PATH):
        if os.path.isfile(p):
            return p
    return None


def find_model_path() -> Optional[str]:
    env = os.environ.get("BLOX_AI_MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    default = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME)
    if os.path.isfile(default):
        return default
    try:
        for f in os.listdir(DEFAULT_MODEL_DIR):
            if f.endswith(".rkllm"):
                return os.path.join(DEFAULT_MODEL_DIR, f)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# RKLLMRuntime — ctypes wrapper with streaming callback
# ---------------------------------------------------------------------------

class RKLLMLoadError(RuntimeError):
    pass


# C callback signature
_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)


@dataclass
class RKLLMRuntime:
    """Wraps librkllmrt.so. Holds the C callback that pushes tokens into
    `self._token_queue` (cleared per generate() call). NOT thread-safe —
    one generate() at a time per instance."""

    so_path: str
    model_path: str
    _lib: ctypes.CDLL = field(init=False, repr=False)
    _handle: ctypes.c_void_p = field(init=False,
                                     default_factory=lambda: ctypes.c_void_p())
    # Callback machinery
    _token_queue: queue.Queue = field(init=False,
                                      default_factory=lambda: queue.Queue())
    _callback: Optional[Any] = field(init=False, default=None, repr=False)
    # Serialize generate() calls (rkllm handle is single-tenant)
    _gen_lock: threading.Lock = field(init=False,
                                      default_factory=threading.Lock)

    def __post_init__(self):
        try:
            self._lib = ctypes.CDLL(self.so_path)
        except OSError as e:
            raise RKLLMLoadError(f"could not load {self.so_path}: {e}") from e
        self._wire_symbols()
        # Pin the wrapped callback so the C side doesn't see a freed pointer.
        self._callback = _CALLBACK_TYPE(self._on_token)

    def _wire_symbols(self) -> None:
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKLLMParam),
            _CALLBACK_TYPE,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int
        self._lib.rkllm_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self._lib.rkllm_destroy.restype = ctypes.c_int
        self._lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

    # Callback runs on a C-spawned thread; ctypes acquires the GIL for us
    # before invoking. Push the token text + state to the queue; the
    # generate() caller drains.
    def _on_token(self, result_ptr, userdata, state):
        if state == RKLLM_RUN_NORMAL or state == RKLLM_RUN_WAITING:
            try:
                text_bytes = result_ptr.contents.text
                if text_bytes:
                    self._token_queue.put(("token", text_bytes))
            except Exception:  # noqa: BLE001
                # Never raise from the C callback - would corrupt rkllm state
                pass
        elif state == RKLLM_RUN_FINISH:
            self._token_queue.put(("finish", None))
        elif state == RKLLM_RUN_ERROR:
            self._token_queue.put(("error", None))

    def init_model(
        self,
        max_context_len: int = 8192,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> None:
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

        rc = self._lib.rkllm_init(
            ctypes.byref(self._handle),
            ctypes.byref(p),
            self._callback,
        )
        if rc != 0:
            raise RKLLMLoadError(f"rkllm_init returned {rc}")

    def generate(self, prompt: str, timeout_s: float = 90.0) -> str:
        """Blocking. Run inference for `prompt`, drain the callback queue,
        return the full decoded text. Raises RKLLMLoadError on rkllm_run
        failure or callback-reported error."""
        with self._gen_lock:
            # Drain any leftover events from prior runs
            while not self._token_queue.empty():
                try:
                    self._token_queue.get_nowait()
                except queue.Empty:
                    break

            inp = RKLLMInput()
            inp.input_mode = RKLLM_INPUT_PROMPT
            inp.input_data.prompt_input = prompt.encode("utf-8")
            infer = RKLLMInferParam()
            ctypes.memset(ctypes.byref(infer), 0, ctypes.sizeof(infer))
            infer.mode = RKLLM_INFER_GENERATE

            # rkllm_run blocks (is_async=False). The callback fires inline
            # for each token. When the model finishes, the callback is
            # invoked with RKLLM_RUN_FINISH, and rkllm_run returns.
            rc = self._lib.rkllm_run(
                self._handle,
                ctypes.byref(inp),
                ctypes.byref(infer),
                None,
            )
            if rc != 0:
                raise RKLLMLoadError(f"rkllm_run returned {rc}")

            # Drain the queue.
            buf: list[bytes] = []
            error = False
            while True:
                try:
                    kind, payload = self._token_queue.get(timeout=timeout_s)
                except queue.Empty:
                    raise RKLLMLoadError(
                        f"rkllm_run produced no FINISH within {timeout_s}s"
                    )
                if kind == "token":
                    if payload:
                        buf.append(payload)
                elif kind == "finish":
                    break
                elif kind == "error":
                    error = True
                    break
            if error:
                raise RKLLMLoadError("rkllm callback signalled RUN_ERROR")

            raw = b"".join(buf)
            return raw.decode("utf-8", errors="replace")

    def destroy(self) -> None:
        if self._handle.value:
            try:
                self._lib.rkllm_destroy(self._handle)
            except Exception as e:  # noqa: BLE001
                logger.warning("rkllm_destroy raised: %s", e)
            self._handle = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# Qwen 2.5 prompt template + tool-call grammar
# ---------------------------------------------------------------------------

# All diag/* tool definitions surfaced to the model. Keep aligned with
# the closed enum in sse_events.schema.json's tool_call payload.tool.
TOOL_DEFINITIONS = [
    {"name": "diag/summary",    "description": "Run all read-only diagnostics in parallel; returns overall severity + per-subsystem status."},
    {"name": "diag/internet",   "description": "Check DNS + HTTPS reachability to Google + discovery.fula.network."},
    {"name": "diag/relay",      "description": "List libp2p relay peers + circuit reservation count."},
    {"name": "diag/time",       "description": "Check NTP sync + clock offset."},
    {"name": "diag/power",      "description": "RK3588 undervoltage events, recent reboots, temp, uptime."},
    {"name": "diag/storage",    "description": "df + ext4 errors + dmesg I/O errors + smartctl health."},
    {"name": "diag/containers", "description": "docker ps + OOMKilled + restart counts for the fula stack."},
    {"name": "diag/wireguard",  "description": "WG handshake age + transfer counters + status triplet."},
    {"name": "diag/heartbeat",  "description": "Last heartbeat attempt to discovery.fula.network."},
    {"name": "diag/events",     "description": "Tail of /var/log/fula/events.jsonl (recent supervision events)."},
    {"name": "diag/readiness",  "description": "journalctl -u fula-readiness-check.service -n 100."},
]


SYSTEM_PROMPT_TEMPLATE = """You are Blox AI, an on-device troubleshooting assistant for a Fula Blox edge device (RK3588 hardware).

# OUTPUT FORMAT — STRICT

You communicate ONLY through XML-tagged blocks. NEVER use markdown code fences. NEVER use triple-backtick json. Each block on its own line.

The three allowed blocks:

  <tool_call>{{"name":"diag/<tool>","arguments":{{}}}}</tool_call>
  <recommendation>{{"action_name":"<name>","args":{{...}},"reasoning":"<why>","confidence":<0-1>,"tier":<2 or 3>}}</recommendation>
  <verdict>{{"summary":"<one sentence>","severity":"<green|yellow|red>","root_cause":"<short>"}}</verdict>

# HARD RULES

1. EVERY conversation MUST end with exactly ONE <verdict>.
2. After you have called the diagnostic tools you need (typically 1-3 calls), you MUST emit a <verdict> based on the results.
3. NEVER call tools indefinitely. After 1-2 follow-up calls, finalize.
4. NEVER output a turn that is prose-only with no <tool_call> AND no <verdict>. Every turn must contain at least one XML block.
5. **ANY action you suggest MUST be emitted as a <recommendation> XML block — NEVER as a markdown numbered list, bullet, or table.** If the user can't tap "Approve" on it, it didn't happen. Prose-only suggestions are invisible to the app. Translate every fix you have in mind into one <recommendation> block per fix.
6. Read tool_response JSON FIELD BY FIELD. Do NOT confuse `internet.latency_ms_avg` with a clock offset. Do NOT call a subsystem "red" if its `status` field says "green". Quote the actual field name you're basing your conclusion on.
7. NEVER use markdown headings (###), markdown bold (**...**), or numbered lists like "1. ntp.resync — ...". Those render as plain text in the chat; only <recommendation> blocks produce an Approve button.
8. **If the user reports a symptom but the diagnostic data CONTRADICTS it, ASK before acting.** Specifically:
     - User says "device disconnected" / "not reachable" / "app can't see my blox" BUT `heartbeat.status` is "green" with `http_status: 200`. → Device IS reachable from the cloud (heartbeat is the canonical "I'm alive" signal posting to discovery.fula.network). The disconnect they see is almost certainly phone-side (app cache, NetInfo wrong, WiFi switched, captive portal). DO NOT recommend restart_fula. INSTEAD emit a <user_question> like: {{"question":"Your device is currently posting heartbeats successfully (heartbeat.status=green, http_status=200). The connection issue may be phone-side. What error message do you see, and is your phone on the same WiFi as your Blox?","options":["Same WiFi","Cellular","Different WiFi","Don't know"]}}
     - User says "slow" but `containers.status: green, oom_count: 0, storage.status: green`. → ASK what specifically is slow.
     - User says "not earning" but `relay.reservation_count > 0` AND `heartbeat.status: green`. → Device IS connected. ASK if they've actually joined a pool.
9. **NEVER emit a tier-2 or tier-3 destructive action (`restart_fula`, `docker.restart`, `systemctl.restart`, `wireguard.bounce`, `reset`) with confidence > 0.7 when severity is "yellow" or "green".** Yellow signals can be normal — relay=yellow on a LAN-only device is expected. Acting on yellow with high confidence creates self-fulfilling problems (the action briefly DISCONNECTS the device, "confirming" the false diagnosis). Confidence > 0.7 on these actions requires severity="red" AND a specific failing subsystem named in the reasoning.
10. `relay.reservation_count: 0` is NOT a problem on its own — it only matters if the user is trying to be reached from outside their LAN. `wireguard.active: false` is NOT a problem unless the user explicitly set up WG. Mention these only as "informational" in your verdict, never as the root cause unless other evidence points to them.

# BAD vs GOOD examples

❌ BAD — prose recommendations get NO Approve button, user can take no action:

  ### Tier 2 Actions:
  1. **ntp.resync** - Resync the clock.
  2. **docker.restart container=ipfs_host** - Restart the container.

✅ GOOD — each recommendation is its own XML block:

  <recommendation>{{"action_name":"ntp.resync","args":{{}},"reasoning":"Clock is unsynced.","confidence":0.85,"tier":2}}</recommendation>
  <recommendation>{{"action_name":"docker.restart","args":{{"container":"ipfs_host"}},"reasoning":"ipfs_host restart-looping.","confidence":0.75,"tier":2}}</recommendation>

❌ BAD — making up a field:

  "Time Status: Clock offset is significant (93 ms)"
  (when tool_response actually said `time.status: green, synced: true` and the 93 was `internet.latency_ms_avg`)

✅ GOOD — quote what you actually read:

  "time.status is green (synced=true). internet.latency_ms_avg is 93ms — that's network latency, not a clock offset."

# AVAILABLE TOOLS (read-only)

{tool_list}

# RECOMMENDATION ACTION NAMES (only these are valid)

- docker.restart — args: container in {{ipfs_host, ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport}} (tier 2)
- systemctl.restart — args: unit in {{fula.service, uniondrive.service, wireguard-support.service}} (tier 2)
- wireguard.bounce — no args (tier 2)
- ntp.resync — no args (tier 2)
- restart_fula — no args (tier 2)
- reset — no args (tier 3, destructive — only after everything else)

# FULL EXAMPLE — three-turn flow

Turn 1 (user said "device feels slow"):
I'll start with a system summary.
<tool_call>{{"name":"diag/summary","arguments":{{}}}}</tool_call>

Turn 2 (after <tool_response> showing overall=red, internet=red, time=red):
Internet and time are both red. Drilling in on internet.
<tool_call>{{"name":"diag/internet","arguments":{{}}}}</tool_call>

Turn 3 (after <tool_response> showing dns_ok=true, https_discovery_ok=false):
Discovery is unreachable; the clock being unsynced makes it worse. Re-sync NTP first.
<verdict>{{"summary":"Discovery unreachable + clock unsynced.","severity":"red","root_cause":"discovery_https_unreachable"}}</verdict>
<recommendation>{{"action_name":"ntp.resync","args":{{}},"reasoning":"Many discovery checks rely on accurate timestamps; resync first.","confidence":0.8,"tier":2}}</recommendation>

# RUNBOOK EXCERPTS

{runbook_excerpt}

Be terse. Start with diag/summary unless the user named a specific symptom. Two or three tool calls, then finalize with a <verdict>."""


# Directive injected as a user message when the model has stalled
# (prose-only turn) OR at MAX_TURNS-1 without a verdict.
FORCE_VERDICT_DIRECTIVE = (
    "You have gathered enough information. STOP calling tools. Based on "
    "the diagnostic results above, emit a <verdict> block NOW. If you "
    "have a fix to recommend, also emit a <recommendation> block. Do "
    "not emit any more <tool_call> blocks. Reply with the <verdict> "
    "(and optional <recommendation>) only."
)


def _build_system_prompt(runbook_text: str = "", max_runbook_chars: int = 2000) -> str:
    tool_list = "\n".join(f"  - {t['name']}: {t['description']}" for t in TOOL_DEFINITIONS)
    excerpt = (runbook_text or "(no runbook loaded)")[:max_runbook_chars]
    return SYSTEM_PROMPT_TEMPLATE.format(tool_list=tool_list, runbook_excerpt=excerpt)


# Qwen 2.5 chat template tokens
_QWEN_IM_START = "<|im_start|>"
_QWEN_IM_END = "<|im_end|>"


def _build_chat_prompt(system: str, history: list[dict]) -> str:
    """Format using Qwen 2.5's ChatML template. history is a list of
    {role: user|assistant|tool, content: str} dicts."""
    parts = [f"{_QWEN_IM_START}system\n{system}{_QWEN_IM_END}"]
    for msg in history:
        parts.append(f"{_QWEN_IM_START}{msg['role']}\n{msg['content']}{_QWEN_IM_END}")
    parts.append(f"{_QWEN_IM_START}assistant\n")
    return "\n".join(parts)


# Grammar parsers — tolerant of unclosed blocks (Qwen 2.5 RKLLM-quantized
# sometimes omits the closing tag when it hits max_new_tokens or its own
# stop sequence). We match the opening tag + JSON body and allow either
# the proper closing tag OR end-of-string / next opening tag as terminator.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$|<tool_call>|<verdict>|<recommendation>)",
    re.DOTALL,
)
_VERDICT_RE = re.compile(
    r"<verdict>\s*(\{.*?\})\s*(?:</verdict>|$|<tool_call>|<verdict>|<recommendation>)",
    re.DOTALL,
)
_RECOMMENDATION_RE = re.compile(
    r"<recommendation>\s*(\{.*?\})\s*(?:</recommendation>|$|<tool_call>|<verdict>|<recommendation>)",
    re.DOTALL,
)


def _strip_partial_block(text: str, open_tag: str) -> str:
    """Cut from `<open_tag>` to end of string. Used by strip_blocks to
    remove unclosed blocks that the regex above matched but that don't
    have a proper closing tag."""
    idx = text.find(f"<{open_tag}>")
    if idx == -1:
        return text
    return text[:idx]


# Fallback: Qwen 3B sometimes ignores the <tool_call> wrapper and emits
# the JSON inside ```json``` markdown fences. Accept that too — same
# {name, arguments} keys identify tool-call intent.
_MARKDOWN_TOOL_CALL_RE = re.compile(
    r"```(?:json)?\s*(\{[^`]*?\"name\"[^`]*?\})\s*```",
    re.DOTALL,
)


def parse_tool_calls(raw_text: str) -> list[dict]:
    """Find tool-call blocks. Accepts both:
      1. <tool_call>{"name":"...","arguments":{...}}</tool_call>  (preferred)
      2. ```json {"name":"...","arguments":{...}} ``` (Qwen 3B sometimes)

    Returns list of {tool, args}. Defensive — malformed JSON skipped."""
    out = []
    seen_calls = set()
    candidates = list(_TOOL_CALL_RE.finditer(raw_text))
    candidates += list(_MARKDOWN_TOOL_CALL_RE.finditer(raw_text))
    for m in candidates:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        # Only count as tool-call if `name` matches the diag/* enum;
        # otherwise the markdown fence might be unrelated JSON.
        name = str(obj["name"])
        if not name.startswith("diag/"):
            continue
        # Dedup (same call_id appearing in both regex matches)
        args_val = obj.get("arguments", {})
        args_dict = args_val if isinstance(args_val, dict) else {}
        key = (name, json.dumps(args_dict, sort_keys=True))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        out.append({"tool": name, "args": args_dict})
    return out


def parse_verdict(raw_text: str) -> Optional[dict]:
    m = _VERDICT_RE.search(raw_text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    summary = obj.get("summary")
    severity = obj.get("severity")
    if not isinstance(summary, str) or severity not in ("green", "yellow", "red"):
        return None
    out = {"summary": summary[:500], "severity": severity}
    if isinstance(obj.get("root_cause"), str):
        out["root_cause"] = obj["root_cause"][:200]
    return out


def parse_recommendations(raw_text: str) -> list[dict]:
    out = []
    for m in _RECOMMENDATION_RE.finditer(raw_text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if not isinstance(obj.get("action_name"), str):
            continue
        tier = obj.get("tier")
        if tier not in (2, 3):
            continue
        confidence = obj.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        out.append({
            "action_name": obj["action_name"][:64],
            "args": obj.get("args") if isinstance(obj.get("args"), dict) else {},
            "reasoning": str(obj.get("reasoning", ""))[:1000] or "(no reasoning)",
            "confidence": confidence,
            "tier": tier,
        })
    return out


# Tier-2/3 destructive actions that should NOT be recommended at high
# confidence unless severity is "red". Empirically observed (2026-05-26
# lab): 1.5B Qwen pattern-matches "user said disconnected → relay
# yellow → restart_fula at 95%" even when heartbeat is green
# (contradicting evidence of actual reachability). The post-processor
# below caps confidence to 0.6 on these names when the verdict's
# severity is not red, AND additionally suppresses them entirely when
# heartbeat is green but the user's prompt mentioned a connectivity
# complaint (the AI is acting on false-positive contradictory data).
RESTART_CLASS_ACTIONS = frozenset({
    "restart_fula", "reset", "wireguard.bounce",
    "docker.restart", "systemctl.restart",
})


def apply_recommendation_guardrails(
    recommendations: list[dict],
    verdict: Optional[dict],
    last_summary_payload: Optional[dict] = None,
    user_prompt: Optional[str] = None,
) -> tuple[list[dict], list[str]]:
    """Cap or drop misbehaving recommendations from the model output.

    Two rules (in order):

    1. CONFIDENCE CAP: restart-class action (restart_fula, reset,
       wireguard.bounce, docker.restart, systemctl.restart) with
       confidence > 0.7 AND verdict.severity != "red" → cap confidence
       at 0.6. Reasoning: yellow/green severity + high-confidence
       destructive action is the false-positive pattern.

    2. CONTRADICTION SUPPRESS: if the user's prompt mentions a
       CONNECTIVITY symptom ("disconnect", "not reachable", "can't see",
       "offline") BUT the last diag/summary's heartbeat.status is
       "green" with http_status 200, suppress restart_fula entirely.
       The device IS reachable from the cloud; restarting it would
       create the very disconnect the user is complaining about.

    Returns (filtered_recommendations, list_of_human_readable_reasons)
    so the caller can log/surface why something was dropped or capped.
    """
    notes: list[str] = []
    severity = (verdict or {}).get("severity") if verdict else None

    # Did the user actually complain about connectivity?
    prompt_lc = (user_prompt or "").lower()
    is_connectivity_complaint = any(
        kw in prompt_lc
        for kw in ("disconnect", "not reachable", "unreachable",
                   "can't see", "cannot see", "offline", "can't reach",
                   "cannot reach", "showing as disconnected", "shows offline")
    )
    # Heartbeat ground truth: device IS reachable iff posted recent 200.
    hb = (last_summary_payload or {}).get("subsystems", {}).get("heartbeat", {})
    heartbeat_green_with_200 = (
        hb.get("status") == "green"
        and hb.get("key_metrics", {}).get("http_status") == 200
    )

    out: list[dict] = []
    for rec in recommendations:
        name = rec.get("action_name", "")
        conf = float(rec.get("confidence", 0.5))

        # Rule 2: drop restart_fula on contradiction (most aggressive).
        if (
            name == "restart_fula"
            and is_connectivity_complaint
            and heartbeat_green_with_200
        ):
            notes.append(
                f"dropped action={name!r} (confidence {conf:.2f}): user reported "
                f"a connectivity issue but heartbeat.status=green http_status=200, "
                f"so device IS reachable — restart_fula would create a real "
                f"disconnect from a non-existent problem"
            )
            continue

        # Rule 1: cap restart-class confidence when severity is not red.
        if name in RESTART_CLASS_ACTIONS and conf > 0.7 and severity != "red":
            capped = 0.6
            notes.append(
                f"capped action={name!r} confidence from {conf:.2f} to "
                f"{capped:.2f}: severity={severity!r} is not red, so this "
                f"action should not be presented as high-confidence"
            )
            rec = {**rec, "confidence": capped}

        out.append(rec)
    return out, notes


def strip_blocks(raw_text: str) -> str:
    """Remove tool_call/verdict/recommendation blocks. What's left is
    the model's prose 'thought' content. Also strips any UNCLOSED
    block at the end (Qwen sometimes truncates mid-block at
    max_new_tokens; we don't want the partial block bleeding into
    the thought text)."""
    out = _TOOL_CALL_RE.sub("", raw_text)
    out = _VERDICT_RE.sub("", out)
    out = _RECOMMENDATION_RE.sub("", out)
    # Belt-and-suspenders: if any opening tag survived (parser missed it
    # somehow), cut the text at the first open tag.
    for tag in ("tool_call", "verdict", "recommendation"):
        out = _strip_partial_block(out, tag)
    return out.strip()


# ---------------------------------------------------------------------------
# RKLLMBackend — async-iterator with tool-call loop
# ---------------------------------------------------------------------------

MAX_TURNS = 8
PER_TURN_TIMEOUT_S = 90.0
ACTION_ID_FMT = "rk-{turn}-{idx}"


@dataclass
class RKLLMBackend:
    """Production backend: real Qwen 2.5 3B + tool-call loop.

    Construct via try_load() which injects the executor + signer hooks
    so the backend can run diag tools inline and mint real HMAC tokens
    for recommended_action events. The bridge in tool_call_loop.py
    detects `consumes_tool_results=True` and skips its own tool_call
    interception (the backend handles it end-to-end)."""

    name: str = "rkllm"
    loaded: bool = False
    runbook_version: int = 0
    _runtime: Optional[RKLLMRuntime] = None
    # Wired by app.py lifespan; None during fallback paths
    _tool_executor: Optional[Callable[[str, dict], Awaitable[dict]]] = None
    _action_signer: Optional[Callable[[str], str]] = None
    _runbook_loader: Optional[Any] = None

    # Tells the bridge: don't intercept tool_call events; we handle them.
    consumes_tool_results: bool = True

    def status_snapshot(self) -> dict:
        return {
            "model_loaded": self.loaded,
            "model_backend": self.name,
            "runbook_version": (
                self._runbook_loader.get_version()
                if self._runbook_loader is not None else self.runbook_version
            ),
            "active_sessions": 0,
            "npu_health": "ok" if self.loaded else "uninitialised",
            "last_error": None,
        }

    def wire_runtime_deps(
        self,
        tool_executor: Callable[[str, dict], Awaitable[dict]],
        action_signer: Callable[[str], str],
        runbook_loader: Optional[Any] = None,
    ) -> None:
        """Called by app.py lifespan AFTER the executor + signer exist.
        Cleanly separates 'model loaded' from 'app-level hooks wired'."""
        self._tool_executor = tool_executor
        self._action_signer = action_signer
        self._runbook_loader = runbook_loader

    async def run_troubleshoot(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        sid = session_id or str(uuid.uuid4())
        yield {
            "type": "session_started",
            "session_id": sid,
            "protocol_version": 3,
            "ttl_seconds": 1800,
        }

        if self._runtime is None:
            yield {
                "type": "error",
                "code": "RKLLM_NOT_LOADED",
                "message": "RKLLMBackend has no runtime instance",
                "recoverable": False,
            }
            return

        runbook_text = ""
        if self._runbook_loader is not None:
            try:
                runbook_text = self._runbook_loader.get_text()
            except Exception:  # noqa: BLE001
                runbook_text = ""
        system_prompt = _build_system_prompt(runbook_text=runbook_text)
        history: list[dict] = [{"role": "user", "content": prompt}]

        emitted_verdict = False
        force_verdict_attempted = False
        loop = asyncio.get_event_loop()
        # Last diag/summary result + the user's original prompt are needed
        # by the recommendation guardrails to detect false positives like
        # "user says disconnected but heartbeat is green → suppress restart_fula".
        last_summary_payload: Optional[dict] = None
        original_user_prompt = prompt

        for turn in range(MAX_TURNS):
            # Last-chance: at MAX_TURNS-1 without a verdict, inject the
            # force-verdict directive so the model knows this is its
            # final shot to finalize.
            if (
                not emitted_verdict
                and not force_verdict_attempted
                and turn == MAX_TURNS - 1
            ):
                history.append({"role": "user", "content": FORCE_VERDICT_DIRECTIVE})
                force_verdict_attempted = True

            full_prompt = _build_chat_prompt(system_prompt, history)

            try:
                output = await loop.run_in_executor(
                    None,
                    lambda: self._runtime.generate(full_prompt,
                                                    timeout_s=PER_TURN_TIMEOUT_S),
                )
            except RKLLMLoadError as e:
                yield {
                    "type": "error",
                    "code": "RKLLM_GENERATE_FAILED",
                    "message": str(e)[:500],
                    "recoverable": False,
                }
                return
            except asyncio.TimeoutError:
                yield {
                    "type": "error",
                    "code": "RKLLM_TIMEOUT",
                    "message": f"generate exceeded {PER_TURN_TIMEOUT_S}s",
                    "recoverable": False,
                }
                return

            # Track this turn's assistant output in conversation history.
            history.append({"role": "assistant", "content": output})

            # Surface the model's prose as a thought event
            thought_text = strip_blocks(output)
            if thought_text:
                # SSE thought schema: minLength 1, maxLength 4000
                yield {"type": "thought", "payload": thought_text[:4000]}

            # Parse blocks
            verdict = parse_verdict(output)
            recommendations = parse_recommendations(output)
            tool_calls = parse_tool_calls(output)

            # Run each tool call inline + feed result back as tool_response
            if tool_calls and self._tool_executor is not None:
                tool_responses_for_context: list[str] = []
                for i, tc in enumerate(tool_calls):
                    call_id = ACTION_ID_FMT.format(turn=turn, idx=i)
                    yield {
                        "type": "tool_call",
                        "call_id": call_id,
                        "payload": {"tool": tc["tool"], "args": tc["args"]},
                    }
                    try:
                        result = await self._tool_executor(tc["tool"], tc["args"])
                        ok = True
                    except Exception as e:  # noqa: BLE001
                        result = None
                        err_msg = str(e)[:2000]
                        ok = False

                    tr_event: dict
                    if ok:
                        tr_event = {
                            "type": "tool_result",
                            "call_id": call_id,
                            "ok": True,
                            "payload": result,
                        }
                        tool_responses_for_context.append(
                            f"<tool_response>{json.dumps({'name': tc['tool'], 'result': result}, separators=(',', ':'))}</tool_response>"
                        )
                    else:
                        tr_event = {
                            "type": "tool_result",
                            "call_id": call_id,
                            "ok": False,
                            "payload": None,
                            "error": err_msg,  # noqa: F821
                        }
                        tool_responses_for_context.append(
                            f"<tool_response>{json.dumps({'name': tc['tool'], 'error': err_msg}, separators=(',', ':'))}</tool_response>"  # noqa: F821
                        )
                    yield tr_event

                    # Capture the diag/summary result for the guardrail
                    # check below. Lets us detect "heartbeat green but
                    # user said disconnected" false-positive pattern.
                    if ok and tc["tool"] == "diag/summary" and isinstance(result, dict):
                        last_summary_payload = result

                # Add tool responses to history for the next turn
                history.append({
                    "role": "tool",
                    "content": "\n".join(tool_responses_for_context),
                })

            # Emit verdict (once)
            if verdict and not emitted_verdict:
                emitted_verdict = True
                yield {"type": "verdict", "payload": verdict}

            # Server-side guardrails on recommendations (caps + drops
            # the false-positive patterns the 1.5B model produces).
            # Logged to stderr so operators can see WHY a recommendation
            # was dropped/capped.
            if recommendations:
                recommendations, guardrail_notes = apply_recommendation_guardrails(
                    recommendations,
                    verdict=verdict,
                    last_summary_payload=last_summary_payload,
                    user_prompt=original_user_prompt,
                )
                for note in guardrail_notes:
                    logger.warning("recommendation guardrail: %s", note)

            # Emit recommendations with real HMAC tokens
            if recommendations and self._action_signer is not None:
                for i, rec in enumerate(recommendations):
                    action_id = f"rk-act-{turn}-{i}"
                    try:
                        token = self._action_signer(action_id)
                    except Exception:  # noqa: BLE001
                        token = "a" * 64  # falls back to a placeholder; executor will reject
                    yield {
                        "type": "recommended_action",
                        "action_id": action_id,
                        "action_name": rec["action_name"],
                        "args": rec["args"],
                        "reasoning": rec["reasoning"],
                        "confidence": rec["confidence"],
                        "tier": rec["tier"],
                        "approval_token": token,
                    }

            # End conditions: verdict emitted AND either recommendations or no more tool calls
            if emitted_verdict and (recommendations or not tool_calls):
                return

            # Prose-only turn (no tool_calls + no verdict + no
            # recommendations): the model is stuck. If we haven't tried
            # the force-verdict directive yet, inject it + give the
            # model one more chance. Otherwise terminate with synthetic
            # verdict.
            if not tool_calls and not verdict and not recommendations:
                if not force_verdict_attempted and turn < MAX_TURNS - 1:
                    history.append({
                        "role": "user",
                        "content": FORCE_VERDICT_DIRECTIVE,
                    })
                    force_verdict_attempted = True
                    continue
                if not emitted_verdict:
                    yield {
                        "type": "verdict",
                        "payload": {
                            "summary": "Model did not produce a structured verdict.",
                            "severity": "yellow",
                            "root_cause": "no_verdict_emitted",
                        },
                    }
                return

        # MAX_TURNS exhausted
        if not emitted_verdict:
            yield {
                "type": "verdict",
                "payload": {
                    "summary": f"Diagnosis exceeded {MAX_TURNS}-turn budget; "
                               "no verdict converged.",
                    "severity": "yellow",
                    "root_cause": "max_turns_exceeded",
                },
            }


# ---------------------------------------------------------------------------
# Try-load entrypoint
# ---------------------------------------------------------------------------

def try_load(model_path_override: Optional[str] = None) -> Optional[RKLLMBackend]:
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
