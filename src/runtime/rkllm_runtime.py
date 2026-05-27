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

Chat template: Qwen 2.5 + Qwen 3 share the same ChatML envelope
(<|im_start|>{role}\\n{content}<|im_end|>). Qwen 3 adds hybrid
thinking mode — the assistant turn is prefixed with `<think>\\n`,
the model emits chain-of-thought, then `</think>`, then the
structured answer. We detect Qwen 3 from the model filename and:
  1. Inject the `<think>\\n` prefix into the assistant turn so the
     model always reasons before answering (highest-intelligence mode).
  2. Strip the `<think>...</think>` content from history BEFORE the
     next turn — keeps KV cache bounded as the tool-call loop runs.
     This matches the Qwen 3 model-card guidance: "historical model
     output should only include the final output, not the thinking".
  3. Keep the think content in the SSE `thought` event so the user
     sees the model's reasoning live (frontend can collapse the
     bubble if it gets verbose).

Tool-call grammar (Qwen-family-agnostic): tool calls wrapped in
<tool_call>{"name":"...","arguments":{...}}</tool_call>; results
appended back as <tool_response>{...}</tool_response>. Runbook +
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
DEFAULT_MODEL_FILENAME = "qwen3-1.7b-rk3588-w8a8.rkllm"


# ---------------------------------------------------------------------------
# ctypes structures — RKLLM v1.2.3 ABI
# ---------------------------------------------------------------------------
#
# These mirror the C structs in rkllm-runtime/Linux/librkllm_api/include/rkllm.h
# at the release-v1.2.3 tag. The Rockchip runtime is strict about struct
# layout — any drift here causes silent corruption OR explicit init errors
# ("The n_batch must be between 1 and 100, but got 0" is the canary).
#
# v1.1.4 → v1.2.3 ABI changes captured here:
#   RKLLMExtendParam: added embed_flash, enabled_cpus_num, enabled_cpus_mask,
#                     n_batch, use_cross_attn; reserved shrunk 112 → 104
#   RKLLMParam:       added n_keep between top_k and top_p
#   RKLLMInput:       restructured — role + enable_thinking + input_type
#                     prefixed BEFORE the union (previously just input_mode)
#   RKLLMInferParam:  added keep_history
#   RKLLMResult:      added token_id + logits + perf fields
#   Callback:         returns int now (was void)

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),  # NEW in v1.2.3
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


class RKLLMMultiModalInput(ctypes.Structure):
    """v1.2.3 added image_width, image_height + n_image fields."""
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    """v1.2.3 restructured: role + enable_thinking + input_type now
    prefix the union. Previously: just input_mode (int)."""
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),  # RKLLMInputType enum
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
        ("keep_history", ctypes.c_int),  # NEW in v1.2.3
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    """v1.2.3 added token_id, logits, perf fields. Drop the legacy
    `size` field (no longer in C struct)."""
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
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


# C callback signature — v1.2.3 returns int (was void in v1.1.4).
# The callback MUST return 0 to indicate normal continuation.
_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
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
    # generate() caller drains. v1.2.3 callback MUST return 0 (was void
    # in v1.1.4).
    def _on_token(self, result_ptr, userdata, state):
        try:
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
        except Exception:  # noqa: BLE001
            pass
        return 0

    def init_model(
        self,
        # 4096 matches the limit baked into the converted .rkllm by the
        # rkllm-toolkit's default conversion (we didn't pass a custom
        # max_context_len at build time). Asking for more here causes
        # the runtime to reject init with
        #   "max_context[N] must be less than the model's
        #    max_context_limit[4096]"
        # If a future conversion bumps the model's limit, raise this
        # value in lock-step.
        max_context_len: int = 4096,
        # max_new_tokens raised to 3072 with the Qwen 3 1.7B + thinking-
        # mode swap (advisor catch). Thinking blocks empirically run
        # 500-1500 tokens; the structured response adds another 200-500.
        # The prior 2048 was tight enough to truncate mid-verdict on
        # hard prompts, which manifests as no </think> in the output
        # → strip_think returns empty → user sees nothing useful.
        max_new_tokens: int = 3072,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> None:
        p = RKLLMParam()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.model_path = self.model_path.encode("utf-8")
        p.max_context_len = max_context_len
        p.max_new_tokens = max_new_tokens
        p.top_k = top_k
        # n_keep: number of KV cache tokens to keep at the start when the
        # context window shifts. -1 = use runtime default (typically the
        # system-prompt portion). New required field in v1.2.3.
        p.n_keep = -1
        p.top_p = top_p
        p.temperature = temperature
        p.repeat_penalty = 1.1
        p.frequency_penalty = 0.0
        p.presence_penalty = 0.0
        p.mirostat = 0
        p.mirostat_tau = 5.0
        p.mirostat_eta = 0.1
        p.skip_special_token = True
        p.is_async = False
        p.img_start = b""
        p.img_end = b""
        p.img_content = b""
        # Extend params — v1.2.3 added required NPU configuration here.
        p.extend_param.base_domain_id = 0
        p.extend_param.embed_flash = 1  # embed from flash (lower RAM)
        p.extend_param.enabled_cpus_num = 4
        # RK3588 big cores are 4-7 (A76); little cores 0-3 (A55). Pin
        # inference to big cores for best per-token latency.
        p.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        # n_batch=1 — single-sample inference. v1.2.3 rejects 0 explicitly
        # ("The n_batch must be between 1 and 100, but got 0").
        p.extend_param.n_batch = 1
        p.extend_param.use_cross_attn = 0

        rc = self._lib.rkllm_init(
            ctypes.byref(self._handle),
            ctypes.byref(p),
            self._callback,
        )
        if rc != 0:
            raise RKLLMLoadError(f"rkllm_init returned {rc}")

        # IMPORTANT — DO NOT call rkllm_set_chat_template here.
        #
        # Lab-observed bug 2026-05-27: calling
        # rkllm_set_chat_template(handle, "", "", "") to "pass our
        # pre-formatted ChatML through verbatim" disabled the
        # runtime's built-in <|im_end|> stop-token handling. Without
        # that, the model generated forever (or until max_new_tokens),
        # producing fictitious continuations like
        #     Calling diag/summary...
        #     tool returned data
        #     user: My BloX is not earning...
        #     [model imagining the next user message]
        # because the model emitted <|im_end|> to end its turn but the
        # runtime never recognized it as a stop signal.
        #
        # The runtime's own warning was explicit:
        #   "Calling rkllm_set_chat_template will disable the internal
        #    automatic chat template parsing, including enable_thinking.
        #    Make sure your custom prompt is complete and valid."
        #
        # Without the override, the runtime applies the model's
        # built-in Qwen 3 chat template, which:
        #   - knows <|im_end|> is the per-turn stop token
        #   - handles role transitions (user/tool/assistant)
        #   - respects the enable_thinking flag on RKLLMInput
        # so role-based input via rkllm_run (role="user"/"tool" +
        # raw content) Just Works. See generate() below for the new
        # role-based contract.

    def generate(
        self,
        prompt: str,
        role: str = "user",
        enable_thinking: bool = False,
        keep_history: int = 0,
        timeout_s: float = 90.0,
    ) -> str:
        """Blocking. Run a single role-based inference. Returns the full
        decoded text from the runtime callback up to the next <|im_end|>.

        v1.2.3 contract:
          - role: "user" or "tool" (the model knows the rest)
          - enable_thinking: True wraps the assistant turn in <think>\\n
            via the runtime's built-in Qwen 3 chat template
          - keep_history: 0 = fresh KV cache (first turn of a session);
                          1 = append to prior KV cache (subsequent turns
                          in a multi-turn loop)

        DO NOT pass pre-formatted ChatML as `prompt` — the runtime applies
        its built-in template, so the prompt should be raw message content
        for the given role. For the FIRST turn of a session, we currently
        concatenate the system rules into the user content (see
        RKLLMBackend.run_troubleshoot) since v1.2.3's role enum doesn't
        document a "system" role and set_chat_template's system arg
        disables thinking-mode handling.
        """
        with self._gen_lock:
            # Drain any leftover events from prior runs
            while not self._token_queue.empty():
                try:
                    self._token_queue.get_nowait()
                except queue.Empty:
                    break

            inp = RKLLMInput()
            ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(inp))
            inp.role = role.encode("utf-8")
            inp.enable_thinking = enable_thinking
            inp.input_type = RKLLM_INPUT_PROMPT
            inp.input_data.prompt_input = prompt.encode("utf-8")
            infer = RKLLMInferParam()
            ctypes.memset(ctypes.byref(infer), 0, ctypes.sizeof(infer))
            infer.mode = RKLLM_INFER_GENERATE
            infer.keep_history = keep_history

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

11. **NEVER mention Kubernetes, kubelet, kube-proxy, kubectl, k8s, kustomize, helm, kubeadm, or any other Kubernetes component.** None of them exist on this device. This device runs Docker (not Kubernetes) and the stack is exactly these named containers + services, no others:
     - `ipfs_host` — kubo (the IPFS daemon; "kubo" is the rename of "go-ipfs", NOT short for "kubernetes")
     - `ipfs_cluster` — ipfs-cluster (the cluster orchestrator on top of kubo)
     - `fula_go` — go-fula libp2p bridge
     - `fula_pinning` — Fula's pinning service
     - `fula_gateway` — Fula's gateway
     - `fula_fxsupport` — the supervision container
     - `wireguard-support.service` — host systemd service (not a container) for the WireGuard tunnel
     - `fula.service`, `uniondrive.service` — host systemd services
     If `diag/containers` returns the above names, that is the COMPLETE list — there is no kubelet, no kube-proxy, no etcd, no apiserver. If the user reports "not earning", look at the actual diag/heartbeat + diag/relay + diag/wireguard signals you see in tool results, NOT at hypothetical Kubernetes components.

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


# Qwen 2.5 + Qwen 3 chat template tokens (identical ChatML envelope)
_QWEN_IM_START = "<|im_start|>"
_QWEN_IM_END = "<|im_end|>"

# Qwen 3 hybrid-thinking sentinel — literal text the model emits when
# thinking mode is on. Per Qwen 3 model card / tokenizer_config.json,
# `<think>` / `</think>` are NOT special tokens; they pass through
# verbatim even with skip_special_token=True. Verified at first lab
# inference (log the raw generate() output once on the new .rkllm to
# confirm — see comment on RKLLMBackend._strip_think_for_history).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _is_qwen3_model(model_path: Optional[str]) -> bool:
    """Detect Qwen 3 model from filename. Matches `qwen3` or `qwen-3`
    case-insensitive. Used to gate the thinking-mode wiring so devices
    that haven't yet downloaded the Qwen 3 file (and still have an old
    Qwen 2.5 cached) continue using the legacy non-thinking format.

    Filename-based rather than tokenizer-introspecting because RKLLM-
    quantized models don't expose tokenizer config the way HF models
    do — the .rkllm file is opaque tensors. Filename is the only signal
    we have at backend-construction time."""
    if not model_path:
        return False
    name = os.path.basename(model_path).lower()
    return "qwen3" in name or "qwen-3" in name


def _strip_think(text: str) -> str:
    """Drop Qwen 3 <think>...</think> reasoning from `text`.

    Caller MUST only invoke when thinking mode is on (output produced
    with the `<think>\\n` assistant prefix). The contract: the output
    starts INSIDE a think block (because the prefix already opened
    one), continues with chain-of-thought prose, hits `</think>`, then
    contains the structured answer. We return everything after the
    first `</think>`.

    Edge cases:
      - TRUNCATED (no `</think>` anywhere): the model hit
        max_new_tokens mid-thought. The whole output is internal
        reasoning with no usable structured content — return empty.
        Caller treats this as a prose-only / no-verdict turn and
        force-verdicts.
      - SELF-WRAPPED (model emits its own `<think>...</think>` pair
        AFTER the prefix closure, e.g. it changes mind mid-answer):
        defensively sub out any further pairs in the tail.
      - TRAILING-OPEN (model started a new `<think>` near the end and
        ran out of tokens): drop from the orphan open tag to end of
        string so it doesn't bleed into history.

    advisor-flagged: an earlier `rfind` variant would have dropped
    content between multiple closing tags. split("</think>", 1) does
    the right thing in a single pass."""
    if _THINK_CLOSE not in text:
        return ""
    text = text.split(_THINK_CLOSE, 1)[1]
    text = _THINK_RE.sub("", text)
    open_idx = text.find(_THINK_OPEN)
    if open_idx != -1:
        text = text[:open_idx]
    return text


def _build_chat_prompt(
    system: str,
    history: list[dict],
    enable_thinking: bool = False,
) -> str:
    """Format using Qwen 2.5 / Qwen 3 ChatML template. history is a
    list of {role: user|assistant|tool, content: str} dicts.

    enable_thinking=True (Qwen 3 path): inject the `<think>\\n` prefix
    into the assistant turn so the model starts inside the think block.
    Matches `apply_chat_template(enable_thinking=True)` from the
    Hugging Face tokenizer config — required for highest-intelligence
    mode on Qwen 3."""
    parts = [f"{_QWEN_IM_START}system\n{system}{_QWEN_IM_END}"]
    for msg in history:
        parts.append(f"{_QWEN_IM_START}{msg['role']}\n{msg['content']}{_QWEN_IM_END}")
    assistant_prefix = f"{_QWEN_IM_START}assistant\n"
    if enable_thinking:
        # Tokenizer template emits exactly `<think>\n` (with trailing
        # newline) immediately after the role marker's newline. Match
        # that byte-for-byte. If this drifts from the HF template by a
        # whitespace character the model is mildly confused but still
        # functional — verify on lab by inspecting one full prompt
        # before generate().
        assistant_prefix += f"{_THINK_OPEN}\n"
    parts.append(assistant_prefix)
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
    """Production backend: real Qwen 2.5 / Qwen 3 + tool-call loop.

    Construct via try_load() which injects the executor + signer hooks
    so the backend can run diag tools inline and mint real HMAC tokens
    for recommended_action events. The bridge in tool_call_loop.py
    detects `consumes_tool_results=True` and skips its own tool_call
    interception (the backend handles it end-to-end).

    Qwen 3 thinking mode is wired automatically based on the model's
    filename (see _is_qwen3_model). When ON:
      - assistant prefix gets `<think>\\n` injected (model thinks first)
      - SSE thought event receives the raw output WITH think content
        (UI transparency — frontend can collapse the bubble)
      - history entries strip <think>...</think> before next turn
        (KV cache stays bounded across the tool-call loop)
    """

    name: str = "rkllm"
    loaded: bool = False
    runbook_version: int = 0
    _runtime: Optional[RKLLMRuntime] = None
    # Wired by app.py lifespan; None during fallback paths
    _tool_executor: Optional[Callable[[str, dict], Awaitable[dict]]] = None
    _action_signer: Optional[Callable[[str], str]] = None
    _runbook_loader: Optional[Any] = None
    # Set by try_load() from the resolved model_path. Controls assistant-
    # prefix injection + per-turn history rewriting.
    _enable_thinking: bool = False

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

        # Per-turn role-based content for v1.2.3.
        # Turn 0: role="user", content = user's actual prompt only. The
        # SYSTEM prompt is configured ONCE via rkllm_set_chat_template
        # below so the runtime injects it in the proper <|im_start|>
        # system\n...<|im_end|> slot — the model sees it as
        # instructions, not as part of the user's request, and doesn't
        # regurgitate it back in its first thought event.
        # Turn 1+: role="tool" with the JSON tool response. The runtime
        # appends to the existing KV cache via keep_history=1.
        # Configure session-specific chat template — uses our system
        # prompt + Qwen 3 markers + `<think>\n` postfix to force
        # thinking-mode (the auto-thinking flag is disabled when
        # set_chat_template is called per runtime warning).
        try:
            self._runtime._lib.rkllm_set_chat_template.restype = ctypes.c_int
            self._runtime._lib.rkllm_set_chat_template.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ]
            system_wrapped = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            ).encode("utf-8")
            prefix = b"<|im_start|>user\n"
            # postfix closes user turn + opens assistant + forces think
            postfix = b"<|im_end|>\n<|im_start|>assistant\n<think>\n"
            self._runtime._lib.rkllm_set_chat_template(
                self._runtime._handle, system_wrapped, prefix, postfix,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("rkllm_set_chat_template failed: %s", e)

        next_role: str = "user"
        next_content: str = prompt
        next_keep_history: int = 0   # 0 on first turn; 1 thereafter

        for turn in range(MAX_TURNS):
            # Last-chance: at MAX_TURNS-1 without a verdict, send the
            # force-verdict directive as the next user message instead
            # of whatever was queued. The runtime sees this fresh user
            # message + the in-KV-cache history.
            if (
                not emitted_verdict
                and not force_verdict_attempted
                and turn == MAX_TURNS - 1
            ):
                next_role = "user"
                next_content = FORCE_VERDICT_DIRECTIVE
                next_keep_history = 1  # keep prior KV
                force_verdict_attempted = True

            try:
                # rkllm v1.2.3: runtime applies built-in Qwen 3 chat
                # template + handles <|im_end|> stop token. The role
                # routes the message to the right slot in the template.
                output = await loop.run_in_executor(
                    None,
                    lambda r=next_role, c=next_content, kh=next_keep_history: (
                        self._runtime.generate(
                            c,
                            role=r,
                            enable_thinking=self._enable_thinking,
                            keep_history=kh,
                            timeout_s=PER_TURN_TIMEOUT_S,
                        )
                    ),
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

            # After this point, any further generate() calls should keep
            # the existing KV cache. (Reset only happens on a fresh
            # run_troubleshoot call.)
            next_keep_history = 1

            # Qwen 3 thinking mode: the raw output starts INSIDE a <think>
            # block (the runtime's built-in template prepends `<think>\n`
            # to the assistant turn when enable_thinking=True). The
            # structured response (tool calls, verdict, recommendations)
            # only exists AFTER `</think>`. Pre-strip so the parsers
            # can't be tripped by stray XML mentions inside the model's
            # reasoning prose ("I should call <tool_call>").
            output_for_parsing = (
                _strip_think(output) if self._enable_thinking else output
            )

            # NOTE: history list is no longer rebuilt-per-turn — v1.2.3
            # maintains KV cache via keep_history=1 on subsequent
            # generate() calls. We keep the local `history` append for
            # parity with the legacy test surface (see existing tests
            # asserting message-count) but it's now informational only.
            history.append({"role": "assistant", "content": output_for_parsing})

            # Surface the model's POST-THINK prose as a thought event.
            # User preference (literal): hide <think> content from UI
            # too, not just from KV. We feed strip_blocks the already-
            # de-thinked text so chain-of-thought reasoning never reaches
            # the SSE stream. If the post-think prose is empty (turn
            # consisted only of structured blocks), emit a short synthetic
            # marker so the stream isn't silent on slow BLE transports.
            thought_text = strip_blocks(output_for_parsing)
            if thought_text:
                # SSE thought schema: minLength 1, maxLength 4000
                yield {"type": "thought", "payload": thought_text[:4000]}
            elif self._enable_thinking:
                # Qwen 3 turn was 100% structured output after <think>;
                # avoid a silent stretch by emitting a tiny marker.
                yield {"type": "thought", "payload": "Analyzing diagnostics..."}

            # Parse blocks from the post-think text so XML mentions inside
            # reasoning prose can't pollute the parse results.
            verdict = parse_verdict(output_for_parsing)
            recommendations = parse_recommendations(output_for_parsing)
            tool_calls = parse_tool_calls(output_for_parsing)

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

                # Queue the tool responses as the NEXT generate() call's
                # role="tool" content. The v1.2.3 runtime appends to the
                # existing KV cache (keep_history=1 set above) so the
                # model sees the prior assistant turn + this tool result.
                # Multiple tool responses concatenated with newlines —
                # the runtime templates the whole blob as one tool turn.
                next_role = "tool"
                next_content = "\n".join(tool_responses_for_context)
                # next_keep_history already set to 1 after first turn.
                history.append({
                    "role": "tool",
                    "content": next_content,
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
            # the force-verdict directive yet, send it as the next
            # user message + give the model one more chance with the
            # existing KV cache. Otherwise terminate with synthetic
            # verdict.
            if not tool_calls and not verdict and not recommendations:
                if not force_verdict_attempted and turn < MAX_TURNS - 1:
                    next_role = "user"
                    next_content = FORCE_VERDICT_DIRECTIVE
                    # keep_history=1 already set after first turn.
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
    # Qwen 3 thinking mode is filename-gated so devices that still have
    # an old Qwen 2.5 cached (Qwen 3 download not yet completed) continue
    # using the legacy non-thinking format. The download_model.sh cleanup
    # logic removes the stale 1.5B AFTER the new Qwen 3 SHA verifies, so
    # this detection flips on automatically once the new file lands.
    enable_thinking = _is_qwen3_model(model_path)
    backend = RKLLMBackend(
        loaded=True,
        _runtime=runtime,
        _enable_thinking=enable_thinking,
    )
    logger.info(
        "RKLLMBackend loaded (so=%s, model=%s, thinking=%s)",
        so_path, model_path, enable_thinking,
    )
    return backend
