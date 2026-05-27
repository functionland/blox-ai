"""C7 — RKLLMRuntime + RKLLMBackend tests.

Mocks the ctypes layer entirely. Real load + inference path is verified
by a separate lab smoke (deferred until the Qwen model file is published).
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# find_so_path / find_model_path
# ---------------------------------------------------------------------------

def test_find_so_path_returns_none_when_neither_present(monkeypatch):
    from src.runtime import rkllm_runtime as mod
    with patch.object(os.path, "isfile", return_value=False):
        assert mod.find_so_path() is None


def test_find_so_path_returns_primary_when_present():
    from src.runtime import rkllm_runtime as mod
    def fake_isfile(p):
        return p == mod.DEFAULT_SO_PATH
    with patch.object(os.path, "isfile", side_effect=fake_isfile):
        assert mod.find_so_path() == mod.DEFAULT_SO_PATH


def test_find_so_path_falls_back_to_vendored():
    from src.runtime import rkllm_runtime as mod
    def fake_isfile(p):
        return p == mod.DEFAULT_FALLBACK_SO_PATH
    with patch.object(os.path, "isfile", side_effect=fake_isfile):
        assert mod.find_so_path() == mod.DEFAULT_FALLBACK_SO_PATH


def test_find_model_path_honors_env_var(tmp_path, monkeypatch):
    from src.runtime import rkllm_runtime as mod
    f = tmp_path / "custom.rkllm"
    f.write_bytes(b"\x00")
    monkeypatch.setenv("BLOX_AI_MODEL_PATH", str(f))
    assert mod.find_model_path() == str(f)


def test_find_model_path_returns_none_when_dir_missing(monkeypatch):
    from src.runtime import rkllm_runtime as mod
    monkeypatch.delenv("BLOX_AI_MODEL_PATH", raising=False)
    with patch.object(os.path, "isfile", return_value=False), \
         patch.object(os, "listdir", side_effect=OSError):
        assert mod.find_model_path() is None


def test_find_model_path_globs_any_rkllm(monkeypatch):
    from src.runtime import rkllm_runtime as mod
    monkeypatch.delenv("BLOX_AI_MODEL_PATH", raising=False)
    def fake_isfile(p):
        # Default-name model not present; some other .rkllm is
        return False
    with patch.object(os.path, "isfile", side_effect=fake_isfile), \
         patch.object(os, "listdir", return_value=["other-v1.rkllm", "readme.txt"]):
        result = mod.find_model_path()
    assert result is not None
    assert result.endswith("other-v1.rkllm")


# ---------------------------------------------------------------------------
# try_load fallback paths
# ---------------------------------------------------------------------------

def test_try_load_returns_none_when_so_missing():
    from src.runtime import rkllm_runtime as mod
    with patch.object(mod, "find_so_path", return_value=None):
        assert mod.try_load() is None


def test_try_load_returns_none_when_model_missing():
    from src.runtime import rkllm_runtime as mod
    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value=None):
        assert mod.try_load() is None


def test_try_load_returns_none_on_ctypes_load_error():
    from src.runtime import rkllm_runtime as mod
    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value="/fake/model.rkllm"), \
         patch("ctypes.CDLL", side_effect=OSError("bad arch")):
        assert mod.try_load() is None


def test_try_load_returns_none_on_init_failure():
    from src.runtime import rkllm_runtime as mod

    class FakeLib:
        def __getattr__(self, name):
            # Every symbol resolves but rkllm_init returns non-zero
            def fake_init(*a, **kw):
                if name == "rkllm_init":
                    return 99  # any non-zero
                return 0
            return MagicMock(side_effect=fake_init, restype=None, argtypes=None)

    fake_lib_instance = FakeLib()
    fake_lib_instance.rkllm_init = MagicMock(return_value=99)
    fake_lib_instance.rkllm_destroy = MagicMock(return_value=0)

    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value="/fake/model.rkllm"), \
         patch("ctypes.CDLL", return_value=fake_lib_instance):
        assert mod.try_load() is None


def test_try_load_returns_backend_on_success():
    from src.runtime import rkllm_runtime as mod
    fake_lib = MagicMock()
    fake_lib.rkllm_init = MagicMock(return_value=0)
    fake_lib.rkllm_destroy = MagicMock(return_value=0)
    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value="/fake/model.rkllm"), \
         patch("ctypes.CDLL", return_value=fake_lib):
        backend = mod.try_load()
    assert backend is not None
    assert backend.name == "rkllm"
    assert backend.loaded is True


# ---------------------------------------------------------------------------
# RKLLMBackend SSE shape
# ---------------------------------------------------------------------------

def test_backend_status_snapshot_matches_other_backends():
    from src.runtime.rkllm_runtime import RKLLMBackend
    snap = RKLLMBackend(loaded=True).status_snapshot()
    for field in ("model_loaded", "model_backend", "runbook_version",
                  "active_sessions", "npu_health", "last_error"):
        assert field in snap
    assert snap["npu_health"] == "ok"
    assert snap["model_backend"] == "rkllm"


def test_backend_status_snapshot_uninitialised_when_not_loaded():
    from src.runtime.rkllm_runtime import RKLLMBackend
    snap = RKLLMBackend(loaded=False).status_snapshot()
    assert snap["npu_health"] == "uninitialised"


def test_backend_run_troubleshoot_yields_session_started_first():
    """The minimal C7 RKLLMBackend yields the same session_started shape
    as MockBackend so the bridge doesn't care which is wired."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    async def collect():
        # Backend with no _runtime — should still emit session_started
        # then a clean RKLLM_NOT_LOADED error (not crash). The C7-final
        # path requires a real runtime to actually generate; the
        # session_started shape stays consistent so the bridge never
        # sees a malformed first event.
        b = RKLLMBackend()
        events = []
        async for ev in b.run_troubleshoot("test prompt", session_id=None):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    assert events[0]["type"] == "session_started"
    assert events[0]["protocol_version"] == 3
    # No runtime wired → next event is an error
    assert any(e["type"] == "error" and e.get("code") == "RKLLM_NOT_LOADED"
               for e in events)


# ---------------------------------------------------------------------------
# Tool-call grammar parser
# ---------------------------------------------------------------------------

def test_parse_tool_calls_extracts_qwen_format():
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = (
        "Thinking... I should check.\n"
        '<tool_call>{"name": "diag/summary", "arguments": {}}</tool_call>\n'
        "Now I'll check internet too.\n"
        '<tool_call>{"name": "diag/internet", "arguments": {"timeout_s": 5}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["tool"] == "diag/summary"
    assert calls[1]["tool"] == "diag/internet"
    assert calls[1]["args"] == {"timeout_s": 5}


def test_parse_tool_calls_skips_malformed_json():
    """Parser now requires name to start with 'diag/' (defense against
    Qwen's markdown-fence false positives matching unrelated JSON)."""
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = (
        '<tool_call>{"name": "diag/time", "arguments": {}}</tool_call>'
        '<tool_call>not even close to JSON</tool_call>'
        '<tool_call>{"name": "diag/power", "arguments": {}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert [c["tool"] for c in calls] == ["diag/time", "diag/power"]


def test_parse_tool_calls_accepts_markdown_fence():
    """Qwen 3B sometimes emits ```json {...} ``` instead of <tool_call>."""
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = '```json\n{"name":"diag/summary","arguments":{}}\n```'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "diag/summary"


def test_parse_tool_calls_ignores_non_diag_markdown_json():
    """Random JSON in markdown fences (e.g. tool_result echo) must NOT
    be picked up as a tool call. Diag-prefix gate enforces this."""
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = '```json\n{"name":"some_other_thing","arguments":{}}\n```'
    assert parse_tool_calls(text) == []


def test_parse_tool_calls_handles_empty_input():
    from src.runtime.rkllm_runtime import parse_tool_calls
    assert parse_tool_calls("") == []
    assert parse_tool_calls("just prose, no tool calls") == []


def test_parse_tool_calls_defaults_missing_args_to_empty_dict():
    """Qwen sometimes omits `arguments` for no-arg tools. Be tolerant —
    default to empty dict rather than dropping the call."""
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = '<tool_call>{"name": "diag/summary"}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "diag/summary"
    assert calls[0]["args"] == {}


def test_parse_tool_calls_drops_missing_name():
    from src.runtime.rkllm_runtime import parse_tool_calls
    assert parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>') == []


# ---------------------------------------------------------------------------
# parse_verdict / parse_recommendations / strip_blocks / prompt building
# ---------------------------------------------------------------------------

def test_parse_verdict_happy_path():
    from src.runtime.rkllm_runtime import parse_verdict
    text = ('Some thinking...\n'
            '<verdict>{"summary": "all green", "severity": "green", "root_cause": "noop"}</verdict>')
    v = parse_verdict(text)
    assert v == {"summary": "all green", "severity": "green", "root_cause": "noop"}


def test_parse_verdict_rejects_invalid_severity():
    from src.runtime.rkllm_runtime import parse_verdict
    text = '<verdict>{"summary": "x", "severity": "purple"}</verdict>'
    assert parse_verdict(text) is None


def test_parse_verdict_rejects_missing_summary():
    from src.runtime.rkllm_runtime import parse_verdict
    text = '<verdict>{"severity": "green"}</verdict>'
    assert parse_verdict(text) is None


def test_parse_recommendations_happy_path():
    from src.runtime.rkllm_runtime import parse_recommendations
    text = ('<recommendation>{"action_name":"docker.restart",'
            '"args":{"container":"ipfs_host"},"reasoning":"kubo wedged",'
            '"confidence":0.85,"tier":2}</recommendation>')
    recs = parse_recommendations(text)
    assert len(recs) == 1
    assert recs[0]["action_name"] == "docker.restart"
    assert recs[0]["args"] == {"container": "ipfs_host"}
    assert recs[0]["confidence"] == 0.85
    assert recs[0]["tier"] == 2


def test_parse_recommendations_clamps_confidence():
    from src.runtime.rkllm_runtime import parse_recommendations
    text = ('<recommendation>{"action_name":"x","reasoning":"y",'
            '"confidence":1.5,"tier":2}</recommendation>')
    recs = parse_recommendations(text)
    assert recs[0]["confidence"] == 1.0


def test_parse_recommendations_drops_invalid_tier():
    from src.runtime.rkllm_runtime import parse_recommendations
    text = ('<recommendation>{"action_name":"x","reasoning":"y",'
            '"tier":1}</recommendation>')
    assert parse_recommendations(text) == []


def test_strip_blocks_removes_all_block_types():
    from src.runtime.rkllm_runtime import strip_blocks
    text = ('I will check.\n'
            '<tool_call>{"name":"diag/summary"}</tool_call>\n'
            'Looks fine.\n'
            '<verdict>{"summary":"ok","severity":"green"}</verdict>\n'
            '<recommendation>{"action_name":"x","reasoning":"y","tier":2}</recommendation>')
    out = strip_blocks(text)
    assert "<tool_call>" not in out
    assert "<verdict>" not in out
    assert "<recommendation>" not in out
    assert "I will check." in out
    assert "Looks fine." in out


def test_build_system_prompt_includes_tool_list_and_runbook():
    from src.runtime.rkllm_runtime import _build_system_prompt
    p = _build_system_prompt(runbook_text="RUNBOOK_BODY_TEST", max_runbook_chars=100)
    assert "diag/summary" in p  # tool list
    assert "RUNBOOK_BODY_TEST" in p
    # Truncation honored
    p2 = _build_system_prompt(runbook_text="X" * 5000, max_runbook_chars=100)
    assert "X" * 100 in p2
    assert "X" * 101 not in p2


def test_build_chat_prompt_uses_qwen_template():
    from src.runtime.rkllm_runtime import _build_chat_prompt
    p = _build_chat_prompt(
        system="SYS",
        history=[{"role": "user", "content": "hello"}],
    )
    assert "<|im_start|>system" in p
    assert "SYS" in p
    assert "<|im_start|>user" in p
    assert "hello" in p
    assert p.endswith("<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
# End-to-end RKLLMBackend.run_troubleshoot with a mock runtime
# ---------------------------------------------------------------------------

def test_run_troubleshoot_end_to_end_with_mock_runtime():
    """Inject a fake runtime that returns a scripted model output
    sequence. Verify the full SSE event flow: session_started, thought,
    tool_call, tool_result, verdict, recommended_action."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    # Scripted turns: turn 0 emits tool_call + thought; turn 1 emits
    # verdict + recommendation.
    turn_outputs = [
        ('I will check the system.\n'
         '<tool_call>{"name":"diag/summary","arguments":{}}</tool_call>'),
        ('All systems appear nominal.\n'
         '<verdict>{"summary":"healthy","severity":"green",'
         '"root_cause":"no_issue"}</verdict>\n'
         '<recommendation>{"action_name":"docker.restart",'
         '"args":{"container":"ipfs_host"},"reasoning":"precautionary",'
         '"confidence":0.3,"tier":2}</recommendation>'),
    ]
    turn_idx = {"i": 0}

    class FakeRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            i = turn_idx["i"]
            turn_idx["i"] = i + 1
            if i < len(turn_outputs):
                return turn_outputs[i]
            return '<verdict>{"summary":"end","severity":"green"}</verdict>'

    async def fake_tool_executor(tool, args):
        return {"overall": "green", "subsystems": {"internet": {"status": "green"}}}

    def fake_signer(action_id):
        return "f" * 64  # valid-length placeholder

    backend = RKLLMBackend(loaded=True, _runtime=FakeRuntime())
    backend.wire_runtime_deps(
        tool_executor=fake_tool_executor,
        action_signer=fake_signer,
    )

    async def collect():
        events = []
        async for ev in backend.run_troubleshoot("device feels slow", session_id="sid-1"):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    types = [e["type"] for e in events]

    # First event session_started
    assert events[0]["type"] == "session_started"
    assert events[0]["session_id"] == "sid-1"

    # Sequence contains the expected event types
    assert "thought" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "verdict" in types
    assert "recommended_action" in types

    # tool_result.call_id matches tool_call.call_id
    tcs = [e for e in events if e["type"] == "tool_call"]
    trs = [e for e in events if e["type"] == "tool_result"]
    assert len(tcs) == 1 and len(trs) == 1
    assert tcs[0]["call_id"] == trs[0]["call_id"]
    assert trs[0]["ok"] is True
    assert trs[0]["payload"]["overall"] == "green"

    # verdict shape
    verdict = next(e for e in events if e["type"] == "verdict")
    assert verdict["payload"]["severity"] == "green"

    # recommended_action carries the signer's token + expected fields
    rec = next(e for e in events if e["type"] == "recommended_action")
    assert rec["action_name"] == "docker.restart"
    assert rec["approval_token"] == "f" * 64
    assert rec["tier"] == 2


def test_run_troubleshoot_with_no_runtime_yields_clean_error():
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    async def collect():
        events = []
        async for ev in RKLLMBackend().run_troubleshoot("x"):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    assert events[0]["type"] == "session_started"
    err = next((e for e in events if e["type"] == "error"), None)
    assert err is not None
    assert err["code"] == "RKLLM_NOT_LOADED"


# ---------------------------------------------------------------------------
# Recommendation guardrails — 2026-05-26 lab false-positive regression
# ---------------------------------------------------------------------------

def _verdict(severity: str):
    return {"summary": "x", "severity": severity, "root_cause": "y"}


def _rec(name: str, conf: float, tier: int = 2, args=None):
    return {
        "action_name": name,
        "args": args or {},
        "reasoning": "...",
        "confidence": conf,
        "tier": tier,
    }


def _summary_with_heartbeat(status: str, http_status: int):
    return {
        "overall": status,
        "subsystems": {
            "heartbeat": {"status": status,
                          "key_metrics": {"http_status": http_status}},
        },
    }


def test_guardrail_caps_restart_class_confidence_when_not_red():
    """1.5B Qwen pattern: 'restart_fula' at confidence 0.95 with verdict
    severity=yellow. Cap to 0.6 — yellow is not red, restart-class
    actions should not be presented as high-confidence."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("restart_fula", 0.95)]
    out, notes = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("yellow"),
        last_summary_payload=None,
        user_prompt="trying to upload a file but it's slow",
    )
    assert len(out) == 1
    assert out[0]["confidence"] == 0.6
    assert any("capped" in n.lower() for n in notes)


def test_guardrail_passes_high_confidence_when_severity_red():
    """If severity IS red, the action is appropriately confident."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("docker.restart", 0.9, args={"container": "ipfs_host"})]
    out, _ = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("red"),
        last_summary_payload=None,
        user_prompt="ipfs_host crashing",
    )
    assert len(out) == 1
    assert out[0]["confidence"] == 0.9


def test_guardrail_drops_restart_fula_when_heartbeat_green_and_user_said_disconnected():
    """The EXACT lab regression 2026-05-26: user clicked 'disconnected'
    scenario, device was healthy + heartbeat green http_status=200, but
    AI emitted restart_fula at 0.95 based on relay=yellow alone. That
    action would create the very disconnect being complained about.
    Must be dropped entirely."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("restart_fula", 0.95)]
    out, notes = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("yellow"),
        last_summary_payload=_summary_with_heartbeat("green", 200),
        user_prompt="My Blox is showing as disconnected in the app. Please diagnose why the device is not reachable",
    )
    assert out == [], (
        f"restart_fula should have been dropped (heartbeat green + connectivity "
        f"complaint = false positive); got {out}"
    )
    assert any("dropped" in n.lower() for n in notes)
    assert any("heartbeat" in n.lower() for n in notes)


def test_guardrail_passes_restart_fula_when_heartbeat_actually_red():
    """If heartbeat IS failing (real connectivity issue), restart_fula
    is legitimate even on user-reported disconnect."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("restart_fula", 0.8)]
    out, _ = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("red"),
        last_summary_payload={
            "subsystems": {
                "heartbeat": {"status": "red",
                              "key_metrics": {"http_status": 0}},
            },
        },
        user_prompt="disconnected",
    )
    assert len(out) == 1
    assert out[0]["confidence"] == 0.8


def test_guardrail_leaves_non_restart_class_actions_alone():
    """ntp.resync is NOT a restart-class action — confidence cap doesn't apply."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("ntp.resync", 0.95)]
    out, _ = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("yellow"),
        last_summary_payload=None,
        user_prompt="clock drift suspected",
    )
    assert out[0]["confidence"] == 0.95


def test_guardrail_does_not_drop_when_user_did_NOT_complain_about_connectivity():
    """If user prompt didn't mention disconnect/unreachable, the
    heartbeat-green check shouldn't fire — restart_fula stays (just
    capped if severity isn't red)."""
    from src.runtime.rkllm_runtime import apply_recommendation_guardrails
    recs = [_rec("restart_fula", 0.95)]
    out, _ = apply_recommendation_guardrails(
        recs,
        verdict=_verdict("yellow"),
        last_summary_payload=_summary_with_heartbeat("green", 200),
        user_prompt="my Blox is slow when streaming",
    )
    assert len(out) == 1, "non-connectivity complaint shouldn't trigger the drop"
    assert out[0]["confidence"] == 0.6  # but still capped


def test_run_troubleshoot_terminates_when_model_only_emits_prose():
    """If the model produces no verdict + no tool_calls, the backend
    INJECTS a force-verdict directive + gives the model one more chance.
    If that ALSO produces nothing, synthesize a verdict + terminate."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    class ProseOnlyRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            return "Just some thinking, nothing else."

    backend = RKLLMBackend(loaded=True, _runtime=ProseOnlyRuntime())
    backend.wire_runtime_deps(tool_executor=None, action_signer=lambda x: "f" * 64)

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    verdict = next(e for e in events if e["type"] == "verdict")
    assert verdict["payload"]["root_cause"] == "no_verdict_emitted"


def test_run_troubleshoot_force_verdict_directive_works():
    """When the model emits prose on turn 0 but produces a verdict on
    turn 1 (after the force-verdict directive is injected), the
    backend yields the REAL model verdict — NOT the synthetic
    fallback. Verifies the force-verdict path."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend, FORCE_VERDICT_DIRECTIVE

    turn_outputs = [
        "I am thinking about your question.",  # prose only -> triggers force-verdict
        '<verdict>{"summary":"all good after thought","severity":"green","root_cause":"thought_through"}</verdict>',
    ]
    turn_idx = {"i": 0}
    prompts_seen = []

    class StagedRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            prompts_seen.append(prompt)
            i = turn_idx["i"]
            turn_idx["i"] = i + 1
            return turn_outputs[i] if i < len(turn_outputs) else "(end)"

    backend = RKLLMBackend(loaded=True, _runtime=StagedRuntime())
    backend.wire_runtime_deps(tool_executor=None, action_signer=lambda x: "f" * 64)

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1
    # Real model verdict, NOT the synthetic fallback
    assert verdicts[0]["payload"]["root_cause"] == "thought_through"
    # The second prompt to the model contained the force-verdict directive
    assert any(FORCE_VERDICT_DIRECTIVE in p for p in prompts_seen)


# ---------------------------------------------------------------------------
# Qwen 3 thinking-mode swap (2026-05-26)
# ---------------------------------------------------------------------------

def test_default_model_filename_is_qwen3():
    """Sanity check: the in-container default points at the active Qwen 3
    file. Drives both find_model_path() fallback AND filename-based
    thinking-mode detection."""
    from src.runtime import rkllm_runtime as mod
    assert mod.DEFAULT_MODEL_FILENAME == "qwen3-1.7b-rk3588-w8a8.rkllm"


def test_is_qwen3_model_detects_canonical_filename():
    from src.runtime.rkllm_runtime import _is_qwen3_model
    assert _is_qwen3_model("/uniondrive/model/qwen3-1.7b-rk3588-w8a8.rkllm") is True
    # Hyphenated variant
    assert _is_qwen3_model("/uniondrive/model/qwen-3-1.7b-rk3588-w8a8.rkllm") is True
    # Case-insensitive
    assert _is_qwen3_model("/path/Qwen3-1.7B-rk3588.rkllm") is True


def test_is_qwen3_model_rejects_qwen_2_5():
    """Critical for rollout safety: devices that still have the prior
    Qwen 2.5 cached must NOT have thinking-mode enabled (the model
    doesn't support `<think>` tags, would emit junk if we prepend the
    prefix). The cleanup of stale 1.5B only happens AFTER the new
    Qwen 3 download verifies, so during the transition window both
    files coexist and find_model_path() may pick the 1.5B."""
    from src.runtime.rkllm_runtime import _is_qwen3_model
    assert _is_qwen3_model("/path/qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm") is False
    assert _is_qwen3_model("/path/qwen2.5-3b-instruct-rk3588-w8a8.rkllm") is False
    assert _is_qwen3_model("/path/deepseek-llm-7b-chat.rkllm") is False
    assert _is_qwen3_model(None) is False
    assert _is_qwen3_model("") is False


def test_strip_think_drops_full_block():
    """Standard shape: the assistant prefix injected `<think>\\n` so the
    raw output starts inside the think block. After the first `</think>`
    is the structured response."""
    from src.runtime.rkllm_runtime import _strip_think
    raw = (
        "Let me reason about this for a moment. The user said disconnected.\n"
        "First I should check heartbeat.</think>\n"
        '<tool_call>{"name":"diag/summary","arguments":{}}</tool_call>'
    )
    out = _strip_think(raw)
    assert "<tool_call>" in out
    assert "Let me reason about this" not in out
    assert "</think>" not in out


def test_strip_think_returns_empty_when_truncated_mid_think():
    """Model hit max_new_tokens mid-thought — no `</think>` ever
    emitted. Whole output is internal reasoning; nothing structured to
    surface. Caller (run_troubleshoot) treats this as a prose-only
    turn and force-verdicts on the next iteration."""
    from src.runtime.rkllm_runtime import _strip_think
    truncated = (
        "Let me think step by step. The user reports a slow connection.\n"
        "Possible causes include kubo, ipfs_cluster, or wireguard. I should"
    )
    assert _strip_think(truncated) == ""


def test_strip_think_handles_self_wrapped_pair_after_main_close():
    """Defensive: model emits its own `<think>X</think>` block in the
    middle of the structured response (e.g., changes its mind). Must
    sub out the inner pair too."""
    from src.runtime.rkllm_runtime import _strip_think
    raw = (
        "reasoning prose</think>\n"
        '<tool_call>{"name":"diag/internet","arguments":{}}</tool_call>\n'
        "<think>actually, let me also check time</think>\n"
        '<tool_call>{"name":"diag/time","arguments":{}}</tool_call>'
    )
    out = _strip_think(raw)
    assert "<think>" not in out
    assert "</think>" not in out
    assert "diag/internet" in out
    assert "diag/time" in out


def test_strip_think_drops_trailing_unclosed_open():
    """Model started a new think block near the end of its turn but ran
    out of tokens before closing. Drop from the orphan `<think>` to end
    of string so the partial reasoning doesn't bleed into history."""
    from src.runtime.rkllm_runtime import _strip_think
    raw = (
        "reasoning</think>\n"
        '<tool_call>{"name":"diag/summary","arguments":{}}</tool_call>\n'
        "<think>wait, let me also"
    )
    out = _strip_think(raw)
    assert "<tool_call>" in out
    assert "<think>" not in out
    assert "wait, let me also" not in out


def test_build_chat_prompt_enable_thinking_injects_think_prefix():
    """With enable_thinking=True the assistant prefix gets `<think>\\n`
    appended so the model starts inside the think block. This matches
    `apply_chat_template(enable_thinking=True)` from the HF tokenizer."""
    from src.runtime.rkllm_runtime import _build_chat_prompt
    p = _build_chat_prompt(
        system="SYS",
        history=[{"role": "user", "content": "diagnose"}],
        enable_thinking=True,
    )
    assert p.endswith("<|im_start|>assistant\n<think>\n")


def test_build_chat_prompt_default_no_think_prefix():
    """Default (Qwen 2.5 legacy path): no `<think>` prefix injected.
    Critical for rollout safety — devices on old cached models must
    not get the prefix because their tokenizer doesn't know about it."""
    from src.runtime.rkllm_runtime import _build_chat_prompt
    p = _build_chat_prompt(system="SYS", history=[{"role": "user", "content": "x"}])
    assert p.endswith("<|im_start|>assistant\n")
    assert "<think>" not in p


def test_try_load_sets_enable_thinking_for_qwen3_model():
    """try_load() must wire the filename-based thinking detection so
    the backend dataclass carries the correct mode for run_troubleshoot.
    Without this end-to-end wiring the prompt prefix injection never
    fires and the model produces non-thinking-mode output."""
    from src.runtime import rkllm_runtime as mod
    fake_lib = MagicMock()
    fake_lib.rkllm_init = MagicMock(return_value=0)
    fake_lib.rkllm_destroy = MagicMock(return_value=0)
    qwen3_path = "/uniondrive/model/qwen3-1.7b-rk3588-w8a8.rkllm"
    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value=qwen3_path), \
         patch("ctypes.CDLL", return_value=fake_lib):
        backend = mod.try_load()
    assert backend is not None
    assert backend._enable_thinking is True


def test_try_load_leaves_thinking_off_for_qwen_2_5_model():
    """Rollout-safety regression: a device with the old 1.5B still
    cached (Qwen 3 .rkllm not yet downloaded) must boot with thinking
    OFF — otherwise the model tokenizes the `<think>\\n` prefix as raw
    text and emits garbled output."""
    from src.runtime import rkllm_runtime as mod
    fake_lib = MagicMock()
    fake_lib.rkllm_init = MagicMock(return_value=0)
    fake_lib.rkllm_destroy = MagicMock(return_value=0)
    legacy_path = "/uniondrive/model/qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm"
    with patch.object(mod, "find_so_path", return_value="/fake/lib.so"), \
         patch.object(mod, "find_model_path", return_value=legacy_path), \
         patch("ctypes.CDLL", return_value=fake_lib):
        backend = mod.try_load()
    assert backend is not None
    assert backend._enable_thinking is False


def test_run_troubleshoot_strips_think_from_history_in_qwen3_mode():
    """The whole point of the Qwen 3 swap: per-turn `<think>` content
    must NOT accumulate in KV cache across the tool-call loop. Verify
    by inspecting the prompt sent on turn N+1 — it should NOT contain
    turn N's chain-of-thought, only the post-`</think>` structured
    output (tool calls + prose)."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    turn_outputs = [
        # Turn 0: think + tool call. Output starts INSIDE think block
        # (prefix already injected via prompt).
        ("CoT-TURN-0-INTERNAL: I should check overall system state.</think>\n"
         '<tool_call>{"name":"diag/summary","arguments":{}}</tool_call>'),
        # Turn 1: think + verdict. CoT-TURN-1 should also not show up
        # on turn 2, but more importantly CoT-TURN-0 must be GONE.
        ("CoT-TURN-1-INTERNAL: Everything looks fine.</think>\n"
         '<verdict>{"summary":"healthy","severity":"green","root_cause":"n_a"}</verdict>'),
    ]
    turn_idx = {"i": 0}
    prompts_seen = []

    class FakeRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            prompts_seen.append(prompt)
            i = turn_idx["i"]
            turn_idx["i"] = i + 1
            return turn_outputs[i] if i < len(turn_outputs) else "(end)"

    async def fake_executor(tool, args):
        return {"overall": "green", "subsystems": {}}

    backend = RKLLMBackend(
        loaded=True, _runtime=FakeRuntime(), _enable_thinking=True,
    )
    backend.wire_runtime_deps(
        tool_executor=fake_executor, action_signer=lambda x: "f" * 64,
    )

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x", session_id="sid")]

    asyncio.run(collect())

    # Turn 1's prompt MUST NOT contain turn 0's internal CoT — that's
    # the entire point of the KV-bloat-safe history rewrite.
    assert len(prompts_seen) >= 2, "Expected at least two turns"
    turn_1_prompt = prompts_seen[1]
    assert "CoT-TURN-0-INTERNAL" not in turn_1_prompt, (
        "Per-turn <think> content leaked into next turn's prompt — "
        "history rewrite is broken; KV cache will bloat across the "
        "tool-call loop. This is the regression Qwen 3 model-card "
        "guidance explicitly warns against."
    )
    # The structured tool_call from turn 0 MUST still be in turn 1's
    # prompt (otherwise the model loses context of what it called).
    assert "diag/summary" in turn_1_prompt


def test_run_troubleshoot_hides_think_content_from_thought_event():
    """User preference (literal): 'if <think> process can be hidden ...
    it is preferred to hide it'. Chain-of-thought must NOT reach the SSE
    stream — only the post-think prose (and structured tool/verdict
    events) are user-visible.

    Verifies the strip happens at the SSE boundary AND that when the
    post-think prose is empty (turn was 100% structured output), a
    short synthetic marker is emitted so the stream isn't silent."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    class FakeRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            # Turn output: CoT inside the think block, then ONLY a
            # verdict block. No post-think prose. The thought event
            # should fall back to the synthetic "Analyzing..." marker.
            return (
                "CHAIN_OF_THOUGHT_CONTENT_SHOULD_NEVER_REACH_SSE</think>\n"
                '<verdict>{"summary":"x","severity":"green","root_cause":"y"}</verdict>'
            )

    backend = RKLLMBackend(
        loaded=True, _runtime=FakeRuntime(), _enable_thinking=True,
    )
    backend.wire_runtime_deps(
        tool_executor=None, action_signer=lambda x: "f" * 64,
    )

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    thoughts = [e for e in events if e["type"] == "thought"]
    # Hard requirement: think content never appears in ANY thought event
    for t in thoughts:
        assert "CHAIN_OF_THOUGHT_CONTENT_SHOULD_NEVER_REACH_SSE" not in t["payload"], (
            f"Think content leaked into SSE thought event — user-visible: "
            f"{t['payload']!r}"
        )
    # Non-silent stream: a synthetic marker should fill the gap when
    # the post-think prose is empty
    assert any("Analyzing" in t["payload"] for t in thoughts), (
        "Expected synthetic 'Analyzing diagnostics...' marker when "
        "post-think prose is empty"
    )


def test_run_troubleshoot_surfaces_post_think_prose_when_present():
    """Sibling test: when the model DOES emit non-block prose after
    </think>, that prose reaches the SSE thought event verbatim.
    Confirms we strip think content but not the actual user-visible
    reasoning the model offers after."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    class FakeRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            return (
                "HIDDEN_COT_CONTENT</think>\n"
                "VISIBLE_PROSE_AFTER_THINK that explains what's happening.\n"
                '<verdict>{"summary":"x","severity":"green","root_cause":"y"}</verdict>'
            )

    backend = RKLLMBackend(
        loaded=True, _runtime=FakeRuntime(), _enable_thinking=True,
    )
    backend.wire_runtime_deps(
        tool_executor=None, action_signer=lambda x: "f" * 64,
    )

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    thoughts = [e for e in events if e["type"] == "thought"]
    assert any("VISIBLE_PROSE_AFTER_THINK" in t["payload"] for t in thoughts), (
        "Post-think prose should reach the SSE thought event"
    )
    for t in thoughts:
        assert "HIDDEN_COT_CONTENT" not in t["payload"], (
            "Think content must NEVER leak into SSE, even when post-think "
            "prose is also present"
        )


def test_run_troubleshoot_force_verdict_at_max_turns_minus_one():
    """Even when the model is happily calling tools but never finalizes,
    the backend injects the force-verdict directive at MAX_TURNS-1 and
    captures the real verdict."""
    import asyncio
    from src.runtime.rkllm_runtime import (
        RKLLMBackend, FORCE_VERDICT_DIRECTIVE, MAX_TURNS,
    )

    turn_idx = {"i": 0}
    prompts_seen = []

    class ToolHappyRuntime:
        def generate(self, prompt, role="user", enable_thinking=False, keep_history=0, timeout_s=90.0):
            prompts_seen.append(prompt)
            i = turn_idx["i"]
            turn_idx["i"] = i + 1
            # Always call diag/summary unless force-verdict directive is in prompt
            if FORCE_VERDICT_DIRECTIVE in prompt:
                return '<verdict>{"summary":"forced","severity":"yellow","root_cause":"max_turns_force"}</verdict>'
            return '<tool_call>{"name":"diag/summary","arguments":{}}</tool_call>'

    async def fake_executor(tool, args):
        return {"overall": "green"}

    backend = RKLLMBackend(loaded=True, _runtime=ToolHappyRuntime())
    backend.wire_runtime_deps(tool_executor=fake_executor, action_signer=lambda x: "f" * 64)

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1
    # Real verdict (from force-verdict turn), NOT the synthetic max_turns fallback
    assert verdicts[0]["payload"]["root_cause"] == "max_turns_force"
    # Backend made MAX_TURNS calls (each tool-only turn iterates; final
    # turn has the directive)
    assert turn_idx["i"] <= MAX_TURNS
