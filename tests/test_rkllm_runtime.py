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
        def generate(self, prompt, timeout_s=90.0):
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
    assert err["recoverable"] is False


def test_run_troubleshoot_terminates_when_model_only_emits_prose():
    """If the model produces no verdict + no tool_calls (just prose),
    the backend yields a synthetic 'no_verdict_emitted' verdict and
    stops. Avoids spinning to MAX_TURNS for a non-cooperative model."""
    import asyncio
    from src.runtime.rkllm_runtime import RKLLMBackend

    class ProseOnlyRuntime:
        def generate(self, prompt, timeout_s=90.0):
            return "Just some thinking, nothing else."

    backend = RKLLMBackend(loaded=True, _runtime=ProseOnlyRuntime())
    backend.wire_runtime_deps(tool_executor=None, action_signer=lambda x: "f" * 64)

    async def collect():
        return [ev async for ev in backend.run_troubleshoot("x")]

    events = asyncio.run(collect())
    verdict = next(e for e in events if e["type"] == "verdict")
    assert verdict["payload"]["root_cause"] == "no_verdict_emitted"
