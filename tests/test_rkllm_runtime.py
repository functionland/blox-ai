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
        b = RKLLMBackend()
        events = []
        async for ev in b.run_troubleshoot("test prompt", session_id=None):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    assert events[0]["type"] == "session_started"
    assert events[0]["protocol_version"] == 3
    # Last event is a verdict (yellow until real model lands)
    types = [e["type"] for e in events]
    assert "verdict" in types


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
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = (
        '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
        '<tool_call>not even close to JSON</tool_call>'
        '<tool_call>{"name": "ok2", "arguments": {}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert [c["tool"] for c in calls] == ["ok", "ok2"]


def test_parse_tool_calls_handles_empty_input():
    from src.runtime.rkllm_runtime import parse_tool_calls
    assert parse_tool_calls("") == []
    assert parse_tool_calls("just prose, no tool calls") == []


def test_parse_tool_calls_drops_missing_required_keys():
    from src.runtime.rkllm_runtime import parse_tool_calls
    text = '<tool_call>{"name": "no-args-field"}</tool_call>'
    assert parse_tool_calls(text) == []
