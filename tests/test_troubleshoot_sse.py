"""C2 — /troubleshoot SSE skeleton + tool-call loop tests.

Coverage:
  - happy-path event sequence: session_started → thought → tool_call →
    tool_result → thought → verdict → recommended_action
  - every emitted event passes sse_events.schema.json validation
  - request body validation (extra fields rejected, empty prompt rejected)
  - tool_result.call_id matches the preceding tool_call.call_id
  - tool_executor failure → tool_result.ok=false + error field
  - backend emitting a schema-invalid event → stream-terminating
    synthetic error event (not an HTTP 500)
  - response headers signal SSE + no-buffering
"""
from __future__ import annotations

import json
from typing import Iterator

import jsonschema
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> list[dict]:
    """Parse a complete SSE response body into a list of event dicts.

    Each event is `data: <json>\\n\\n`. The test client buffers everything
    into one string, so we split on the empty-line terminator.
    """
    events = []
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("data: "):
            payload = raw[len("data: "):]
            events.append(json.loads(payload))
    return events


def _post_troubleshoot(client, prompt: str = "my device is slow", **extra):
    body = {"prompt": prompt, **extra}
    return client.post("/troubleshoot", json=body)


def _events_from_response(resp) -> list[dict]:
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    return _parse_sse(resp.text)


# ---------------------------------------------------------------------------
# Response framing
# ---------------------------------------------------------------------------

def test_troubleshoot_returns_sse_content_type(client):
    r = _post_troubleshoot(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_troubleshoot_sets_no_buffering_headers(client):
    r = _post_troubleshoot(client)
    assert r.headers.get("cache-control") == "no-cache"
    assert r.headers.get("x-accel-buffering") == "no"


def test_each_event_is_terminated_by_double_newline(client):
    r = _post_troubleshoot(client)
    # SSE spec: events terminated by empty line. The body must end with
    # at least one `\n\n` after the last event.
    assert r.text.endswith("\n\n"), (
        "stream must end with the SSE double-newline so clients know the "
        "last event is complete"
    )


# ---------------------------------------------------------------------------
# Event sequence
# ---------------------------------------------------------------------------

def test_first_event_is_session_started(client):
    events = _events_from_response(_post_troubleshoot(client))
    assert events[0]["type"] == "session_started"


def test_session_started_carries_uuid_session_id(client):
    events = _events_from_response(_post_troubleshoot(client))
    started = events[0]
    assert isinstance(started["session_id"], str)
    assert len(started["session_id"]) >= 1
    # UUID v4 format check (we mint via uuid.uuid4)
    import re
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        started["session_id"],
    ), f"expected UUIDv4, got {started['session_id']!r}"


def test_session_started_protocol_version_is_3(client):
    events = _events_from_response(_post_troubleshoot(client))
    assert events[0]["protocol_version"] == 3


def test_caller_supplied_session_id_is_echoed(client):
    fixed = "caller-supplied-session-id"
    events = _events_from_response(
        _post_troubleshoot(client, session_id=fixed),
    )
    assert events[0]["session_id"] == fixed


def test_sequence_contains_a_tool_call_and_matching_tool_result(client):
    events = _events_from_response(_post_troubleshoot(client))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_calls, "expected at least one tool_call event"
    assert tool_results, "expected at least one tool_result event"
    # IDs must match between paired call/result
    call_ids_called = [e["call_id"] for e in tool_calls]
    call_ids_resulted = [e["call_id"] for e in tool_results]
    for cid in call_ids_called:
        assert cid in call_ids_resulted, (
            f"tool_call(call_id={cid}) had no matching tool_result"
        )


def test_tool_result_appears_after_its_matching_tool_call(client):
    """Ordering invariant: a tool_result event MUST come after its
    matching tool_call, never before."""
    events = _events_from_response(_post_troubleshoot(client))
    indexed = list(enumerate(events))
    for i, ev in indexed:
        if ev["type"] == "tool_result":
            # Find the matching tool_call earlier in the stream
            matching = [j for j, e in indexed[:i]
                        if e["type"] == "tool_call" and e["call_id"] == ev["call_id"]]
            assert matching, (
                f"tool_result(call_id={ev['call_id']}) emitted before its "
                f"matching tool_call (positions: {indexed})"
            )


def test_tool_result_payload_is_the_executor_canned_response(client):
    """Bridge wires the executor's return value into tool_result.payload."""
    from src.runtime.mock_diag import _CANNED  # noqa: PLC2701 (test internal)
    events = _events_from_response(_post_troubleshoot(client))
    summary_tool_result = next(
        (e for e in events if e["type"] == "tool_result"
         and e["call_id"] == "mock-call-1"),
        None,
    )
    assert summary_tool_result is not None
    assert summary_tool_result["payload"] == _CANNED["diag/summary"]


def test_sequence_ends_with_verdict_then_recommended_action(client):
    events = _events_from_response(_post_troubleshoot(client))
    types = [e["type"] for e in events]
    verdict_idx = types.index("verdict")
    assert verdict_idx >= 0
    # Recommended_action(s) follow verdict in the canned sequence
    later = types[verdict_idx + 1:]
    assert "recommended_action" in later, (
        f"expected recommended_action after verdict; sequence: {types}"
    )


def test_verdict_severity_is_in_closed_enum(client):
    events = _events_from_response(_post_troubleshoot(client))
    verdict = next(e for e in events if e["type"] == "verdict")
    assert verdict["payload"]["severity"] in ("green", "yellow", "red")


def test_recommended_action_carries_required_fields(client):
    events = _events_from_response(_post_troubleshoot(client))
    action = next(e for e in events if e["type"] == "recommended_action")
    for f in ("action_id", "action_name", "args", "reasoning",
              "confidence", "tier", "approval_token"):
        assert f in action, f"missing required field on recommended_action: {f}"
    assert action["tier"] in (2, 3)
    assert 0.0 <= action["confidence"] <= 1.0
    assert len(action["approval_token"]) >= 64


# ---------------------------------------------------------------------------
# Schema validation invariants
# ---------------------------------------------------------------------------

def test_every_event_validates_against_sse_events_schema(client):
    """The bridge MUST validate every emit before yielding. This is the
    hard invariant of the C2 contract — any schema-invalid event in the
    stream means the bridge failed open.

    Requires the REAL sse_events.schema.json (the stub fallback would
    accept anything and provide no signal). Skips under stubs.
    """
    from .conftest import _real_schemas_in_use
    if not _real_schemas_in_use():
        pytest.skip(
            "set BLOX_AI_FULA_OTA_SCHEMA_DIR or place a fula-ota sibling "
            "checkout to run this test"
        )
    events = _events_from_response(_post_troubleshoot(client))
    from pathlib import Path
    import json as _json
    schema_dir = Path(client.app.state.schemas.schema_dir)
    schema = _json.loads(
        (schema_dir / "sse_events.schema.json").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    for ev in events:
        errors = sorted(validator.iter_errors(ev), key=lambda e: e.path)
        assert not errors, (
            f"event failed validation: type={ev.get('type')!r} "
            f"errors={[e.message for e in errors]}"
        )


# ---------------------------------------------------------------------------
# Request body validation
# ---------------------------------------------------------------------------

def test_empty_prompt_returns_422(client):
    r = client.post("/troubleshoot", json={"prompt": ""})
    assert r.status_code == 422


def test_missing_prompt_returns_422(client):
    r = client.post("/troubleshoot", json={})
    assert r.status_code == 422


def test_extra_field_returns_422(client):
    """Closed model — typo'd field name surfaces immediately."""
    r = client.post(
        "/troubleshoot",
        json={"prompt": "x", "prmopt": "typo"},
    )
    assert r.status_code == 422


def test_oversized_prompt_returns_422(client):
    r = client.post(
        "/troubleshoot",
        json={"prompt": "x" * 10_001},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Bridge error-handling
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_failing_executor(schema_dir_with_all_required, monkeypatch):
    """A client wired with an executor that raises for every call."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    import sys
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.runtime.mock_diag", "src.session.tool_call_loop",
                "src.routes.troubleshoot"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app

    class FailingExecutor:
        name = "failing"

        async def __call__(self, tool, args):
            raise RuntimeError(f"simulated failure for {tool}")

    # Override after lifespan runs; we need to mutate state inside a context.
    with TestClient(fresh_app) as c:
        c.app.state.tool_executor = FailingExecutor()
        yield c


def test_executor_failure_produces_ok_false_tool_result(client_with_failing_executor):
    events = _events_from_response(
        _post_troubleshoot(client_with_failing_executor),
    )
    tool_result = next(
        (e for e in events if e["type"] == "tool_result"),
        None,
    )
    assert tool_result is not None
    assert tool_result["ok"] is False
    assert "error" in tool_result
    assert "simulated failure" in tool_result["error"]


@pytest.fixture
def client_with_broken_backend(schema_dir_with_all_required, monkeypatch):
    """A client wired with a backend that emits a schema-invalid event.
    Mirrors the failing_executor fixture pattern to avoid the inline-mutation
    state-bleed that surfaces on aarch64 (Linux) but not on Windows."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    import sys
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.runtime.mock_diag", "src.session.tool_call_loop",
                "src.routes.troubleshoot"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app

    class BrokenBackend:
        name = "broken"
        loaded = True

        def status_snapshot(self):
            return {}

        async def run_troubleshoot(self, prompt, session_id=None):
            # Missing required fields → invalid against sse_events schema
            yield {"type": "thought"}  # 'payload' required + minLength:1

    with TestClient(fresh_app) as c:
        c.app.state.backend = BrokenBackend()
        yield c


def test_schema_invalid_backend_event_emits_synthetic_error(client_with_broken_backend):
    """If the backend somehow emits a malformed NON-tool_call event,
    the bridge emits a recoverable error event and continues iterating
    (no return) — so a single bad event doesn't kill the whole session.

    Bug fix 2026-05-26: previously the bridge `return`ed on first
    schema-invalid event, which killed any in-flight troubleshooting
    session (user saw '[SCHEMA_VIOLATION]' instead of recommendations).
    Now: invalid event → recoverable error + keep streaming.

    Requires the REAL sse_events.schema.json (the permissive stub
    fallback would accept anything). Skips when the stub is in use.
    """
    from .conftest import _real_schemas_in_use
    if not _real_schemas_in_use():
        pytest.skip(
            "set BLOX_AI_FULA_OTA_SCHEMA_DIR or place a fula-ota sibling "
            "checkout to run this test; the permissive stub fallback "
            "validates any object as a valid SSE event"
        )
    r = _post_troubleshoot(client_with_broken_backend)
    events = _events_from_response(r)
    err = next((e for e in events if e["type"] == "error"), None)
    assert err is not None, (
        f"expected an error event; got {len(events)} event(s): "
        f"{[e.get('type') for e in events]}"
    )
    # The bad event was a thought (not a tool_call) so the bridge marks
    # the recovery as ongoing — recoverable=True.
    assert err["code"] == "SCHEMA_VIOLATION_RECOVERED", (
        f"expected SCHEMA_VIOLATION_RECOVERED, got {err.get('code')}"
    )
    assert err["recoverable"] is True


@pytest.fixture
def client_with_bad_tool_name_backend(schema_dir_with_all_required, monkeypatch):
    """Backend whose first event is a tool_call with a NON-WHITELISTED
    tool name ('diag/discovery'). Mirrors the user's lab observation
    2026-05-26 where the 1.5B Qwen hallucinated a tool that doesn't
    exist + the bridge killed the whole stream."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    import sys
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.runtime.mock_diag", "src.session.tool_call_loop",
                "src.routes.troubleshoot"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app

    class BadToolBackend:
        name = "bad-tool"
        loaded = True
        consumes_tool_results = False
        def status_snapshot(self): return {}
        async def run_troubleshoot(self, prompt, session_id=None):
            yield {"type": "tool_call", "call_id": "rk-0-0",
                   "payload": {"tool": "diag/discovery", "args": {}}}

    with TestClient(fresh_app) as c:
        c.app.state.backend = BadToolBackend()
        yield c


def test_schema_invalid_tool_call_yields_synthetic_tool_result(
    client_with_bad_tool_name_backend,
):
    """Regression guard 2026-05-26: if the LLM hallucinates a tool name
    not in the closed enum (e.g. 'diag/discovery'), the bridge MUST
    synthesize a tool_result with ok=false + a clear 'unknown tool'
    error — so the LLM has a chance to self-correct on the next turn,
    AND the stream keeps flowing.

    Before fix: bridge `return`ed on first invalid tool_call → user
    saw [SCHEMA_VIOLATION] and no further events. Lab observed:
    LLM emitted `{"tool": "diag/discovery"}` and entire session bombed.
    """
    from .conftest import _real_schemas_in_use
    if not _real_schemas_in_use():
        pytest.skip("real sse_events.schema.json required")
    r = _post_troubleshoot(client_with_bad_tool_name_backend)
    events = _events_from_response(r)
    types = [e.get("type") for e in events]
    err = next((e for e in events if e.get("type") == "error"), None)
    tr = next((e for e in events if e.get("type") == "tool_result"), None)
    assert err is not None, f"expected error event; saw types: {types}"
    assert err["recoverable"] is True, (
        "schema-invalid tool_call must NOT kill the stream; bridge "
        "should yield recoverable error and continue"
    )
    assert tr is not None, (
        "expected synthetic tool_result with ok=false so the LLM can "
        "read the failure and pick a valid tool next turn"
    )
    assert tr["ok"] is False
    assert tr["call_id"] == "rk-0-0"
    assert "diag/discovery" in (tr.get("error") or "")


def test_unknown_tool_in_executor_fails_open_with_ok_false(
    schema_dir_with_all_required, monkeypatch,
):
    """If the bridge sees a valid tool_call but the executor raises
    UnknownToolError, tool_result.ok=false + error captures the bad tool."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    import sys
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.runtime.mock_diag", "src.session.tool_call_loop",
                "src.routes.troubleshoot"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app
    from src.runtime.mock_diag import UnknownToolError

    class PartialExecutor:
        name = "partial"

        async def __call__(self, tool, args):
            raise UnknownToolError(tool)

    with TestClient(fresh_app) as c:
        c.app.state.tool_executor = PartialExecutor()
        r = _post_troubleshoot(c)
    events = _events_from_response(r)
    tool_result = next(
        (e for e in events if e["type"] == "tool_result"),
        None,
    )
    assert tool_result is not None
    assert tool_result["ok"] is False
    assert "diag/summary" in tool_result["error"]


# ---------------------------------------------------------------------------
# Session-id round-trip
# ---------------------------------------------------------------------------

def test_each_call_without_session_id_mints_a_fresh_one(client):
    a = _events_from_response(_post_troubleshoot(client))
    b = _events_from_response(_post_troubleshoot(client))
    assert a[0]["session_id"] != b[0]["session_id"]
