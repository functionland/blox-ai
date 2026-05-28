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

    Each SSE event may carry multiple field lines (id, event, data,
    retry) separated by `\\n`, terminated by an empty line (`\\n\\n`).
    Per the 2026-05-28 resume feature each event now ALSO carries an
    `id: <seq>` line that the client uses as lastEventId. We pluck out
    the JSON-bearing `data:` line and parse it.
    """
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                break  # one data line per event in this codebase
    return events


def _parse_sse_with_ids(text: str) -> list[tuple[str | None, dict]]:
    """Like _parse_sse but also returns the SSE `id:` field (the
    monotonic seq number for resume). Used by resume tests."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        sid = None
        data = None
        for line in block.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("id: "):
                sid = line[len("id: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if data is not None:
            out.append((sid, data))
    return out


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


# ---------------------------------------------------------------------------
# Resume (mid-session reattach after SSE disconnect) — 2026-05-28
# ---------------------------------------------------------------------------

def test_each_event_carries_a_monotonic_id(client):
    """Resume relies on each event carrying an `id:` SSE field with a
    monotonic per-session sequence number. Client persists the highest
    seen id and supplies it as `?from=` on resume so the server can
    skip already-delivered events."""
    pairs = _parse_sse_with_ids(_post_troubleshoot(client).text)
    seen = [int(sid) for sid, _ in pairs if sid is not None and sid != "-1"]
    assert seen, f"expected SSE id fields on every event; got {pairs!r}"
    # Monotonic + starting from 0
    assert seen == sorted(seen), f"ids not monotonic: {seen}"
    assert seen[0] == 0, f"first id should be 0; got {seen[0]}"


def test_resume_404s_on_unknown_session_id(client):
    r = client.get("/troubleshoot/resume", params={"session_id": "nonexistent", "from": 0})
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "session_not_found"


def test_resume_from_zero_replays_full_buffer(client):
    """After a complete /troubleshoot run, calling /troubleshoot/resume
    with from=0 yields exactly the same events the original POST did.
    Proves the buffer survives generator completion and that resume
    works as a replay path (the use case: user closed app mid-session,
    reopens, server still has the buffered events)."""
    original = _events_from_response(_post_troubleshoot(client))
    sid = original[0]["session_id"]
    r = client.get("/troubleshoot/resume", params={"session_id": sid, "from": 0})
    assert r.status_code == 200
    replayed = _events_from_response(r)
    # Same sequence (compare type + key field per type to avoid jitter
    # in approval_token / non-deterministic fields)
    orig_types = [e["type"] for e in original]
    repl_types = [e["type"] for e in replayed]
    assert orig_types == repl_types, f"replay shape diverged: {orig_types} vs {repl_types}"
    # Session_id matches
    assert replayed[0]["session_id"] == sid


def test_resume_from_midstream_skips_already_delivered_events(client):
    """Client persists lastEventId (the highest seq it received) and
    supplies it as ?from=N+1 on resume. Server yields only events
    with seq > N. Proves we don't double-deliver."""
    pairs = _parse_sse_with_ids(_post_troubleshoot(client).text)
    sid = next(e["session_id"] for _, e in pairs if e.get("type") == "session_started")
    real_ids = [int(s) for s, _ in pairs if s is not None and s != "-1"]
    cutoff = real_ids[len(real_ids) // 2]  # midpoint
    r = client.get(
        "/troubleshoot/resume",
        params={"session_id": sid, "from": cutoff + 1},
    )
    assert r.status_code == 200
    later_pairs = _parse_sse_with_ids(r.text)
    later_ids = [int(s) for s, _ in later_pairs if s is not None and s != "-1"]
    # All replayed ids are strictly greater than cutoff
    assert all(i > cutoff for i in later_ids), (
        f"resume from {cutoff+1} returned an id <= cutoff: {later_ids}"
    )
    # And the full tail (everything after cutoff) is present
    expected_tail = [i for i in real_ids if i > cutoff]
    assert later_ids == expected_tail, (
        f"resume tail mismatch: expected {expected_tail}, got {later_ids}"
    )


def test_truncation_marker_synthesized_when_from_predates_buffer_head(client):
    """When the client supplies a `from` that's older than the oldest
    seq still in the buffer (buffer overflowed during the disconnect
    window), the server injects a synthetic `thought` event explaining
    the gap. Reused-type (thought) so we don't grow the SSE schema for
    a flow-control concern."""
    # Drive a normal session so a session is registered + the buffer
    # finishes naturally, then mutate the buffer in place to simulate
    # the post-truncation state.
    pairs = _parse_sse_with_ids(_post_troubleshoot(client).text)
    sid = next(e["session_id"] for _, e in pairs if e.get("type") == "session_started")
    session = client.app.state.session_manager.get(sid)
    # Force a synthetic gap: pretend events 0-4 were evicted from the
    # head, the buffer now starts at seq=5. dropped_count is for
    # observability — _stream_from_buffer derives the marker text
    # from the buffer head vs from_seq, not from dropped_count.
    session.event_buffer = [(5, {"type": "thought", "payload": "head"})]
    session.dropped_count = 5

    r = client.get(
        "/troubleshoot/resume",
        params={"session_id": sid, "from": 0},
    )
    assert r.status_code == 200
    events = _events_from_response(r)
    # First event must be the truncation marker (thought, mentions
    # "dropped" — the exact wording is documented in
    # _stream_from_buffer).
    assert events, "expected at least the truncation marker + the head event"
    assert events[0]["type"] == "thought"
    assert "dropped" in events[0]["payload"].lower(), (
        f"expected 'dropped' in marker; got {events[0]['payload']!r}"
    )
    # Second event is the actual buffered head event
    assert events[1]["payload"] == "head"


def test_post_to_active_session_returns_409(client):
    """Two concurrent POST /troubleshoot calls to the same session_id
    would spawn duplicate generator tasks writing into one buffer —
    chaos. Reject the second with 409 + a hint to use /resume.

    Direct state injection: we don't actually run a parallel generator
    (that risks hanging the test); we just plant a fake-still-running
    task on the session_id and verify the route guards against it."""
    fixed_sid = "active-session-409-test"
    session_mgr = client.app.state.session_manager
    session = session_mgr.create(session_id=fixed_sid)

    class _FakeRunningTask:
        def done(self):
            return False

    session.generator_task = _FakeRunningTask()
    session.generator_done = False

    r = client.post("/troubleshoot",
                    json={"prompt": "p", "session_id": fixed_sid})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["error"] == "session_already_active"
    # Resume hint in detail
    assert "resume" in body.get("detail", "").lower()


def test_post_to_completed_session_starts_fresh_generator(client):
    """The 409 guard is gated on `not generator_done` — once a session's
    prior run has finished, POSTing again on the same session_id starts
    a fresh generator and resets the buffer. Lets the client recycle a
    session_id after a verdict instead of being stuck with a 409."""
    fixed_sid = "completed-session-reuse-test"
    session_mgr = client.app.state.session_manager
    session = session_mgr.create(session_id=fixed_sid)

    class _FakeCompletedTask:
        def done(self):
            return True

    session.generator_task = _FakeCompletedTask()
    session.generator_done = True
    session.next_seq = 999  # simulate prior session activity
    session.event_buffer = [(998, {"type": "thought", "payload": "stale"})]

    r = client.post("/troubleshoot",
                    json={"prompt": "p", "session_id": fixed_sid})
    assert r.status_code == 200
    events = _events_from_response(r)
    # New session reset seq to 0; first event is session_started with
    # the SAME session_id (caller-supplied echoes back).
    assert events[0]["type"] == "session_started"
    assert events[0]["session_id"] == fixed_sid
    # Buffer was reset
    fresh_session = session_mgr.get(fixed_sid)
    assert fresh_session.next_seq > 0  # populated by the new run
    # Stale "thought" payload from prior buffer should NOT appear
    payloads = [e.get("payload") for e in events if e.get("type") == "thought"]
    assert "stale" not in payloads
