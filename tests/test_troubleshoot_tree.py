"""POST /troubleshoot/tree — endpoint tests for Phase 1.c.

Reuses the existing test client fixtures + session manager + executor;
mocks the tree runner with canned tree YAMLs so we can exercise the
endpoint independently of the production trees in fula-ota."""
from __future__ import annotations

import json

import pytest

from src.runtime.tree_dsl import parse_tree
from src.runtime.tree_runner import TreeRunner


def _parse(text):
    import textwrap
    import yaml
    return parse_tree(yaml.safe_load(textwrap.dedent(text)))


def _read_sse_events(content_bytes: bytes) -> list[dict]:
    """Parse SSE response into a list of parsed JSON events."""
    events: list[dict] = []
    text = content_bytes.decode("utf-8")
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                break
    return events


def _make_test_runner():
    """Build a small registry the endpoint can dispatch on."""
    trees = {
        "happy": _parse("""
            id: happy
            version: 1
            title: Always green
            nodes:
              - id: only
                branches:
                  - default:
                      emit_verdict:
                        summary: All good
                        severity: green
                        root_cause: nominal
                      stop: true
        """),
        "diag_call": _parse("""
            id: diag_call
            version: 1
            title: Calls a diag tool
            nodes:
              - id: only
                diag: internet
                branches:
                  - when: "result.dns_ok == False"
                    then:
                      emit_verdict:
                        summary: DNS down
                        severity: red
                        root_cause: dns_unreachable
                      stop: true
                  - default:
                      emit_verdict:
                        summary: ok
                        severity: green
                        root_cause: nominal
                      stop: true
        """),
    }

    async def fake_executor(tool, args):
        if tool == "diag/internet":
            return {"dns_ok": False}
        return {}

    return TreeRunner(trees=trees, tool_executor=fake_executor)


@pytest.fixture
def client_with_tree_runner(client):
    """Inject our test TreeRunner into the app under test."""
    client.app.state.tree_runner = _make_test_runner()
    yield client
    client.app.state.tree_runner = None


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_tree_route_returns_sse_with_verdict(client_with_tree_runner):
    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "happy"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _read_sse_events(r.content)
    verdicts = [e for e in events if e.get("type") == "verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["payload"]["root_cause"] == "nominal"


def test_tree_route_emits_tool_call_and_result_events(client_with_tree_runner):
    """Scenario that calls a diag tool should emit tool_call + tool_result."""
    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "diag_call"},
    )
    assert r.status_code == 200
    events = _read_sse_events(r.content)
    types = [e.get("type") for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    verdicts = [e for e in events if e.get("type") == "verdict"]
    assert verdicts[-1]["payload"]["root_cause"] == "dns_unreachable"


def test_tree_route_accepts_supplied_session_id(client_with_tree_runner):
    """Caller-supplied session_id should be preserved + reusable for resume."""
    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "happy", "session_id": "caller-chosen-id"},
    )
    assert r.status_code == 200
    # Drain the response stream so the generator task can register.
    _ = r.content
    sess = client_with_tree_runner.app.state.session_manager.get("caller-chosen-id")
    assert sess is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_tree_route_503_when_runner_not_loaded(client):
    """If lifespan failed to load trees, /troubleshoot/tree must 503,
    not 500 — the caller should know it's a transient/config issue."""
    client.app.state.tree_runner = None
    r = client.post("/troubleshoot/tree", json={"scenario_id": "happy"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "tree_runner_unavailable"


def test_tree_route_404_on_unknown_scenario(client_with_tree_runner):
    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "not_a_scenario"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "unknown_scenario_id"


def test_tree_route_rejects_missing_scenario_id(client_with_tree_runner):
    r = client_with_tree_runner.post("/troubleshoot/tree", json={})
    assert r.status_code == 422   # pydantic body-validation


def test_tree_route_rejects_extra_fields(client_with_tree_runner):
    """Extra-forbid means a typo like 'scenarioId' (camelCase) is 422
    instead of silently being ignored."""
    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "happy", "scenarioId": "happy"},
    )
    assert r.status_code == 422


def test_tree_route_409_on_concurrent_post(client_with_tree_runner):
    """Posting twice to the same session_id while the first generator
    is still active must 409 (use resume to reattach instead of
    spawning a parallel writer)."""
    from unittest.mock import MagicMock
    # Pre-create a session with a fake long-running task.
    sess = client_with_tree_runner.app.state.session_manager.create(
        session_id="hot-session",
    )

    class FakeTask:
        def done(self):
            return False
    sess.generator_task = FakeTask()
    sess.generator_done = False

    r = client_with_tree_runner.post(
        "/troubleshoot/tree",
        json={"scenario_id": "happy", "session_id": "hot-session"},
    )
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "session_already_active"
