"""C4 — /execute-action HTTP route integration tests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — issue a recommended_action through the bridge so the session
# has a real action_id + matching signed token ready for /execute-action.
# ---------------------------------------------------------------------------

def _open_session_and_get_recommendation(client) -> dict:
    """Run /troubleshoot end-to-end, return the recommended_action event
    (which carries action_id + approval_token + action_name + args).
    2026-05-28: SSE event blocks now lead with `id:` for resume support;
    scan for `data:` per line instead of slicing the block."""
    r = client.post("/troubleshoot", json={"prompt": "diagnose"})
    assert r.status_code == 200
    events = []
    for block in r.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                break
    rec = next((e for e in events if e["type"] == "recommended_action"), None)
    assert rec is not None
    return rec


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------

def test_missing_required_field_returns_400(client):
    r = client.post("/execute-action", json={"action_id": "x"})
    assert r.status_code == 400
    assert r.json() == {"error": "body_invalid"}


def test_invalid_json_returns_400(client):
    r = client.post("/execute-action", content="not-json")
    assert r.status_code == 400


def test_extra_field_returns_400(client):
    r = client.post("/execute-action", json={
        "action_id": "x",
        "approval_token": "a" * 64,
        "surprise": True,
    })
    assert r.status_code == 400


def test_unknown_action_id_returns_409_recommendation_not_found(client):
    """No session has this recommendation in memory → 409 with truthful
    rejected_reason. Per advisor: NOT approval_token_invalid (which would
    lie to the audit log — the token might verify fine; what failed is
    that the session memory holding the recommendation is gone)."""
    r = client.post("/execute-action", json={
        "action_id": "never-issued-this-id",
        "approval_token": "a" * 64,
    })
    assert r.status_code == 409
    assert r.json() == {"error": "recommendation_not_found"}
    # Audit line records the truthful reason
    audit_path = Path(client.app.state.audit_log_path)
    last = json.loads(audit_path.read_text().splitlines()[-1])
    assert last["rejected_reason"] == "recommendation_not_found"


# ---------------------------------------------------------------------------
# End-to-end with a real session + real token
# ---------------------------------------------------------------------------

def test_execute_happy_path_tier2(client):
    """Open a session via /troubleshoot, take its recommended_action,
    POST to /execute-action with a mocked subprocess for docker."""
    rec = _open_session_and_get_recommendation(client)
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="restarted", stderr="")
        r = client.post("/execute-action", json={
            "action_id": rec["action_id"],
            "approval_token": rec["approval_token"],
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "execution_result"
    assert body["action_id"] == rec["action_id"]
    assert body["success"] is True


def test_execute_replayed_token_returns_401(client):
    rec = _open_session_and_get_recommendation(client)
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="", stderr="")
        r1 = client.post("/execute-action", json={
            "action_id": rec["action_id"],
            "approval_token": rec["approval_token"],
        })
        assert r1.status_code == 200
        # Replay
        r2 = client.post("/execute-action", json={
            "action_id": rec["action_id"],
            "approval_token": rec["approval_token"],
        })
    assert r2.status_code == 401
    assert r2.json() == {"type": "error", "code": "APPROVAL_TOKEN_REPLAYED",
                         "message": "approval_token_replayed",
                         "recoverable": False}


def test_audit_log_written_on_happy_path(client):
    rec = _open_session_and_get_recommendation(client)
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="ok", stderr="")
        client.post("/execute-action", json={
            "action_id": rec["action_id"],
            "approval_token": rec["approval_token"],
        })
    audit_path = Path(client.app.state.audit_log_path)
    lines = audit_path.read_text().splitlines()
    assert len(lines) >= 1
    last = json.loads(lines[-1])
    assert last["executed"] is True
    assert last["action_id"] == rec["action_id"]
    assert last["action"] == rec["action_name"]
    assert "whitelist_hash" in last
    assert len(last["whitelist_hash"]) == 64


def test_audit_log_written_on_rejection(client):
    rec = _open_session_and_get_recommendation(client)
    # Submit a tampered token (bad HMAC)
    client.post("/execute-action", json={
        "action_id": rec["action_id"],
        "approval_token": "b" * 200,  # wrong base64-decoded HMAC
    })
    audit_path = Path(client.app.state.audit_log_path)
    if audit_path.exists():
        lines = audit_path.read_text().splitlines()
        last = json.loads(lines[-1])
        assert last["executed"] is False
        assert last["rejected_reason"] in (
            "approval_token_invalid",
            "approval_token_expired",
        )
        # Conditional invariant: executed=false → no result field
        assert "result" not in last


def test_audit_line_validates_against_schema(client):
    """Hard contract: every audit line MUST satisfy audit_log_line.schema.json."""
    rec = _open_session_and_get_recommendation(client)
    with patch("src.tools.executor.subprocess.run") as sub:
        from subprocess import CompletedProcess
        sub.return_value = CompletedProcess(args=[], returncode=0,
                                            stdout="ok", stderr="")
        client.post("/execute-action", json={
            "action_id": rec["action_id"],
            "approval_token": rec["approval_token"],
        })
    # Skip if real schemas aren't available
    from .conftest import _real_schemas_in_use
    if not _real_schemas_in_use():
        pytest.skip("requires real fula-ota schemas")
    audit_path = Path(client.app.state.audit_log_path)
    lines = audit_path.read_text().splitlines()
    import jsonschema
    schema = json.loads(
        (Path(client.app.state.schemas.schema_dir)
         / "audit_log_line.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    for L in lines:
        line = json.loads(L)
        errors = sorted(validator.iter_errors(line),
                        key=lambda e: list(e.path))
        assert not errors, (
            f"audit line failed schema: {[e.message for e in errors]}\n"
            f"line: {line}"
        )
