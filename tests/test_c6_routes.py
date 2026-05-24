"""C6 — feedback + pending + cancel route tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_tmp_feedback_log(client, tmp_path):
    """The conftest `client` fixture sets up a TestClient with MockDiag.
    Override the feedback log path to a tmp file so we can read what
    was written."""
    log_path = tmp_path / "ai-feedback.jsonl"
    client.app.state.feedback_log_path = str(log_path)
    yield client, log_path


def _open_session(client) -> str:
    r = client.post("/troubleshoot", json={"prompt": "just diagnose"})
    first = r.text.split("\n\n")[0]
    return json.loads(first[len("data: "):])["session_id"]


def test_feedback_happy_path_writes_log_line(client_with_tmp_feedback_log):
    client, log_path = client_with_tmp_feedback_log
    sid = _open_session(client)
    r = client.post("/feedback", json={
        "session_id": sid,
        "rating": 1,
        "comment": "worked great",
    })
    assert r.status_code == 200
    assert r.json() == {}
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["session_id"] == sid
    assert rec["user_rating"] == 1
    assert rec["comment"] == "worked great"
    assert rec["anonymized_transcript_uploaded"] is False
    assert rec["actions_taken"] == []


def test_feedback_session_detached_still_logged(client_with_tmp_feedback_log):
    """Per Phase 16: feedback after session eviction is still logged
    with empty verdict_summary + empty actions_taken."""
    client, log_path = client_with_tmp_feedback_log
    r = client.post("/feedback", json={
        "session_id": "evicted-session-id",
        "rating": -1,
    })
    assert r.status_code == 200
    rec = json.loads(log_path.read_text().splitlines()[0])
    assert rec["verdict_summary"] == ""
    assert rec["actions_taken"] == []


def test_feedback_strips_crlf_from_comment(client_with_tmp_feedback_log):
    client, log_path = client_with_tmp_feedback_log
    sid = _open_session(client)
    r = client.post("/feedback", json={
        "session_id": sid,
        "rating": 0,
        "comment": "line1\nline2\r\nline3",
    })
    assert r.status_code == 200
    rec = json.loads(log_path.read_text().splitlines()[0])
    assert "\n" not in rec["comment"]
    assert "\r" not in rec["comment"]
    assert "line1" in rec["comment"] and "line3" in rec["comment"]


def test_feedback_invalid_rating_returns_400(client_with_tmp_feedback_log):
    client, log_path = client_with_tmp_feedback_log
    r = client.post("/feedback", json={
        "session_id": "sid",
        "rating": 5,  # not in {-1, 0, 1}
    })
    assert r.status_code == 400
    assert log_path.exists() is False or log_path.read_text() == ""


def test_feedback_missing_required_returns_400(client_with_tmp_feedback_log):
    client, _ = client_with_tmp_feedback_log
    r = client.post("/feedback", json={"rating": 1})
    assert r.status_code == 400


def test_feedback_non_json_returns_400(client_with_tmp_feedback_log):
    client, _ = client_with_tmp_feedback_log
    r = client.post("/feedback", content="not json")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /pending
# ---------------------------------------------------------------------------

def test_pending_returns_empty_when_log_missing(client, tmp_path):
    client.app.state.pending_log_path = str(tmp_path / "no-such-file.jsonl")
    r = client.get("/pending")
    assert r.status_code == 200
    assert r.json() == {}


def test_pending_returns_last_entry(client, tmp_path):
    log = tmp_path / "ai-pending-actions.jsonl"
    log.write_text(
        json.dumps({"ts": "2026-05-24T01:00:00Z", "trigger": "isolation_mode",
                    "actions": [{"action_id": "old"}]}) + "\n"
        + json.dumps({"ts": "2026-05-24T07:00:00Z", "trigger": "isolation_mode",
                      "actions": [{"action_id": "newer"}]}) + "\n"
    )
    client.app.state.pending_log_path = str(log)
    r = client.get("/pending")
    assert r.status_code == 200
    body = r.json()
    assert body["actions"][0]["action_id"] == "newer"


def test_pending_returns_empty_on_malformed_last_line(client, tmp_path):
    log = tmp_path / "ai-pending-actions.jsonl"
    log.write_text("not-json\n")
    client.app.state.pending_log_path = str(log)
    r = client.get("/pending")
    assert r.status_code == 200
    assert r.json() == {}


def test_pending_empty_file_returns_empty(client, tmp_path):
    log = tmp_path / "ai-pending-actions.jsonl"
    log.write_text("")
    client.app.state.pending_log_path = str(log)
    r = client.get("/pending")
    assert r.status_code == 200
    assert r.json() == {}


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------

def test_cancel_unknown_session_returns_404(client):
    r = client.post("/cancel", json={"session_id": "no-such-session"})
    assert r.status_code == 404


def test_cancel_removes_session(client):
    sid = _open_session(client)
    mgr = client.app.state.session_manager
    assert mgr.get(sid) is not None
    r = client.post("/cancel", json={"session_id": sid})
    assert r.status_code == 200
    assert mgr.get(sid) is None


def test_cancel_validates_body(client):
    r = client.post("/cancel", json={})
    assert r.status_code == 422
    r = client.post("/cancel", json={"session_id": "x", "extra": True})
    assert r.status_code == 422
