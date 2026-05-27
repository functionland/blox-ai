"""C5 — conversational endpoint tests.

Coverage:
  - POST /troubleshoot/user-reply 404 on unknown session
  - POST /troubleshoot/user-reply 400 on question_id mismatch
  - POST /troubleshoot/phone-context 404 on unknown session
  - POST /troubleshoot/phone-context 400 on invalid phone_context (no PII echo)
  - POST /troubleshoot/phone-context 200 on valid + attaches to session
  - End-to-end: prompt with "ask" triggers user_question; reply unblocks
    the SSE stream and emits user_reply_received with matching question_id
"""
from __future__ import annotations

import json
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# user-reply endpoint shape
# ---------------------------------------------------------------------------

def test_user_reply_unknown_session_returns_404(client):
    r = client.post("/troubleshoot/user-reply", json={
        "session_id": "no-such-session",
        "question_id": "q1",
        "reply_text": "hi",
    })
    assert r.status_code == 404
    assert r.json() == {"error": "session_not_found"}


def test_user_reply_no_pending_question_returns_400(client):
    # Create a session via /troubleshoot but don't trigger a user_question
    r0 = client.post("/troubleshoot", json={"prompt": "just diagnose"})
    assert r0.status_code == 200
    # Extract session_id from the first SSE event
    first_line = r0.text.split("\n\n")[0]
    payload = json.loads(first_line[len("data: "):])
    sid = payload["session_id"]
    # Now post a user-reply with no pending question
    r = client.post("/troubleshoot/user-reply", json={
        "session_id": sid,
        "question_id": "q1",
        "reply_text": "anything",
    })
    assert r.status_code == 400
    assert r.json()["error"] == "question_id_mismatch"


def test_user_reply_validates_body_shape(client):
    # Missing required field
    r = client.post("/troubleshoot/user-reply", json={"session_id": "x"})
    assert r.status_code == 422


def test_user_reply_rejects_extra_fields(client):
    r = client.post("/troubleshoot/user-reply", json={
        "session_id": "x", "question_id": "q", "reply_text": "y",
        "surprise": True,
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# phone-context endpoint shape
# ---------------------------------------------------------------------------

def _good_phone_context() -> dict:
    return {
        "app_version": "1.0.0",
        "os": "android",
        "os_version": "14",
        "device_model": "Pixel 7",
    }


def test_phone_context_unknown_session_auto_creates(client):
    """Behavior change 2026-05-27: phone-context with an unknown
    session_id now AUTO-CREATES the session (matches the prime-then-
    troubleshoot pattern + survives container restart). Previously
    returned 404 with `session_not_found` and the phone app showed
    "[http-not-found] Session not found" to the user.

    Real-world cases this unblocks:
      - User taps "Share my phone's context" before starting
        /troubleshoot — the app's pre-generated session_id should
        prime the session; the next /troubleshoot with that same
        session_id finds the session with phone_context already
        attached.
      - Container restarted between sessions; the app's cached
        session_id is no longer in the in-memory map. Auto-create
        avoids a confusing "Session not found" error in the UI.
    """
    mgr = client.app.state.session_manager
    assert mgr.get("nope") is None
    r = client.post("/troubleshoot/phone-context", json={
        "session_id": "nope",
        "phone_context": _good_phone_context(),
    })
    assert r.status_code == 200
    # Session was created with the supplied id + phone_context attached
    s = mgr.get("nope")
    assert s is not None
    assert s.phone_context == _good_phone_context()


def test_phone_context_validates_body_envelope(client):
    r = client.post("/troubleshoot/phone-context", json={
        "session_id": "x", "phone_context": {}, "extra": True,
    })
    assert r.status_code == 422


def _open_session(client) -> str:
    """Helper: mint a session via a /troubleshoot call, return session_id."""
    r = client.post("/troubleshoot", json={"prompt": "just diagnose"})
    first = r.text.split("\n\n")[0]
    payload = json.loads(first[len("data: "):])
    return payload["session_id"]


def test_phone_context_happy_path_attaches_to_session(client):
    sid = _open_session(client)
    r = client.post("/troubleshoot/phone-context", json={
        "session_id": sid,
        "phone_context": _good_phone_context(),
    })
    assert r.status_code == 200
    assert r.json() == {}
    # Verify it was attached server-side (test_inspection — not part of
    # the public API but proves the path).
    mgr = client.app.state.session_manager
    s = mgr.get(sid)
    assert s.phone_context == _good_phone_context()


def test_phone_context_invalid_returns_generic_400(client):
    """Per Phase 11 privacy contract: validation error must NOT echo
    the raw values back to the client."""
    sid = _open_session(client)
    bad = _good_phone_context()
    bad["os"] = "windowsphone"  # not in the enum
    # Stuff a fake SSID + IP to verify they don't leak
    bad["netinfo"] = {
        "wifi_ssid": "SecretWiFi-CompanyName",
        "is_connected": True,
        "type": "wifi",
    }
    bad["unknown_field"] = "leaky"  # additionalProperties:false
    r = client.post("/troubleshoot/phone-context", json={
        "session_id": sid,
        "phone_context": bad,
    })
    assert r.status_code == 400
    text = r.text
    # No PII in response body
    assert "SecretWiFi" not in text
    assert "CompanyName" not in text
    # Error envelope is generic
    assert "phone_context_invalid" in text


def test_phone_context_replaces_prior_snapshot(client):
    sid = _open_session(client)
    first = _good_phone_context()
    second = {**_good_phone_context(), "device_model": "Newer Pixel"}
    client.post("/troubleshoot/phone-context", json={
        "session_id": sid, "phone_context": first,
    })
    client.post("/troubleshoot/phone-context", json={
        "session_id": sid, "phone_context": second,
    })
    mgr = client.app.state.session_manager
    s = mgr.get(sid)
    assert s.phone_context == second  # latest replaces


# ---------------------------------------------------------------------------
# End-to-end: user_question pauses SSE; /user-reply unblocks
# ---------------------------------------------------------------------------

def test_user_question_event_emitted_when_prompt_asks(client):
    """The mock backend's `ask` branch emits a user_question. The bridge
    PAUSES; /user-reply unblocks. TestClient streams synchronously, so
    we drive the reply from a sibling thread before the SSE response is
    fully consumed."""

    sid_holder: dict = {}
    reply_done: dict = {"ok": False}

    def post_reply_after_delay():
        # Poll for the session to appear in the manager + have a pending
        # question, then post the reply. This races correctly against
        # the bridge's await_for(reply_queue).
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            sid = sid_holder.get("sid")
            if sid:
                s = client.app.state.session_manager.get(sid)
                if s and s.pending_question_id:
                    r = client.post("/troubleshoot/user-reply", json={
                        "session_id": sid,
                        "question_id": s.pending_question_id,
                        "reply_text": "yesterday",
                    })
                    reply_done["ok"] = r.status_code == 200
                    return
            time.sleep(0.05)

    # We need to know the session_id BEFORE the SSE generator pauses on
    # user_question. The session_started event is the first one emitted,
    # so we can extract it from a partial read — but TestClient buffers.
    # Workaround: pre-create a session via SessionManager and pass its id.
    pre_session = client.app.state.session_manager.create()
    sid_holder["sid"] = pre_session.session_id

    t = threading.Thread(target=post_reply_after_delay, daemon=True)
    t.start()

    r = client.post("/troubleshoot", json={
        "prompt": "please ask me something first",
        "session_id": pre_session.session_id,
    })
    t.join(timeout=35)

    assert r.status_code == 200
    assert reply_done["ok"], "reply post never succeeded"
    # The stream should contain user_question + user_reply_received
    events = []
    for raw in r.text.split("\n\n"):
        raw = raw.strip()
        if raw.startswith("data: "):
            events.append(json.loads(raw[len("data: "):]))
    types = [e["type"] for e in events]
    assert "user_question" in types, (
        f"expected user_question event; got {types}"
    )
    assert "user_reply_received" in types, (
        f"expected user_reply_received event; got {types}"
    )
    # Ordering: user_question MUST come before user_reply_received
    assert types.index("user_question") < types.index("user_reply_received")
    # The user_reply_received's question_id matches the user_question's
    uq = next(e for e in events if e["type"] == "user_question")
    urr = next(e for e in events if e["type"] == "user_reply_received")
    assert uq["question_id"] == urr["question_id"]


def test_user_reply_with_wrong_question_id_rejects(client):
    """During a paused stream, an /user-reply with a non-matching
    question_id MUST be rejected with 400 (and the stream stays paused
    waiting for the right id)."""
    pre_session = client.app.state.session_manager.create()
    # We can't easily run an actual SSE stream here (it'd block), so
    # simulate the paused state by setting pending_question_id directly.
    pre_session.pending_question_id = "real-q-id"

    r = client.post("/troubleshoot/user-reply", json={
        "session_id": pre_session.session_id,
        "question_id": "wrong-q-id",
        "reply_text": "anything",
    })
    assert r.status_code == 400
    assert r.json()["error"] == "question_id_mismatch"
    # Pending unchanged
    assert pre_session.pending_question_id == "real-q-id"
