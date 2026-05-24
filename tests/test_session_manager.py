"""C5 — SessionManager unit tests.

Coverage:
  - create + get round-trip
  - TTL expiry (manipulate time.monotonic via patch)
  - touch slides TTL
  - LRU eviction when at cap
  - count() ignores expired
  - sanitize_for_log redacts IP + BSSID patterns
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.session.manager import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_TTL_SEC,
    SessionManager,
    sanitize_for_log,
)


def test_create_returns_session_with_uuid_id():
    mgr = SessionManager()
    s = mgr.create()
    import re
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        s.session_id,
    )


def test_create_honors_caller_supplied_id():
    mgr = SessionManager()
    s = mgr.create(session_id="caller-id")
    assert s.session_id == "caller-id"


def test_get_returns_known_session():
    mgr = SessionManager()
    s = mgr.create()
    assert mgr.get(s.session_id) is s


def test_get_returns_none_for_unknown():
    mgr = SessionManager()
    assert mgr.get("never-existed") is None


def test_get_returns_none_for_expired():
    mgr = SessionManager(ttl_sec=10)
    base = 1000.0
    with patch("src.session.manager.time.monotonic", return_value=base):
        s = mgr.create()
    # 11s later → expired
    with patch("src.session.manager.time.monotonic", return_value=base + 11):
        assert mgr.get(s.session_id) is None
    # And the entry should be purged
    assert s.session_id not in mgr._sessions


def test_touch_slides_ttl():
    mgr = SessionManager(ttl_sec=10)
    base = 1000.0
    with patch("src.session.manager.time.monotonic", return_value=base):
        s = mgr.create()
    # 8s later: touch
    with patch("src.session.manager.time.monotonic", return_value=base + 8):
        assert mgr.touch(s.session_id) is True
    # 8s after the touch (total 16s from create, would be expired without
    # the touch): still alive
    with patch("src.session.manager.time.monotonic", return_value=base + 16):
        assert mgr.get(s.session_id) is not None


def test_touch_returns_false_for_unknown():
    mgr = SessionManager()
    assert mgr.touch("never-existed") is False


def test_touch_returns_false_for_expired():
    mgr = SessionManager(ttl_sec=5)
    base = 1000.0
    with patch("src.session.manager.time.monotonic", return_value=base):
        s = mgr.create()
    with patch("src.session.manager.time.monotonic", return_value=base + 100):
        assert mgr.touch(s.session_id) is False


def test_lru_eviction_at_cap():
    mgr = SessionManager(max_sessions=3)
    base = 1000.0
    sessions = []
    with patch("src.session.manager.time.monotonic", return_value=base):
        sessions.append(mgr.create())
    with patch("src.session.manager.time.monotonic", return_value=base + 1):
        sessions.append(mgr.create())
    with patch("src.session.manager.time.monotonic", return_value=base + 2):
        sessions.append(mgr.create())
    # Bump session[1]'s last_active so session[0] is the LRU
    with patch("src.session.manager.time.monotonic", return_value=base + 10):
        mgr.touch(sessions[1].session_id)
    # Creating a 4th must evict session[0] (oldest last_active)
    with patch("src.session.manager.time.monotonic", return_value=base + 20):
        mgr.create()
        # Verify state from INSIDE the patch so the get() calls also use
        # the patched monotonic (real time would expire everything via TTL)
        assert mgr.get(sessions[0].session_id) is None
        assert mgr.get(sessions[1].session_id) is not None
        assert mgr.get(sessions[2].session_id) is not None


def test_remove_pops_session():
    mgr = SessionManager()
    s = mgr.create()
    assert mgr.remove(s.session_id) is True
    assert mgr.remove(s.session_id) is False  # idempotent-ish
    assert mgr.get(s.session_id) is None


def test_count_excludes_expired():
    mgr = SessionManager(ttl_sec=10)
    base = 1000.0
    with patch("src.session.manager.time.monotonic", return_value=base):
        mgr.create()
        mgr.create()
        mgr.create()
    with patch("src.session.manager.time.monotonic", return_value=base + 100):
        # All 3 expired
        assert mgr.count() == 0


def test_defaults_match_plan():
    """Phase 11 spec: 30min TTL, 50 cap."""
    assert DEFAULT_TTL_SEC == 30 * 60
    assert DEFAULT_MAX_SESSIONS == 50


# ---------------------------------------------------------------------------
# sanitize_for_log
# ---------------------------------------------------------------------------

def test_sanitize_for_log_redacts_ipv4():
    assert sanitize_for_log("peer at 192.168.1.5 failed") == "peer at <ip> failed"
    assert sanitize_for_log("10.0.0.1 and 10.0.0.2") == "<ip> and <ip>"


def test_sanitize_for_log_redacts_bssid():
    assert sanitize_for_log("AP aa:bb:cc:dd:ee:ff") == "AP <bssid>"
    assert sanitize_for_log("MAC DE-AD-BE-EF-12-34") == "MAC <bssid>"


def test_sanitize_for_log_truncates():
    long = "x" * 500
    out = sanitize_for_log(long, max_len=50)
    assert len(out) == 50
    assert out.endswith("…")


def test_sanitize_for_log_leaves_clean_string_alone():
    assert sanitize_for_log("nothing to redact here") == "nothing to redact here"
