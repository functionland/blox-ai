"""Tests for POST /support/wireguard — the LAN-only "Enable Remote Support"
action that restarts wireguard-support.service on the host.

SECURITY-CRITICAL. Two independent gates, checked IN ORDER:
  1. Custom header  X-Fula-Support: enable  — a custom header forces a CORS
     preflight in every modern browser, so a drive-by page on the same LAN
     can't trigger this with a simple cross-origin POST. The core BLE proxy
     can't send custom headers either, which is *why* this endpoint is
     deliberately LAN-only; the BLE "SUPPORT ON" button covers the BLE path.
  2. The tier-3 security code (same on-device file the action executor
     reads), so enabling a remote-support tunnel is a deliberate,
     authenticated act.

Only when BOTH gates pass do we shell out (nsenter → systemctl restart). The
subprocess is mocked in every test — we can't (and must not) restart
WireGuard in CI/dev, and the behaviour under test is the gate logic + the
shell-out contract, not systemd itself.

The `client` fixture (tests/conftest.py) stages a security-code file
containing "1234" and points BLOX_AI_SECURITY_CODE_PATH at it.
"""
from __future__ import annotations

import pytest

import src.routes.support as support_mod


ENABLE = {"X-Fula-Support": "enable"}
GOOD_CODE = {"security_code": "1234"}


@pytest.fixture
def fake_run(monkeypatch):
    """Replace support._run with an async stub so no real nsenter/systemctl
    is invoked. Captures the (cmd, timeout) it was called with so tests can
    assert the shell-out happened — or, crucially, did NOT happen when a
    gate rejects the request."""
    calls: list[tuple[list[str], float]] = []

    async def _fake(cmd, timeout):
        calls.append((cmd, timeout))
        return {"success": True, "exit_code": 0,
                "stdout_excerpt": "", "stderr_excerpt": ""}

    monkeypatch.setattr(support_mod, "_run", _fake)
    return calls


# ---------------------------------------------------------------------------
# Gate 1 — custom header
# ---------------------------------------------------------------------------

def test_missing_header_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "support_header_required"}
    assert fake_run == [], "subprocess MUST NOT run when the header gate fails"


def test_wrong_header_value_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE,
                    headers={"X-Fula-Support": "yes-please"})
    assert r.status_code == 403
    assert r.json()["error"] == "support_header_required"
    assert fake_run == []


def test_header_match_is_case_and_whitespace_tolerant(client, fake_run):
    """Header compare is .strip().lower() == 'enable', so a padded/upper
    value still passes gate 1 (and, with a good code, restarts)."""
    r = client.post("/support/wireguard", json=GOOD_CODE,
                    headers={"X-Fula-Support": "  ENABLE  "})
    assert r.status_code == 200
    assert len(fake_run) == 1


# ---------------------------------------------------------------------------
# Gate 2 — security code
# ---------------------------------------------------------------------------

def test_wrong_security_code_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json={"security_code": "0000"},
                    headers=ENABLE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "security_code_invalid"}
    assert fake_run == [], "subprocess MUST NOT run on a bad code"


def test_missing_security_code_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json={}, headers=ENABLE)
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run == []


def test_empty_body_is_tolerated_and_rejected_403_not_500(client, fake_run):
    """A POST with no body at all must be tolerated (parsed as {}) and
    rejected cleanly on the code gate — never a 500 from a json-parse
    exception."""
    r = client.post("/support/wireguard", headers=ENABLE)
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run == []


def test_malformed_json_body_is_tolerated_and_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", content=b"{not json",
                    headers={**ENABLE, "Content-Type": "application/json"})
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run == []


def test_missing_security_code_file_is_distinct_403(client, fake_run, monkeypatch):
    """If the on-device security-code file is missing entirely,
    read_security_code() returns None → a DISTINCT error
    (security_code_file_missing), still 403, still no subprocess. This lets
    the app tell 'wrong code' apart from 'device not provisioned'."""
    monkeypatch.setattr(support_mod, "read_security_code", lambda: None)
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "security_code_file_missing"}
    assert fake_run == []


# ---------------------------------------------------------------------------
# Gate ordering — header is checked before the code (don't even read the
# code if the header is absent; don't leak that a code was supplied)
# ---------------------------------------------------------------------------

def test_header_gate_fires_before_security_code_gate(client, fake_run):
    """A request with a VALID code but NO header still returns
    support_header_required, not security_code_invalid."""
    r = client.post("/support/wireguard", json=GOOD_CODE)
    assert r.status_code == 403
    assert r.json()["error"] == "support_header_required"
    assert fake_run == []


# ---------------------------------------------------------------------------
# Both gates pass — the restart actually runs
# ---------------------------------------------------------------------------

def test_happy_path_restarts_and_returns_200(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["exit_code"] == 0
    # The shell-out ran exactly once, against the support unit, via nsenter
    # into PID 1, with the documented timeout.
    assert len(fake_run) == 1
    cmd, timeout = fake_run[0]
    assert cmd[0] == "nsenter"
    assert "systemctl" in cmd
    assert "restart" in cmd
    assert support_mod.WG_SUPPORT_UNIT in cmd
    assert timeout == support_mod.RESTART_TIMEOUT_S


def test_restart_failure_returns_500_with_captured_result(client, monkeypatch):
    """If systemctl restart exits non-zero, the endpoint surfaces 500 with
    the captured stdout/stderr excerpts so the app can show the user why."""
    async def _fail(cmd, timeout):
        return {"success": False, "exit_code": 1,
                "stdout_excerpt": "", "stderr_excerpt": "Job for wg failed"}

    monkeypatch.setattr(support_mod, "_run", _fail)
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["exit_code"] == 1
    assert body["stderr_excerpt"] == "Job for wg failed"
