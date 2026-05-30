"""Tests for POST /support/wireguard — the LAN-only "Enable Remote Support"
action that installs/sets-up/restarts wireguard-support.service on the host
AND verifies the tunnel actually came up.

SECURITY-CRITICAL. Two independent gates, checked IN ORDER:
  1. Custom header  X-Fula-Support: enable  — a custom header forces a CORS
     preflight in every modern browser, so a drive-by page on the same LAN
     can't trigger this with a simple cross-origin POST. The core BLE proxy
     can't send custom headers either, which is *why* this endpoint is
     deliberately LAN-only; the BLE "SUPPORT ON" button covers the BLE path.
  2. The tier-3 security code (same on-device file the action executor
     reads), so enabling a remote-support tunnel is a deliberate,
     authenticated act.

Only when BOTH gates pass do we shell out. The endpoint does NOT blindly
`systemctl restart` and trust the exit code — `wireguard-support.service` is
`Type=oneshot RemainAfterExit=yes`, so `systemctl is-active` lies. Instead it
orchestrates, in order:
    status.sh (pre)  →  [install.sh if not installed]  →  reset-failed
                     →  restart  →  status.sh (post, the GROUND TRUTH)

Every subprocess is mocked — we can't (and must not) touch WireGuard in
CI/dev. The behaviour under test is the gate logic + the orchestration
contract + how the verified post-state maps to the HTTP response, not systemd
itself. The `_run` stub is command-aware (see FakeRun): status.sh calls return
a configurable parsed-status JSON; install/systemctl calls return configurable
results — so a single test can drive any branch of the lifecycle.

The `client` fixture (tests/conftest.py) stages a security-code file
containing "1234" and points BLOX_AI_SECURITY_CODE_PATH at it.
"""
from __future__ import annotations

import json

import pytest

import src.routes.support as support_mod


ENABLE = {"X-Fula-Support": "enable"}
GOOD_CODE = {"security_code": "1234"}


# ---------------------------------------------------------------------------
# Command-aware _run stub
# ---------------------------------------------------------------------------

def _ok(stdout: str = "") -> dict:
    return {"success": True, "exit_code": 0,
            "stdout_excerpt": stdout, "stderr_excerpt": ""}


def _fail(exit_code: int = 1, stderr: str = "boom") -> dict:
    return {"success": False, "exit_code": exit_code,
            "stdout_excerpt": "", "stderr_excerpt": stderr}


def _status_json(**overrides) -> dict:
    """A representative status.sh payload (all fields healthy by default).
    Override `installed`/`registered`/`active` to drive lifecycle branches."""
    base = {
        "installed": True,
        "registered": True,
        "active": True,
        "endpoint": "203.0.113.7:51820",
        "assigned_ip": "10.13.13.2",
        "peer_id_registered": True,
        "last_handshake_age_sec": 11,
        "rx_bytes": 4096,
        "tx_bytes": 2048,
        "persistent_keepalive_sec": 25,
    }
    base.update(overrides)
    return base


class FakeRun:
    """Async stand-in for support._run, routed by command:

      * status.sh  → pops the next item from `status_results`; a dict is
        emitted as status.sh stdout (success), a None simulates status.sh
        failing to run / unparseable output (so _wg_status() returns None).
      * install.sh → returns `install_result`.
      * systemctl  → returns `systemctl_result` (covers both reset-failed and
        restart; reset-failed's result is ignored by the route, restart's is
        what matters).

    Records every (cmd, timeout) for ordering/argument assertions.
    """

    def __init__(self, status_results, install_result=None, systemctl_result=None):
        self._status_results = list(status_results)
        self._install_result = install_result if install_result is not None else _ok()
        self._systemctl_result = (
            systemctl_result if systemctl_result is not None else _ok()
        )
        self.calls: list[tuple[list[str], float]] = []

    async def __call__(self, cmd, timeout):
        self.calls.append((cmd, timeout))
        if support_mod.WG_STATUS_SCRIPT in cmd:
            nxt = self._status_results.pop(0) if self._status_results else None
            if nxt is None:
                return _fail(stderr="status.sh unavailable")
            return _ok(stdout=json.dumps(nxt))
        if support_mod.WG_INSTALL_SCRIPT in cmd:
            return self._install_result
        return self._systemctl_result  # systemctl reset-failed / restart

    # -- convenience views for assertions ---------------------------------
    def _of_kind(self, predicate):
        return [c for c in self.calls if predicate(c[0])]

    @property
    def status_calls(self):
        return self._of_kind(lambda cmd: support_mod.WG_STATUS_SCRIPT in cmd)

    @property
    def install_calls(self):
        return self._of_kind(lambda cmd: support_mod.WG_INSTALL_SCRIPT in cmd)

    @property
    def restart_calls(self):
        return self._of_kind(lambda cmd: "restart" in cmd)

    @property
    def reset_failed_calls(self):
        return self._of_kind(lambda cmd: "reset-failed" in cmd)


def _install_fake(monkeypatch, *, status_results,
                  install_result=None, systemctl_result=None) -> FakeRun:
    fake = FakeRun(status_results, install_result, systemctl_result)
    monkeypatch.setattr(support_mod, "_run", fake)
    return fake


@pytest.fixture
def fake_run(monkeypatch) -> FakeRun:
    """Healthy default: status.sh reports installed+active both pre and post;
    install/systemctl succeed. Gate tests assert NO calls happened; the
    happy-path tests inspect the recorded calls."""
    return _install_fake(
        monkeypatch, status_results=[_status_json(), _status_json()],
    )


# ---------------------------------------------------------------------------
# Gate 1 — custom header
# ---------------------------------------------------------------------------

def test_missing_header_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "support_header_required"}
    assert fake_run.calls == [], "subprocess MUST NOT run when the header gate fails"


def test_wrong_header_value_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE,
                    headers={"X-Fula-Support": "yes-please"})
    assert r.status_code == 403
    assert r.json()["error"] == "support_header_required"
    assert fake_run.calls == []


def test_header_match_is_case_and_whitespace_tolerant(client, fake_run):
    """Header compare is .strip().lower() == 'enable', so a padded/upper
    value still passes gate 1 (and, with a good code + healthy status,
    restarts and verifies → 200)."""
    r = client.post("/support/wireguard", json=GOOD_CODE,
                    headers={"X-Fula-Support": "  ENABLE  "})
    assert r.status_code == 200
    assert len(fake_run.restart_calls) == 1


# ---------------------------------------------------------------------------
# Gate 2 — security code
# ---------------------------------------------------------------------------

def test_wrong_security_code_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json={"security_code": "0000"},
                    headers=ENABLE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "security_code_invalid"}
    assert fake_run.calls == [], "subprocess MUST NOT run on a bad code"


def test_missing_security_code_is_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", json={}, headers=ENABLE)
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run.calls == []


def test_empty_body_is_tolerated_and_rejected_403_not_500(client, fake_run):
    """A POST with no body at all must be tolerated (parsed as {}) and
    rejected cleanly on the code gate — never a 500 from a json-parse
    exception."""
    r = client.post("/support/wireguard", headers=ENABLE)
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run.calls == []


def test_malformed_json_body_is_tolerated_and_rejected_403(client, fake_run):
    r = client.post("/support/wireguard", content=b"{not json",
                    headers={**ENABLE, "Content-Type": "application/json"})
    assert r.status_code == 403
    assert r.json()["error"] == "security_code_invalid"
    assert fake_run.calls == []


def test_missing_security_code_file_is_distinct_403(client, fake_run, monkeypatch):
    """If the on-device security-code file is missing entirely,
    read_security_code() returns None → a DISTINCT error
    (security_code_file_missing), still 403, still no subprocess. This lets
    the app tell 'wrong code' apart from 'device not provisioned'."""
    monkeypatch.setattr(support_mod, "read_security_code", lambda: None)
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 403
    assert r.json() == {"success": False, "error": "security_code_file_missing"}
    assert fake_run.calls == []


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
    assert fake_run.calls == []


# ---------------------------------------------------------------------------
# Both gates pass — full lifecycle, happy path
# ---------------------------------------------------------------------------

def test_happy_path_restarts_verifies_and_returns_200(client, fake_run):
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["exit_code"] == 0
    # The verified post-status is surfaced so the app can show tunnel details.
    assert body["status"]["active"] is True
    assert body["installed_on_demand"] is False
    # The restart ran exactly once, via nsenter into PID 1, against the
    # support unit, with the documented (bumped) timeout.
    assert len(fake_run.restart_calls) == 1
    cmd, timeout = fake_run.restart_calls[0]
    assert cmd[0] == "nsenter"
    assert "systemctl" in cmd
    assert support_mod.WG_SUPPORT_UNIT in cmd
    assert timeout == support_mod.RESTART_TIMEOUT_S
    # Both a pre-check and a post-check status.sh ran (verification is real).
    assert len(fake_run.status_calls) == 2
    # An already-installed device is NOT re-installed.
    assert fake_run.install_calls == []


def test_lifecycle_call_order_is_status_resetfailed_restart_status(client, fake_run):
    """Contract: pre-status → reset-failed → restart → post-status, in that
    exact order. reset-failed must precede restart (it clears a latched
    start-limit lock that would otherwise make the restart a no-op)."""
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 200
    kinds = []
    for cmd, _ in fake_run.calls:
        if support_mod.WG_STATUS_SCRIPT in cmd:
            kinds.append("status")
        elif "reset-failed" in cmd:
            kinds.append("reset-failed")
        elif "restart" in cmd:
            kinds.append("restart")
        else:
            kinds.append("other")
    assert kinds == ["status", "reset-failed", "restart", "status"]


# ---------------------------------------------------------------------------
# Verified-inactive — restart ran but the interface never came up
# ---------------------------------------------------------------------------

def test_tunnel_inactive_after_restart_returns_500(client, monkeypatch):
    """status.sh reports active=False AFTER the restart (the exact lie
    `systemctl is-active` would hide). The endpoint must surface a distinct
    500 so the app can say 'we tried but the tunnel didn't come up'."""
    fake = _install_fake(
        monkeypatch,
        status_results=[_status_json(active=False), _status_json(active=False)],
    )
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "tunnel_inactive_after_restart"
    assert body["status"]["active"] is False
    # We DID attempt the restart before declaring failure.
    assert len(fake.restart_calls) == 1


# ---------------------------------------------------------------------------
# Install-on-demand — device not yet set up
# ---------------------------------------------------------------------------

def test_not_installed_triggers_install_then_succeeds(client, monkeypatch):
    """Pre-status says installed=False → run install.sh on demand, THEN
    restart, THEN verify active. 200 with installed_on_demand=True."""
    fake = _install_fake(
        monkeypatch,
        status_results=[_status_json(installed=False, active=False),
                        _status_json(active=True)],
        install_result=_ok(),
    )
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["installed_on_demand"] is True
    assert len(fake.install_calls) == 1
    # install.sh ran with its generous cold-apt timeout, via nsenter.
    cmd, timeout = fake.install_calls[0]
    assert cmd[0] == "nsenter"
    assert timeout == support_mod.INSTALL_TIMEOUT_S
    # install precedes the restart.
    assert len(fake.restart_calls) == 1


def test_install_failure_returns_500_wireguard_not_installed(client, monkeypatch):
    """If install.sh fails, bail with a DISTINCT 500 (wireguard_not_installed)
    and DO NOT attempt a restart — a restart can't possibly succeed without
    the wg binary/keys."""
    fake = _install_fake(
        monkeypatch,
        status_results=[_status_json(installed=False, active=False)],
        install_result=_fail(exit_code=2, stderr="apt-get: wireguard-tools unavailable"),
    )
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "wireguard_not_installed"
    assert body["installed_on_demand"] is True
    assert body["exit_code"] == 2
    assert len(fake.install_calls) == 1
    assert fake.restart_calls == [], "must NOT restart when install failed"


# ---------------------------------------------------------------------------
# status.sh unavailable — graceful fallback to the restart exit code, so a
# transient status-check hiccup never regresses a working restart to a 500
# (nor reports success on a failed one).
# ---------------------------------------------------------------------------

def test_status_unavailable_falls_back_to_restart_success(client, monkeypatch):
    """status.sh can't be read pre OR post (returns None). With the restart
    succeeding, fall back to its exit code → 200, status=None. Install is
    NOT triggered (None != explicit installed=False)."""
    fake = _install_fake(
        monkeypatch, status_results=[None, None], systemctl_result=_ok(),
    )
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["status"] is None
    assert body["installed_on_demand"] is False
    assert fake.install_calls == [], "unknown status must NOT trigger install"
    assert len(fake.restart_calls) == 1


def test_status_unavailable_and_restart_fails_returns_500(client, monkeypatch):
    """status.sh unavailable AND the restart itself errors → fall back to the
    restart exit code → 500 with the captured stderr surfaced to the app."""
    fake = _install_fake(
        monkeypatch, status_results=[None, None],
        systemctl_result=_fail(exit_code=1, stderr="Job for wg failed"),
    )
    r = client.post("/support/wireguard", json=GOOD_CODE, headers=ENABLE)
    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["exit_code"] == 1
    assert body["stderr_excerpt"] == "Job for wg failed"
    assert body["status"] is None
    assert len(fake.restart_calls) == 1
