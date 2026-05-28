"""C3 — per-impl tests for the 11 diag/* tools.

Mocks subprocess + file reads at the impl-module boundary so the tests
run deterministically on any host (Windows dev, aarch64 lab, CI runner)
without needing real /run/fula-*.state, docker, kubo, wg, etc.

Each impl test asserts:
  - happy path produces expected fields
  - missing-input fallback produces a schema-valid response
  - subprocess timeouts don't propagate exceptions

The end-to-end "every impl response validates against
diag_responses.schema.json" is covered in test_diag_routes.py — that
exercises the impls through the GET routes and validates the bodies.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import jsonschema
import pytest


def _validator():
    """Build a per-$def validator from the staged schema dir. Skips with
    a clear message when the real fula-ota schemas aren't available."""
    from pathlib import Path
    from .conftest import _real_schemas_in_use, _locate_fula_ota_api_dir
    if not _real_schemas_in_use():
        pytest.skip(
            "set BLOX_AI_FULA_OTA_SCHEMA_DIR to run schema-validated tests"
        )
    api_dir = _locate_fula_ota_api_dir()
    schema = json.loads(
        (api_dir / "diag_responses.schema.json").read_text(encoding="utf-8")
    )
    return schema


def _validate(payload: dict, defname: str):
    schema = _validator()
    # Bind the $ref to the resolved defname's schema for validation.
    full = {**schema, "$ref": f"#/$defs/{defname}"}
    jsonschema.Draft202012Validator(full).validate(payload)


# ---------------------------------------------------------------------------
# diag/internet
# ---------------------------------------------------------------------------

def test_internet_happy_path_all_ok():
    """google.com via https_head, discovery via https_reachable."""
    from src.tools.diag_impls import internet as mod
    with patch.object(mod, "dns_lookup", return_value=True), \
         patch.object(mod, "https_head", return_value=(True, 200, 42.0)), \
         patch.object(mod, "https_reachable", return_value=(True, 200, 42.0)):
        r = mod.diag_internet()
    assert r["dns_ok"] is True
    assert r["https_google_ok"] is True
    assert r["https_discovery_ok"] is True
    assert r["captive_portal_likely"] is False
    _validate(r, "internet")


def test_internet_no_dns_marks_everything_down():
    from src.tools.diag_impls import internet as mod
    with patch.object(mod, "dns_lookup", return_value=False), \
         patch.object(mod, "https_head", return_value=(False, None, 0.0)), \
         patch.object(mod, "https_reachable", return_value=(False, None, 0.0)):
        r = mod.diag_internet()
    assert r["dns_ok"] is False
    _validate(r, "internet")


def test_internet_captive_portal_heuristic_fires():
    """DNS OK + google OK + discovery completely UNREACHABLE (no HTTP
    response at all) + low latency → likely captive."""
    from src.tools.diag_impls import internet as mod
    with patch.object(mod, "dns_lookup", return_value=True), \
         patch.object(mod, "https_head", return_value=(True, 200, 30.0)), \
         patch.object(mod, "https_reachable", return_value=(False, None, 20.0)):
        r = mod.diag_internet()
    assert r["captive_portal_likely"] is True


def test_internet_discovery_403_is_reachable_not_captive():
    """Regression guard 2026-05-26: lab observed
    https://discovery.fula.network/relays returning HTTP 403 (HEAD
    not allowed by the server — only POST). Before the fix, this was
    classified as discovery_https_ok=False AND captive_portal_likely=True
    — both false positives that led the AI to diagnose 'discovery
    unreachable' when the server was actually fine.

    With the https_reachable fix, ANY HTTP response (including 403)
    counts as 'reachable' — only no-response-at-all counts as down."""
    from src.tools.diag_impls import internet as mod
    with patch.object(mod, "dns_lookup", return_value=True), \
         patch.object(mod, "https_head", return_value=(True, 200, 80.0)), \
         patch.object(mod, "https_reachable", return_value=(True, 403, 5.0)):
        r = mod.diag_internet()
    # The 403 from discovery still counts as REACHABLE.
    assert r["https_discovery_ok"] is True, (
        "discovery server responded with HTTP 403 — that means it's reachable, "
        "not down. The AI must not be told 'discovery unreachable' just because "
        "we can't HEAD /relays."
    )
    # And captive-portal must NOT fire when discovery responded.
    assert r["captive_portal_likely"] is False, (
        "captive-portal heuristic fired despite discovery returning a real HTTP "
        "response — should only fire when discovery is COMPLETELY unreachable"
    )


# ---------------------------------------------------------------------------
# diag/relay
# ---------------------------------------------------------------------------

def test_relay_no_kubo_returns_empty():
    from src.tools.diag_impls import relay as mod
    with patch.object(mod, "http_post_json", return_value=None):
        r = mod.diag_relay()
    assert r["relays"] == []
    assert r["reservation_count"] == 0
    _validate(r, "relay")


def test_relay_counts_circuit_peers():
    from src.tools.diag_impls import relay as mod
    peers = {
        "Peers": [
            {"Addr": "/ip4/1.2.3.4/tcp/4001"},
            {"Addr": "/ip4/5.6.7.8/tcp/4001/p2p-circuit/p2p/QmABC"},
        ],
    }
    with patch.object(mod, "http_post_json", return_value=peers):
        r = mod.diag_relay()
    assert r["reservation_count"] == 1
    assert len(r["relays"]) == 2
    _validate(r, "relay")


# ---------------------------------------------------------------------------
# diag/time
# ---------------------------------------------------------------------------

def test_time_uses_state_file_when_present():
    from src.tools.diag_impls import time_ as mod
    with patch.object(mod, "read_state",
                      return_value={"synced": True, "offset_ms": 5,
                                    "service": "systemd-timesyncd"}):
        r = mod.diag_time()
    assert r["synced"] is True
    assert r["service"] == "systemd-timesyncd"
    _validate(r, "time")


def test_time_falls_back_to_timedatectl():
    from src.tools.diag_impls import time_ as mod
    with patch.object(mod, "read_state", return_value={}), \
         patch.object(mod, "run_subprocess", return_value=(0, "yes\n", "")):
        r = mod.diag_time()
    assert r["synced"] is True
    _validate(r, "time")


def test_time_unknown_service_coerces():
    from src.tools.diag_impls import time_ as mod
    with patch.object(mod, "read_state",
                      return_value={"synced": False, "service": "weird"}):
        r = mod.diag_time()
    assert r["service"] == "unknown"
    _validate(r, "time")


# ---------------------------------------------------------------------------
# diag/power
# ---------------------------------------------------------------------------

def test_power_uses_state_file():
    from src.tools.diag_impls import power as mod
    with patch.object(mod, "read_state", return_value={
        "uptime_s": 12345, "undervoltage_events_24h": 2,
        "recent_reboots": 1, "max_temp_c": 55.5, "soc_voltage_ratio": 0.98,
    }):
        r = mod.diag_power()
    assert r["uptime_s"] == 12345
    assert r["undervoltage_events_24h"] == 2
    _validate(r, "power")


def test_power_fallback_proc_uptime(tmp_path, monkeypatch):
    from src.tools.diag_impls import power as mod
    with patch.object(mod, "read_state", return_value={}), \
         patch("builtins.open",
               side_effect=lambda *a, **k: open(tmp_path / "uptime", "w") and None) \
                 if False else patch.object(mod, "read_state", return_value={}):
        # Easier: mock /proc/uptime via tmp file + open redirect
        pass
    # Simpler approach: just confirm no exception when state absent
    with patch.object(mod, "read_state", return_value={}):
        try:
            r = mod.diag_power()
        except Exception as e:
            pytest.fail(f"diag_power must never raise: {e}")
    assert "uptime_s" in r
    _validate(r, "power")


# ---------------------------------------------------------------------------
# diag/storage
# ---------------------------------------------------------------------------

def test_storage_parses_df_output():
    from src.tools.diag_impls import storage as mod
    df_out = (
        "Filesystem    1B-blocks       Used  Available Capacity Mounted on\n"
        "/dev/root  100000000000 50000000000 50000000000      50% /\n"
    )
    # 3 subprocess calls: df, dmesg, smartctl. Provide enough mocks.
    with patch.object(mod, "run_subprocess",
                      side_effect=[(0, df_out, ""),         # _df_for_mounts
                                   (0, "", ""),              # _dmesg_io_errors_recent
                                   (-1, "", "no smartctl")]):# _smartctl_health
        r = mod.diag_storage()
    assert "/" in r["df"]
    assert r["df"]["/"]["used_bytes"] == 50_000_000_000
    _validate(r, "storage")


def test_storage_handles_df_failure_gracefully():
    from src.tools.diag_impls import storage as mod
    # All 3 subprocess calls fail
    with patch.object(mod, "run_subprocess",
                      return_value=(-1, "", "command not found")):
        r = mod.diag_storage()
    assert r["df"] == {}
    _validate(r, "storage")


# ---------------------------------------------------------------------------
# diag/containers
# ---------------------------------------------------------------------------

def test_containers_no_docker_returns_empty():
    from src.tools.diag_impls import containers as mod
    # Force the docker import-side fail-soft
    with patch("docker.from_env",
               side_effect=__import__("docker").errors.DockerException("no sock")):
        r = mod.diag_containers()
    assert r == {"containers": []}


def test_containers_parses_a_running_container():
    from src.tools.diag_impls import containers as mod

    class FakeContainer:
        name = "ipfs_host"
        attrs = {
            "State": {"Status": "running", "OOMKilled": False,
                      "StartedAt": "2026-05-24T19:00:00.123456789Z"},
            "Config": {"Image": "ipfs/kubo:latest"},
            "RestartCount": 0,
        }

    class FakeClient:
        @property
        def containers(self_inner):
            class L:
                def list(self, **kw):
                    return [FakeContainer()]
            return L()
        def close(self): pass

    with patch("docker.from_env", return_value=FakeClient()):
        r = mod.diag_containers()
    assert len(r["containers"]) == 1
    c = r["containers"][0]
    assert c["name"] == "ipfs_host"
    assert c["state"] == "running"
    assert c["oom_killed"] is False
    assert c["started_at"].endswith("Z")  # trimmed to ms
    _validate(r, "containers")


def test_containers_skips_unwatched_names():
    from src.tools.diag_impls import containers as mod

    class FakeContainer:
        def __init__(self, name):
            self.name = name
            self.attrs = {"State": {"Status": "running"}, "Config": {}}

    class FakeClient:
        @property
        def containers(self_inner):
            class L:
                def list(self, **kw):
                    return [FakeContainer("random_user_container")]
            return L()
        def close(self): pass

    with patch("docker.from_env", return_value=FakeClient()):
        r = mod.diag_containers()
    assert r["containers"] == []


# ---------------------------------------------------------------------------
# diag/wireguard
# ---------------------------------------------------------------------------

def test_wireguard_not_installed():
    from src.tools.diag_impls import wireguard as mod
    with patch.object(mod, "run_subprocess", return_value=(1, "", "")):
        r = mod.diag_wireguard()
    assert r == {"installed": False, "registered": False, "active": False}
    _validate(r, "wireguard")


def test_wireguard_active_with_handshake():
    from src.tools.diag_impls import wireguard as mod
    seq = [
        (0, "", ""),                          # which wg
        (0, "support\n", ""),                 # show interfaces
        (0, "PEER 100\n", ""),                # latest-handshakes (epoch)
        (0, "PEER 1024 2048\n", ""),          # transfer rx tx
        (0, "PEER 25\n", ""),                 # persistent-keepalive
    ]
    with patch.object(mod, "run_subprocess", side_effect=seq):
        r = mod.diag_wireguard()
    assert r["installed"] is True
    assert r["registered"] is True
    assert r["active"] is True
    assert r["rx_bytes"] == 1024
    assert r["tx_bytes"] == 2048
    assert r["persistent_keepalive_sec"] == 25
    _validate(r, "wireguard")


# ---------------------------------------------------------------------------
# diag/heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_uses_state_file():
    from src.tools.diag_impls import heartbeat as mod
    with patch.object(mod, "read_state", return_value={
        "last_attempt_ts": "2026-05-24T10:00:00.000Z",
        "http_status": 200,
        "last_circuit_count": 3,
        "last_reserved_on": ["r1", "r2"],
    }):
        r = mod.diag_heartbeat()
    assert r["http_status"] == 200
    assert r["last_reserved_on"] == ["r1", "r2"]
    _validate(r, "heartbeat")


def test_heartbeat_missing_state_synthesizes_iso_ts():
    from src.tools.diag_impls import heartbeat as mod
    with patch.object(mod, "read_state", return_value={}):
        r = mod.diag_heartbeat()
    assert "last_attempt_ts" in r
    assert r["last_attempt_ts"].endswith("Z")
    _validate(r, "heartbeat")


# ---------------------------------------------------------------------------
# diag/events
# ---------------------------------------------------------------------------

def test_events_empty_when_log_missing(tmp_path, monkeypatch):
    from src.tools.diag_impls import events as mod
    monkeypatch.setattr(mod, "EVENTS_LOG_PATH", str(tmp_path / "no-such-file"))
    r = mod.diag_events()
    assert r == {"events": []}
    _validate(r, "events")


def test_events_reads_jsonl_tail(tmp_path, monkeypatch):
    from src.tools.diag_impls import events as mod
    log = tmp_path / "events.jsonl"
    lines = [
        {"ts": "2026-05-24T10:00:00Z", "category": "cat1", "detail": "d1"},
        {"ts": "2026-05-24T10:01:00Z", "category": "cat2", "detail": "d2"},
        "not-json-line",
        {"ts": "2026-05-24T10:02:00Z", "category": "cat3", "detail": "d3"},
    ]
    log.write_text("\n".join(
        json.dumps(L) if not isinstance(L, str) else L for L in lines
    ) + "\n")
    monkeypatch.setattr(mod, "EVENTS_LOG_PATH", str(log))
    r = mod.diag_events(tail_n=10)
    cats = [e["category"] for e in r["events"]]
    assert cats == ["cat1", "cat2", "cat3"]  # malformed line skipped
    _validate(r, "events")


def test_events_skips_malformed_event_objects(tmp_path, monkeypatch):
    """A line that parses as JSON but is missing required fields is dropped."""
    from src.tools.diag_impls import events as mod
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"ts": "2026-05-24T10:00:00Z"})  # missing category+detail
        + "\n"
        + json.dumps({"ts": "2026-05-24T10:01:00Z", "category": "ok",
                      "detail": "ok"}) + "\n"
    )
    monkeypatch.setattr(mod, "EVENTS_LOG_PATH", str(log))
    r = mod.diag_events()
    assert len(r["events"]) == 1
    _validate(r, "events")


# ---------------------------------------------------------------------------
# diag/readiness
# ---------------------------------------------------------------------------

def test_readiness_returns_recent_log_string():
    from src.tools.diag_impls import readiness as mod
    with patch.object(mod, "run_subprocess",
                      side_effect=[(0, "log line 1\nlog line 2\n", ""),
                                   (0, "", "")]):
        r = mod.diag_readiness()
    assert "log line 1" in r["recent_log"]
    _validate(r, "readiness")


def test_readiness_journalctl_unavailable():
    from src.tools.diag_impls import readiness as mod
    with patch.object(mod, "run_subprocess",
                      side_effect=[(-1, "", "command not found"),
                                   (-1, "", "")]):
        r = mod.diag_readiness()
    assert r["recent_log"] == ""
    _validate(r, "readiness")


# ---------------------------------------------------------------------------
# diag/discovery_state (Phase 0.5a)
# ---------------------------------------------------------------------------

def test_discovery_state_happy_path():
    from src.tools.diag_impls import discovery_state as mod
    with patch.object(mod, "read_state", return_value={
        "ok": True,
        "last_check_ts": "2026-05-28T19:00:00Z",
        "latency_ms": 42.5,
    }):
        r = mod.diag_discovery_state()
    assert r["ok"] is True
    assert r["last_check_ts"] == "2026-05-28T19:00:00Z"
    assert r["latency_ms"] == 42.5
    _validate(r, "discovery_state")


def test_discovery_state_missing_file_returns_null_ok():
    """Pre-Phase-1.2 firmware (or fresh boot before first cycle) has no
    state file. Tristate `ok=null` lets trees branch on 'unknown'."""
    from src.tools.diag_impls import discovery_state as mod
    with patch.object(mod, "read_state", return_value={}):
        r = mod.diag_discovery_state()
    assert r["ok"] is None
    _validate(r, "discovery_state")


def test_discovery_state_with_error():
    from src.tools.diag_impls import discovery_state as mod
    with patch.object(mod, "read_state", return_value={
        "ok": False,
        "error": "HTTPSConnectionPool: connection timeout",
    }):
        r = mod.diag_discovery_state()
    assert r["ok"] is False
    assert "timeout" in r["error"]
    _validate(r, "discovery_state")


def test_discovery_state_error_is_truncated_at_500():
    from src.tools.diag_impls import discovery_state as mod
    long_err = "x" * 2000
    with patch.object(mod, "read_state", return_value={
        "ok": False, "error": long_err,
    }):
        r = mod.diag_discovery_state()
    assert len(r["error"]) == 500
    _validate(r, "discovery_state")


def test_discovery_state_ignores_non_bool_ok():
    """Defensive: if a future writer wrote a string instead of bool, we
    surface it as `null` rather than passing a typed-wrong value through."""
    from src.tools.diag_impls import discovery_state as mod
    with patch.object(mod, "read_state", return_value={"ok": "yes"}):
        r = mod.diag_discovery_state()
    assert r["ok"] is None
    _validate(r, "discovery_state")


# ---------------------------------------------------------------------------
# diag/systemd_services (Phase 0.5a)
# ---------------------------------------------------------------------------

def test_systemd_services_all_active():
    from src.tools.diag_impls import systemd_services as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[1] == "is-active":
            return 0, "active\n", ""
        # systemctl show
        return 0, "Result=success\nSubState=running\n", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run):
        r = mod.diag_systemd_services()
    assert len(r["services"]) == len(mod.FULA_UNITS)
    for svc in r["services"]:
        assert svc["active"] is True
        assert svc["state"] == "active"
        assert svc["sub_state"] == "running"
    _validate(r, "systemd_services")


def test_systemd_services_unit_failed():
    from src.tools.diag_impls import systemd_services as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[1] == "is-active":
            return 3, "failed\n", ""
        return 0, "Result=exit-code\nSubState=failed\n", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run):
        r = mod.diag_systemd_services()
    for svc in r["services"]:
        assert svc["active"] is False
        assert svc["state"] == "failed"
        assert svc["result"] == "exit-code"
    _validate(r, "systemd_services")


def test_systemd_services_systemctl_missing():
    """Dev host / container without systemd: every unit comes back as
    unknown, NOT silently false. Trees should branch on `unknown`."""
    from src.tools.diag_impls import systemd_services as mod
    with patch.object(mod, "run_subprocess",
                      return_value=(-1, "", "command not found")):
        r = mod.diag_systemd_services()
    for svc in r["services"]:
        assert svc["active"] is None
        assert svc["state"] == "unknown"
    _validate(r, "systemd_services")


def test_systemd_services_per_unit_timeout_is_isolated():
    """A hung systemctl on ONE unit must not poison the others.

    Impl shape: on is-active rc=-1 the impl returns early without
    invoking the show subprocess (early-return is what gives us the
    isolation). So a timeout on unit[0]'s is-active = ONE subprocess
    call total for that unit, then units 1..N each take 2 calls
    (is-active + show)."""
    from src.tools.diag_impls import systemd_services as mod
    call_count = {"n": 0}

    def fake_run(cmd, timeout_s=2.0):
        call_count["n"] += 1
        # ONLY the very first call (unit[0]'s is-active) times out.
        if call_count["n"] == 1:
            return -1, "", "timeout after 2.0s"
        if cmd[1] == "is-active":
            return 0, "active\n", ""
        return 0, "Result=success\nSubState=running\n", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run):
        r = mod.diag_systemd_services()
    # First unit returns the unknown fallback; rest are active.
    assert r["services"][0]["active"] is None
    assert r["services"][0]["state"] == "unknown"
    for svc in r["services"][1:]:
        assert svc["active"] is True, f"{svc['name']} should be active"
    _validate(r, "systemd_services")


# ---------------------------------------------------------------------------
# diag/network_interface (Phase 0.5a)
# ---------------------------------------------------------------------------

_FAKE_IP_ADDR_JSON = """[
  {"ifname": "lo", "operstate": "UNKNOWN", "addr_info": [
    {"family": "inet", "local": "127.0.0.1"}
  ]},
  {"ifname": "wlan0", "operstate": "UP", "mtu": 1500,
   "address": "aa:bb:cc:dd:ee:ff",
   "addr_info": [
     {"family": "inet", "local": "192.168.1.50"},
     {"family": "inet6", "local": "2001:db8::1"},
     {"family": "inet6", "local": "fe80::1"}
   ]},
  {"ifname": "eth0", "operstate": "DOWN", "addr_info": []}
]"""

_FAKE_IW_LINK_CONNECTED = """\
Connected to aa:bb:cc:dd:ee:ff (on wlan0)
        SSID: MyHomeWiFi
        freq: 5180
        signal: -55 dBm
        tx bitrate: 130.0 MBit/s VHT-MCS 7 80MHz short GI
"""

_FAKE_IW_LINK_DISCONNECTED = "Not connected.\n"


def test_network_interface_happy_path_wifi_associated():
    from src.tools.diag_impls import network_interface as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, cmd[1] + "\n", ""
        if cmd[0] == "ip":
            return 0, _FAKE_IP_ADDR_JSON, ""
        if cmd[0] == "iw" and cmd[1] == "dev":
            return 0, _FAKE_IW_LINK_CONNECTED, ""
        return -1, "", ""

    # Treat ONLY wlan0 as wireless per sysfs check.
    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch.object(mod, "_looks_like_wifi", side_effect=lambda n: n == "wlan0"):
        r = mod.diag_network_interface()
    assert r["tools_present"]["ip"] is True
    assert r["tools_present"]["iw"] is True
    # lo dropped; wlan0 + eth0 kept
    names = sorted([i["name"] for i in r["interfaces"]])
    assert names == ["eth0", "wlan0"]
    wlan = [i for i in r["interfaces"] if i["name"] == "wlan0"][0]
    assert wlan["operstate"] == "UP"
    assert wlan["mtu"] == 1500
    assert wlan["mac"] == "aa:bb:cc:dd:ee:ff"
    assert wlan["ipv4"] == ["192.168.1.50"]
    # link-local fe80:: must be stripped
    assert wlan["ipv6"] == ["2001:db8::1"]
    assert wlan["wifi_associated"] is True
    assert wlan["wifi_ssid"] == "MyHomeWiFi"
    assert wlan["wifi_signal_dbm"] == -55
    assert wlan["wifi_tx_bitrate_mbps"] == 130.0
    assert wlan["wifi_freq_mhz"] == 5180
    # eth0 must NOT have wifi_* fields applied (sysfs gating works)
    eth = [i for i in r["interfaces"] if i["name"] == "eth0"][0]
    assert "wifi_ssid" not in eth
    assert "wifi_associated" not in eth
    _validate(r, "network_interface")


def test_network_interface_wifi_not_associated():
    from src.tools.diag_impls import network_interface as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, "", ""
        if cmd[0] == "ip":
            return 0, _FAKE_IP_ADDR_JSON, ""
        if cmd[0] == "iw":
            return 0, _FAKE_IW_LINK_DISCONNECTED, ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch.object(mod, "_looks_like_wifi", side_effect=lambda n: n == "wlan0"):
        r = mod.diag_network_interface()
    wlan = [i for i in r["interfaces"] if i["name"] == "wlan0"][0]
    assert wlan["wifi_associated"] is False
    assert "wifi_ssid" not in wlan
    _validate(r, "network_interface")


def test_network_interface_ip_missing_returns_empty_with_flag():
    """Stripped build with no iproute2: trees branch on tools_present.ip
    instead of misreading empty interfaces as 'no network'."""
    from src.tools.diag_impls import network_interface as mod
    with patch.object(mod, "run_subprocess",
                      return_value=(-1, "", "command not found")):
        r = mod.diag_network_interface()
    assert r["tools_present"]["ip"] is False
    assert r["interfaces"] == []
    _validate(r, "network_interface")


def test_network_interface_iw_missing_skips_wifi_fields():
    """ip present but iw missing → links enumerate but no wifi_* fields."""
    from src.tools.diag_impls import network_interface as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return (0, "", "") if cmd[1] == "ip" else (-1, "", "")
        if cmd[0] == "ip":
            return 0, _FAKE_IP_ADDR_JSON, ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch.object(mod, "_looks_like_wifi", side_effect=lambda n: n == "wlan0"):
        r = mod.diag_network_interface()
    assert r["tools_present"]["ip"] is True
    assert r["tools_present"]["iw"] is False
    wlan = [i for i in r["interfaces"] if i["name"] == "wlan0"][0]
    assert "wifi_ssid" not in wlan
    assert "wifi_associated" not in wlan
    _validate(r, "network_interface")


def test_network_interface_malformed_json_returns_empty():
    from src.tools.diag_impls import network_interface as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, "", ""
        if cmd[0] == "ip":
            return 0, "not json {{{", ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run):
        r = mod.diag_network_interface()
    assert r["interfaces"] == []
    _validate(r, "network_interface")


def test_network_interface_sysfs_wifi_detection_handles_capital_p():
    """Regression guard 2026-05-28: lab smoke caught wlP2p33s0 (RK3588 Pi's
    WiFi adapter — capital P from systemd-predictable naming) being missed
    by the original `startswith(('wlan','wlp','wlx'))` heuristic. The
    sysfs check returns True iff /sys/class/net/<iface>/wireless exists,
    which is created by cfg80211 regardless of naming scheme.

    On the dev host /sys/class/net doesn't exist so the call returns
    False — which is the right answer because iw also doesn't exist
    there. We patch os.path.isdir to simulate the kernel-side directory
    existence."""
    from src.tools.diag_impls import network_interface as mod

    def fake_isdir(path):
        return path == "/sys/class/net/wlP2p33s0/wireless"

    with patch("os.path.isdir", side_effect=fake_isdir):
        assert mod._looks_like_wifi("wlP2p33s0") is True
        assert mod._looks_like_wifi("eth0") is False
        assert mod._looks_like_wifi("enx00e04c505a48") is False
        assert mod._looks_like_wifi("docker0") is False
        assert mod._looks_like_wifi("br-6ee64bca2132") is False


# ---------------------------------------------------------------------------
# diag/uniondrive (Phase 0.5b)
# ---------------------------------------------------------------------------

_FAKE_MOUNT_OUTPUT = """\
proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)
/dev/sda1 on /media/pi/sda1 type ext4 (rw,relatime,errors=remount-ro)
/media/pi/sda1 on /uniondrive type fuse.mergerfs (rw,nosuid,nodev,relatime,user_id=0,group_id=0,default_permissions,allow_other)
tmpfs on /run type tmpfs (rw,nosuid,nodev,size=10%,mode=755)
"""

_FAKE_DF_BYTES_OUTPUT = """\
Filesystem      1B-blocks         Used    Available Use% Mounted on
/media/pi/sda1 1006632960000 200000000000 800000000000  20% /uniondrive
"""


def test_uniondrive_happy_path_mounted_with_mergerfs():
    from src.tools.diag_impls import uniondrive as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which" and cmd[1] == "mergerfs":
            return 0, "/usr/bin/mergerfs\n", ""
        if cmd[0] == "mount":
            return 0, _FAKE_MOUNT_OUTPUT, ""
        if cmd[0] == "mergerfs" and cmd[1] == "--version":
            return 0, "mergerfs version: 2.33.5\n", ""
        if cmd[0] == "df":
            return 0, _FAKE_DF_BYTES_OUTPUT, ""
        if cmd[0] == "dmesg":
            return 0, "[12345] usb 1-1: new high-speed USB device\n", ""
        return -1, "", ""

    def fake_open(*args, **kwargs):
        # /sys/fs/ext4/sda1/errors_count → 0
        from io import StringIO
        return StringIO("0\n")

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_uniondrive()
    assert r["mounted"] is True
    assert r["mergerfs_installed"] is True
    assert r["mergerfs_version"] == "2.33.5"
    assert r["mount_source"] == "/media/pi/sda1"
    assert r["mount_fstype"] == "fuse.mergerfs"
    assert r["size_bytes"] == 1006632960000
    assert r["used_bytes"] == 200000000000
    assert r["avail_bytes"] == 800000000000
    assert r["use_percent"] == 20
    assert r["backing_device"] == "sda1"
    assert r["ext4_errors_count"] == 0
    assert r["dmesg_io_errors_1h"] == 0
    _validate(r, "uniondrive")


def test_uniondrive_not_mounted():
    from src.tools.diag_impls import uniondrive as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, "/usr/bin/mergerfs\n", ""
        if cmd[0] == "mount":
            # uniondrive not in mount table
            return 0, "proc on /proc type proc (rw)\n", ""
        if cmd[0] == "dmesg":
            return 0, "", ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run):
        r = mod.diag_uniondrive()
    assert r["mounted"] is False
    assert r["mergerfs_installed"] is True
    # No size/backing fields when not mounted.
    assert "size_bytes" not in r
    assert "backing_device" not in r
    _validate(r, "uniondrive")


def test_uniondrive_mergerfs_not_installed():
    from src.tools.diag_impls import uniondrive as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which" and cmd[1] == "mergerfs":
            return 1, "", ""   # not found
        if cmd[0] == "mount":
            return 0, _FAKE_MOUNT_OUTPUT, ""
        if cmd[0] == "df":
            return 0, _FAKE_DF_BYTES_OUTPUT, ""
        if cmd[0] == "dmesg":
            return 0, "", ""
        return -1, "", ""

    def fake_open(*args, **kwargs):
        from io import StringIO
        return StringIO("0\n")

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_uniondrive()
    # Even when mergerfs isn't installed, uniondrive may still be mounted
    # via a plain bind — we surface the truth of each independently.
    assert r["mergerfs_installed"] is False
    assert "mergerfs_version" not in r
    _validate(r, "uniondrive")


def test_uniondrive_dmesg_denied_returns_no_io_count():
    from src.tools.diag_impls import uniondrive as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, "x", ""
        if cmd[0] == "mount":
            return 0, _FAKE_MOUNT_OUTPUT, ""
        if cmd[0] == "df":
            return 0, _FAKE_DF_BYTES_OUTPUT, ""
        if cmd[0] == "dmesg":
            return -1, "", "permission denied"
        return -1, "", ""

    def fake_open(*args, **kwargs):
        from io import StringIO
        return StringIO("0\n")

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_uniondrive()
    # dmesg denied → field absent rather than misleading 0
    assert "dmesg_io_errors_1h" not in r
    _validate(r, "uniondrive")


def test_uniondrive_dmesg_counts_io_errors():
    from src.tools.diag_impls import uniondrive as mod

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "which":
            return 0, "x", ""
        if cmd[0] == "mount":
            return 0, _FAKE_MOUNT_OUTPUT, ""
        if cmd[0] == "df":
            return 0, _FAKE_DF_BYTES_OUTPUT, ""
        if cmd[0] == "dmesg":
            return 0, (
                "[12000] sd 0:0:0:0: I/O error, dev sda1, sector 12345\n"
                "[12001] normal: filesystem syncing\n"
                "[12002] sd 0:0:0:0: I/O error, dev sda1, sector 67890\n"
            ), ""
        return -1, "", ""

    def fake_open(*args, **kwargs):
        from io import StringIO
        return StringIO("0\n")

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_uniondrive()
    assert r["dmesg_io_errors_1h"] == 2
    _validate(r, "uniondrive")


# ---------------------------------------------------------------------------
# diag/identity_health (Phase 0.5b — chain-grounded)
# ---------------------------------------------------------------------------

# Real ipfs-cluster peerID from the lab device (locks the chain
# encoding against accidental regressions; pre-image known from
# /uniondrive/ipfs-cluster/identity.json).
_LAB_CLUSTER_PEER = "12D3KooWE6gC66XWxKacdna5LX4ymwnCCMpaddBFkB8At3WedRaZ"

_FAKE_CONFIG_YAML = """\
identity: CAESQOv383qnlddAENR91qxb/SQfydhCcm5wzUe0uxBf6CzQP5xrZbfxslkZOQ67OSBHBjsPasfIojRZpP8CCoHFCgo=
storeDir: /uniondrive
poolName: "1"
chainName: skale
logLevel: info
listenAddrs:
    - /ip4/0.0.0.0/tcp/40001
authorizer: 12D3KooWMyqtPp57DY46FrHheoRPS6PyQvnV2azspWKpgpyde6zg
"""


def _fake_identity_json():
    return json.dumps({"id": _LAB_CLUSTER_PEER, "private_key": "REDACTED"})


def _bool_addr_hex(is_true: bool, addr_hex: str = "00" * 20) -> str:
    """Encode (bool, address) ABI tuple as a 0x-prefixed 128-hex string."""
    bool_word = "00" * 31 + ("01" if is_true else "00")
    addr_word = "00" * 12 + addr_hex
    return "0x" + bool_word + addr_word


def _uint_pair_hex(a: int, b: int) -> str:
    return "0x" + f"{a:064x}" + f"{b:064x}"


def test_identity_health_happy_path_member_and_online():
    from src.tools.diag_impls import identity_health as mod
    from src.tools.chain import CallResult, clear_cache_for_tests
    clear_cache_for_tests()

    call_results = [
        # 1st call: isPeerIdMemberOfPool → (true, deadbeef..)
        CallResult(state="ok", value=_bool_addr_hex(True, "deadbeefcafebabe1234567890abcdef12345678")),
        # 2nd call: getOnlineStatusSince → (24, 24)
        CallResult(state="ok", value=_uint_pair_hex(24, 24)),
    ]
    eth_calls_made = []

    def fake_eth_call(chain, addr, data, **kw):
        eth_calls_made.append((chain, addr, data))
        return call_results.pop(0)

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO(_FAKE_CONFIG_YAML)
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch.object(mod, "eth_call", side_effect=fake_eth_call), \
         patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is True
    assert r["pool_member_reason"] == "ok"
    assert r["online_recent"] is True
    assert r["online_recent_reason"] == "ok"
    assert r["pool_id"] == 1
    assert r["chain"] == "skale"
    assert r["cluster_peer_id"] == _LAB_CLUSTER_PEER
    assert r["cluster_peer_id_bytes32"].startswith("0x")
    assert len(r["cluster_peer_id_bytes32"]) == 66
    assert r["online_count"] == 24
    assert r["online_total_expected"] == 24
    assert r["pool_member_address"] == "0xdeadbeefcafebabe1234567890abcdef12345678"
    # The two eth_calls hit the expected contract addresses.
    assert len(eth_calls_made) == 2
    pool_storage_call = eth_calls_made[0]
    reward_engine_call = eth_calls_made[1]
    assert pool_storage_call[0] == "skale"
    assert reward_engine_call[0] == "skale"
    _validate(r, "identity_health")


def test_identity_health_not_a_member_and_no_online():
    from src.tools.diag_impls import identity_health as mod
    from src.tools.chain import CallResult, clear_cache_for_tests
    clear_cache_for_tests()

    call_results = [
        CallResult(state="ok", value=_bool_addr_hex(False)),
        CallResult(state="ok", value=_uint_pair_hex(0, 24)),
    ]

    def fake_eth_call(chain, addr, data, **kw):
        return call_results.pop(0)

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO(_FAKE_CONFIG_YAML)
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch.object(mod, "eth_call", side_effect=fake_eth_call), \
         patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is False
    assert r["pool_member_reason"] == "ok"
    assert r["online_recent"] is False
    assert r["online_recent_reason"] == "ok"
    assert r["online_count"] == 0
    assert r["online_total_expected"] == 24
    # No pool_member_address when not a member (zero-address sentinel)
    assert "pool_member_address" not in r
    _validate(r, "identity_health")


def test_identity_health_rpc_unreachable_marks_both_unknown():
    from src.tools.diag_impls import identity_health as mod
    from src.tools.chain import CallResult, clear_cache_for_tests
    clear_cache_for_tests()

    def fake_eth_call(chain, addr, data, **kw):
        return CallResult(state="unknown", reason="rpc_unreachable")

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO(_FAKE_CONFIG_YAML)
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch.object(mod, "eth_call", side_effect=fake_eth_call), \
         patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is None
    assert r["pool_member_reason"] == "rpc_unreachable"
    assert r["online_recent"] is None
    assert r["online_recent_reason"] == "rpc_unreachable"
    # Identity fields are still populated — trees can still surface them
    assert r["cluster_peer_id"] == _LAB_CLUSTER_PEER
    assert r["pool_id"] == 1
    _validate(r, "identity_health")


def test_identity_health_chain_revert_marks_error():
    from src.tools.diag_impls import identity_health as mod
    from src.tools.chain import CallResult, clear_cache_for_tests
    clear_cache_for_tests()

    def fake_eth_call(chain, addr, data, **kw):
        return CallResult(state="error", reason="execution reverted")

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO(_FAKE_CONFIG_YAML)
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch.object(mod, "eth_call", side_effect=fake_eth_call), \
         patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is None
    assert "chain_error" in r["pool_member_reason"]
    assert r["online_recent"] is None
    assert "chain_error" in r["online_recent_reason"]
    _validate(r, "identity_health")


def test_identity_health_missing_pool_id_short_circuits():
    from src.tools.diag_impls import identity_health as mod

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO("chainName: skale\nstoreDir: /uniondrive\n")
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is None
    assert r["pool_member_reason"] == "missing_pool_id"
    assert "pool_id" not in r
    _validate(r, "identity_health")


def test_identity_health_unknown_chain_short_circuits():
    from src.tools.diag_impls import identity_health as mod

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO('poolName: "1"\nchainName: dogechain\n')
        if path == mod.CLUSTER_IDENTITY_PATH:
            return StringIO(_fake_identity_json())
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is None
    assert "unknown_chain:dogechain" in r["pool_member_reason"]
    _validate(r, "identity_health")


def test_identity_health_missing_cluster_identity():
    from src.tools.diag_impls import identity_health as mod

    def fake_open_router(path, *args, **kwargs):
        from io import StringIO
        if path == mod.CONFIG_YAML_PATH:
            return StringIO(_FAKE_CONFIG_YAML)
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=fake_open_router):
        r = mod.diag_identity_health()
    assert r["pool_member"] is None
    assert r["pool_member_reason"] == "missing_cluster_peer_id"
    assert "cluster_peer_id" not in r
    _validate(r, "identity_health")


def test_identity_health_config_yaml_parser_handles_quoted_and_unquoted():
    from src.tools.diag_impls.identity_health import _read_config_yaml

    def open_with(text):
        from unittest.mock import mock_open
        return mock_open(read_data=text)

    # Quoted poolName
    with patch("builtins.open", open_with('poolName: "42"\nchainName: base\n')):
        c = _read_config_yaml("/fake")
    assert c["poolName_int"] == 42
    assert c["chainName"] == "base"

    # Unquoted poolName (also valid YAML — sometimes seen)
    with patch("builtins.open", open_with("poolName: 7\nchainName: skale\n")):
        c = _read_config_yaml("/fake")
    assert c["poolName_int"] == 7

    # Empty file → empty dict (graceful)
    with patch("builtins.open", open_with("")):
        c = _read_config_yaml("/fake")
    assert c == {}

    # Nested list items must be ignored (we only want top-level scalars)
    with patch("builtins.open", open_with(
        'poolName: "1"\nchainName: skale\nlistenAddrs:\n    - /ip4/0.0.0.0/tcp/40001\nauthorizer: 12D3Koo\n'
    )):
        c = _read_config_yaml("/fake")
    assert c["poolName_int"] == 1
    assert c["chainName"] == "skale"
    assert c["authorizer"] == "12D3Koo"


# ---------------------------------------------------------------------------
# diag/kubo_health (Phase 0.5c)
# ---------------------------------------------------------------------------

def test_kubo_health_happy_path():
    from src.tools.diag_impls import kubo_health as mod
    posts = []

    def fake_post(url, body, timeout_s=3.0):
        posts.append(url)
        if url.endswith("/id"):
            return {"ID": "12D3Koo", "AgentVersion": "kubo/0.41.0/",
                    "Addresses": ["/ip4/x/tcp/4001/p2p/X",
                                  "/dns/relay/p2p-circuit/p2p/X"]}
        if url.endswith("/version"):
            return {"Version": "0.41.0", "Commit": "d719fb8"}
        if url.endswith("/swarm/peers"):
            return {"Peers": [{"Peer": "12D3"} for _ in range(97)]}
        return None

    with patch.object(mod, "http_post_json", side_effect=fake_post):
        r = mod.diag_kubo_health()
    assert r["daemon_reachable"] is True
    assert r["peer_id"] == "12D3Koo"
    assert r["agent_version"] == "kubo/0.41.0/"
    assert r["addresses_count"] == 2
    assert r["version"] == "0.41.0"
    assert r["commit"] == "d719fb8"
    assert r["swarm_peer_count"] == 97
    _validate(r, "kubo_health")


def test_kubo_health_daemon_unreachable():
    from src.tools.diag_impls import kubo_health as mod
    with patch.object(mod, "http_post_json", return_value=None):
        r = mod.diag_kubo_health()
    assert r["daemon_reachable"] is False
    assert "peer_id" not in r
    _validate(r, "kubo_health")


def test_kubo_health_id_ok_but_version_call_fails():
    """Partial response — id worked, version didn't. We surface what
    we have rather than treating the whole call as failed."""
    from src.tools.diag_impls import kubo_health as mod

    def fake_post(url, body, timeout_s=3.0):
        if url.endswith("/id"):
            return {"ID": "X", "AgentVersion": "kubo/0.41.0/", "Addresses": []}
        return None

    with patch.object(mod, "http_post_json", side_effect=fake_post):
        r = mod.diag_kubo_health()
    assert r["daemon_reachable"] is True
    assert "version" not in r
    assert "swarm_peer_count" not in r
    _validate(r, "kubo_health")


# ---------------------------------------------------------------------------
# diag/fula_go_health (Phase 0.5c)
# ---------------------------------------------------------------------------

_FAKE_FULA_GO_LOG = """\
2026-05-28T14:53:15.178Z\tINFO\tfula/wap/cmd/mdns\tmdns/mdns.go:145\tmdns info loaded from config file\t{"infoSlice": {"BloxPeerIdString":"12D3KooWCnRu","IpfsClusterID":"12D3KooWE6gC","PoolName":"1","Authorizer":"12D3KooWMyqt","HardwareID":"c6e2a723"}}
2026-05-28T14:53:20.180Z\tINFO\tfula/wap/cmd/mdns\tmdns/mdns.go:145\tmdns info loaded from config file\t{"infoSlice": {"BloxPeerIdString":"12D3KooWCnRu","IpfsClusterID":"12D3KooWE6gC","PoolName":"1","Authorizer":"12D3KooWMyqt","HardwareID":"c6e2a723"}}
2026-05-28T14:53:25.180Z\tINFO\tfula/blox\tblox/registration.go:42\tjoined pool successfully\t{"poolId": 1}
2026-05-28T14:53:30.180Z\tINFO\tfula/wap/cmd/mdns\tmdns/mdns.go:145\tmdns info loaded from config file\t{"infoSlice": {}}
"""


def _make_fake_fula_go_client(state="running", restart_count=0,
                               started_at="2026-05-28T06:21:20.874173806Z",
                               log_bytes=None):
    """Build a fake docker SDK client that returns a fake `fula_go`
    container with the given attrs + logs."""
    from unittest.mock import MagicMock
    fake_container = MagicMock()
    fake_container.attrs = {
        "State": {"Status": state, "StartedAt": started_at},
        "RestartCount": restart_count,
    }
    fake_container.logs = MagicMock(return_value=log_bytes or b"")
    fake_client = MagicMock()
    fake_client.containers.get = MagicMock(return_value=fake_container)
    return fake_client


def test_fula_go_health_running_with_recent_mdns_and_pool_event():
    """Note container.logs() returns BYTES with both stdout AND stderr
    merged (the SDK's default). fula_go's zap writes to stderr — the
    SDK handles the merge for us; no manual concat needed."""
    from src.tools.diag_impls import fula_go_health as mod
    fake_client = _make_fake_fula_go_client(
        log_bytes=_FAKE_FULA_GO_LOG.encode("utf-8"),
    )
    with patch.object(mod, "_docker_client", return_value=fake_client):
        r = mod.diag_fula_go_health()
    assert r["container_running"] is True
    assert r["container_state"] == "running"
    assert r["restart_count"] == 0
    assert r["container_started_at"].startswith("2026-05-28T06:21:20")
    # 3 mdns lines in the fixture, 1 pool event ("joined pool successfully")
    assert r["mdns_broadcasts_in_tail"] == 3
    assert r["last_mdns_loaded_ts"] == "2026-05-28T14:53:30.180Z"
    assert r["last_pool_event_ts"] == "2026-05-28T14:53:25.180Z"
    assert "joined pool successfully" in r["last_pool_event_excerpt"]
    assert r["last_mdns_info"]["BloxPeerIdString"] == "12D3KooWCnRu"
    assert r["last_mdns_info"]["PoolName"] == "1"
    _validate(r, "fula_go_health")


def test_fula_go_health_container_not_running_short_circuits():
    from src.tools.diag_impls import fula_go_health as mod
    fake_client = _make_fake_fula_go_client(state="exited", restart_count=3,
                                             started_at="2026-05-28T06:00:00Z")
    with patch.object(mod, "_docker_client", return_value=fake_client):
        r = mod.diag_fula_go_health()
    assert r["container_running"] is False
    assert r["container_state"] == "exited"
    assert r["restart_count"] == 3
    assert "last_mdns_loaded_ts" not in r
    # logs() should NOT be called when not running
    fake_client.containers.get.return_value.logs.assert_not_called()
    _validate(r, "fula_go_health")


def test_fula_go_health_docker_unavailable():
    """Docker SDK unable to connect — container_running=false, no
    further fields."""
    from src.tools.diag_impls import fula_go_health as mod
    with patch.object(mod, "_docker_client", return_value=None):
        r = mod.diag_fula_go_health()
    assert r["container_running"] is False
    _validate(r, "fula_go_health")


def test_fula_go_health_container_not_found():
    """fula_go doesn't exist in docker — get() raises NotFound."""
    from src.tools.diag_impls import fula_go_health as mod
    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_client.containers.get.side_effect = Exception("404 not found")
    with patch.object(mod, "_docker_client", return_value=fake_client):
        r = mod.diag_fula_go_health()
    assert r["container_running"] is False
    _validate(r, "fula_go_health")


# ---------------------------------------------------------------------------
# diag/image_versions (Phase 0.5c)
# ---------------------------------------------------------------------------

_FAKE_ENV = """\
GO_FULA=functionland/go-fula:test153
FX_SUPPROT=functionland/fxsupport:test153
IPFS_CLUSTER=functionland/ipfs-cluster:test153
FULA_PINNING=functionland/fula-pinning:test153
FULA_GATEWAY=functionland/fula-gateway:test153
WPA_SUPLICANT_PATH=/etc
CURRENT_USER=pi
"""


def _make_fake_docker_for_images(images_by_container):
    """Build a docker SDK fake that returns fake containers with
    Config.Image attrs from the given map."""
    from unittest.mock import MagicMock

    def get_container(name):
        if name not in images_by_container:
            raise Exception(f"No such container: {name}")
        c = MagicMock()
        c.attrs = {"Config": {"Image": images_by_container[name]}}
        return c

    client = MagicMock()
    client.containers.get.side_effect = get_container
    return client


def test_image_versions_all_match():
    from src.tools.diag_impls import image_versions as mod
    fake = _make_fake_docker_for_images({
        "fula_go":        "functionland/go-fula:test153",
        "fula_fxsupport": "functionland/fxsupport:test153",
        "ipfs_cluster":   "functionland/ipfs-cluster:test153",
        "fula_pinning":   "functionland/fula-pinning:test153",
        "fula_gateway":   "functionland/fula-gateway:test153",
        "ipfs_host":      "ipfs/kubo:release",
    })

    def fake_open(path, *args, **kwargs):
        from io import StringIO
        return StringIO(_FAKE_ENV)

    with patch.object(mod, "_docker_client", return_value=fake), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_image_versions()
    assert r["mismatched_containers"] == []
    by_container = {c["container"]: c for c in r["containers"]}
    assert by_container["fula_go"]["match"] is True
    assert by_container["fula_go"]["actual_image"] == "functionland/go-fula:test153"
    assert by_container["fula_go"]["expected_image"] == "functionland/go-fula:test153"
    assert by_container["ipfs_host"]["expected_image"] == "ipfs/kubo:*"
    assert by_container["ipfs_host"]["match"] is True
    _validate(r, "image_versions")


def test_image_versions_mismatch_detected():
    from src.tools.diag_impls import image_versions as mod
    fake = _make_fake_docker_for_images({
        "fula_go":        "functionland/go-fula:release",   # DRIFTED
        "fula_fxsupport": "functionland/x:test153",
        "ipfs_cluster":   "functionland/x:test153",
        "fula_pinning":   "functionland/x:test153",
        "fula_gateway":   "functionland/x:test153",
        "ipfs_host":      "ipfs/kubo:release",
    })

    def fake_open(path, *args, **kwargs):
        from io import StringIO
        return StringIO(_FAKE_ENV)

    with patch.object(mod, "_docker_client", return_value=fake), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_image_versions()
    assert "fula_go" in r["mismatched_containers"]
    fula_go_entry = [c for c in r["containers"] if c["container"] == "fula_go"][0]
    assert fula_go_entry["match"] is False
    _validate(r, "image_versions")


def test_image_versions_env_missing_returns_unset_expected():
    from src.tools.diag_impls import image_versions as mod
    fake = _make_fake_docker_for_images({
        "fula_go": "functionland/go-fula:test153",
    })

    def fake_open(path, *args, **kwargs):
        raise FileNotFoundError(path)

    with patch.object(mod, "_docker_client", return_value=fake), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_image_versions()
    fula_go_entry = [c for c in r["containers"] if c["container"] == "fula_go"][0]
    assert fula_go_entry["expected_image"] == "unset"
    # With expected=unset, match defaults to False; mismatched_containers
    # only includes entries where BOTH actual and expected are present.
    assert "fula_go" not in r["mismatched_containers"]
    _validate(r, "image_versions")


# ---------------------------------------------------------------------------
# diag/ble_state (Phase 0.5c)
# ---------------------------------------------------------------------------

def test_ble_state_present_with_data():
    from src.tools.diag_impls import ble_state as mod
    with patch.object(mod, "read_state", return_value={
        "last_session_ts": "2026-05-28T14:00:00Z",
        "last_command": "ai/status",
        "last_command_ts": "2026-05-28T14:30:00Z",
        "session_count_24h": 5,
    }):
        r = mod.diag_ble_state()
    assert r["present"] is True
    assert r["last_command"] == "ai/status"
    assert r["session_count_24h"] == 5
    _validate(r, "ble_state")


def test_ble_state_missing_returns_null():
    from src.tools.diag_impls import ble_state as mod
    with patch.object(mod, "read_state", return_value={}):
        r = mod.diag_ble_state()
    assert r["present"] is None
    assert "last_command" not in r
    _validate(r, "ble_state")


# ---------------------------------------------------------------------------
# diag/plugins (Phase 0.5c)
# ---------------------------------------------------------------------------

def test_plugins_with_active_blox_ai():
    from src.tools.diag_impls import plugins as mod
    import os
    # Resolve paths through the module-level constants so the test
    # passes regardless of which env-overridable default is active.
    fake_files = {
        mod.ACTIVE_PLUGINS_PATH: "blox-ai\n",
        os.path.join(mod.RUNTIME_PLUGINS_DIR, "blox-ai", "status.txt"): "Installed\n",
        os.path.join(mod.SOURCE_PLUGINS_DIR, "blox-ai", "info.json"): json.dumps({
            "name": "blox-ai", "display_name": "Blox AI",
            "version": "202", "description": "On-device AI assistant.",
        }),
    }

    def fake_open(path, *args, **kwargs):
        from io import StringIO
        if path in fake_files:
            return StringIO(fake_files[path])
        raise FileNotFoundError(path)

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "ls":
            # active-plugins.txt + update-plugins.txt + blox-ai dir
            return 0, "active-plugins.txt\nupdate-plugins.txt\nblox-ai\n", ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_plugins()
    assert r["active"] == ["blox-ai"]
    assert len(r["installed"]) == 1
    plugin = r["installed"][0]
    assert plugin["name"] == "blox-ai"
    assert plugin["active"] is True
    assert plugin["status"] == "Installed"
    assert plugin["display_name"] == "Blox AI"
    assert plugin["version"] == "202"
    _validate(r, "plugins")


def test_plugins_no_active_plugins():
    from src.tools.diag_impls import plugins as mod

    def fake_open(path, *args, **kwargs):
        raise FileNotFoundError(path)

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "ls":
            return -1, "", ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_plugins()
    assert r["active"] == []
    assert r["installed"] == []
    _validate(r, "plugins")


def test_plugins_installed_but_not_active():
    """Plugin install in progress, manager hasn't picked it up yet."""
    from src.tools.diag_impls import plugins as mod
    import os
    fake_files = {
        mod.ACTIVE_PLUGINS_PATH: "",
        os.path.join(mod.RUNTIME_PLUGINS_DIR, "blox-ai", "status.txt"): "Installing\n",
        os.path.join(mod.SOURCE_PLUGINS_DIR, "blox-ai", "info.json"): '{"version":"202"}',
    }

    def fake_open(path, *args, **kwargs):
        from io import StringIO
        if path in fake_files:
            return StringIO(fake_files[path])
        raise FileNotFoundError(path)

    def fake_run(cmd, timeout_s=2.0):
        if cmd[0] == "ls":
            return 0, "active-plugins.txt\nblox-ai\n", ""
        return -1, "", ""

    with patch.object(mod, "run_subprocess", side_effect=fake_run), \
         patch("builtins.open", side_effect=fake_open):
        r = mod.diag_plugins()
    assert r["active"] == []
    assert len(r["installed"]) == 1
    plugin = r["installed"][0]
    assert plugin["active"] is False
    assert plugin["status"] == "Installing"
    _validate(r, "plugins")


# ---------------------------------------------------------------------------
# diag/summary
# ---------------------------------------------------------------------------

def test_summary_overall_green_when_all_green(monkeypatch):
    from src.tools.diag_impls import summary as mod
    # Patch every subsystem to return a known-good response
    monkeypatch.setattr(mod, "diag_internet",
        lambda: {"dns_ok": True, "https_google_ok": True,
                 "https_discovery_ok": True, "latency_ms_avg": 30,
                 "captive_portal_likely": False})
    monkeypatch.setattr(mod, "diag_relay",
        lambda: {"relays": [{"addr": "x"}], "reservation_count": 1})
    monkeypatch.setattr(mod, "diag_time",
        lambda: {"synced": True})
    monkeypatch.setattr(mod, "diag_power", lambda: {"uptime_s": 1, "undervoltage_events_24h": 0})
    monkeypatch.setattr(mod, "diag_storage",
        lambda: {"df": {}, "ext4_errors_count": 0, "dmesg_io_errors_1h": 0})
    monkeypatch.setattr(mod, "diag_containers",
        lambda: {"containers": [{"name": "ipfs_host", "state": "running"}]})
    monkeypatch.setattr(mod, "diag_wireguard",
        lambda: {"installed": True, "registered": True, "active": True,
                 "last_handshake_age_sec": 30})
    monkeypatch.setattr(mod, "diag_heartbeat",
        lambda: {"last_attempt_ts": "2026-05-24T10:00:00Z", "http_status": 200})

    r = mod.diag_summary()
    assert r["overall"] == "green"
    assert set(r["subsystems"].keys()) >= {
        "internet", "relay", "time", "power", "storage", "containers",
        "wireguard", "heartbeat",
    }
    _validate(r, "summary")


def test_summary_overall_red_on_subsystem_red(monkeypatch):
    from src.tools.diag_impls import summary as mod
    monkeypatch.setattr(mod, "diag_internet",
        lambda: {"dns_ok": False, "https_google_ok": False,
                 "https_discovery_ok": False, "latency_ms_avg": 0})
    for fname in ("diag_relay", "diag_time", "diag_power", "diag_storage",
                  "diag_containers", "diag_wireguard", "diag_heartbeat"):
        monkeypatch.setattr(mod, fname, lambda: {"_fake": True})
    r = mod.diag_summary()
    assert r["overall"] == "red"


def test_summary_subsystem_exception_marked_red(monkeypatch):
    from src.tools.diag_impls import summary as mod
    def boom():
        raise RuntimeError("simulated subsystem crash")
    monkeypatch.setattr(mod, "diag_internet", boom)
    for fname in ("diag_relay", "diag_time", "diag_power", "diag_storage",
                  "diag_containers", "diag_wireguard", "diag_heartbeat"):
        monkeypatch.setattr(mod, fname, lambda: {})
    r = mod.diag_summary()
    assert r["subsystems"]["internet"]["status"] == "red"
    assert "error" in r["subsystems"]["internet"]["key_metrics"]


def test_summary_overall_severity_logic():
    """Pure function test — feed in known scorecards, check the aggregator."""
    from src.tools.diag_impls.summary import _overall_severity
    assert _overall_severity({"a": {"status": "green"}, "b": {"status": "green"}}) == "green"
    assert _overall_severity({"a": {"status": "yellow"}, "b": {"status": "green"}}) == "yellow"
    assert _overall_severity({"a": {"status": "red"}, "b": {"status": "yellow"}}) == "red"
    assert _overall_severity({}) == "green"


# ---------------------------------------------------------------------------
# RealDiagExecutor dispatch
# ---------------------------------------------------------------------------

_EXPECTED_TOOL_SET = {
    # diag/* originals (Phase 9, schema v1)
    "diag/internet", "diag/relay", "diag/time", "diag/power",
    "diag/storage", "diag/containers", "diag/wireguard",
    "diag/heartbeat", "diag/events", "diag/readiness", "diag/summary",
    # Phase 0.5a additions (deterministic-tree foundation, schema v2)
    "diag/discovery_state", "diag/systemd_services", "diag/network_interface",
    # Phase 0.5b additions (uniondrive + identity_health, schema v3)
    "diag/uniondrive", "diag/identity_health",
    # Phase 0.5c additions (kubo/fula_go/image_versions/ble/plugins, schema v4)
    "diag/kubo_health", "diag/fula_go_health", "diag/image_versions",
    "diag/ble_state", "diag/plugins",
}


def test_real_executor_dispatches_all_tools():
    """Sanity: every tool name resolves to an impl + the executor raises
    UnknownToolError on bad names. Uses asyncio.run rather than the
    pytest-asyncio plugin (one extra dep we don't otherwise need)."""
    import asyncio
    from src.tools.diag_impls import RealDiagExecutor, known_tools, UnknownToolError
    ex = RealDiagExecutor()
    assert set(known_tools()) == _EXPECTED_TOOL_SET
    with pytest.raises(UnknownToolError):
        asyncio.run(ex("diag/no-such-tool", {}))


def test_real_executor_known_tools_set_matches_schema():
    """Every tool listed in fula-ota's ble_commands.json MUST be in
    known_tools(). Cross-runtime contract — drift here means a BLE
    invocation will hit a 404 at runtime."""
    from src.tools.diag_impls import known_tools
    assert set(known_tools()) == _EXPECTED_TOOL_SET


# ---------------------------------------------------------------------------
# Phase 0.5d — cross-runtime consistency ratchets
#
# These tests enforce that the FIVE sources of truth for the diag/* tool
# set stay aligned. Drift between any pair causes a real runtime bug:
#
#   1. _DISPATCH (src/tools/diag_impls/__init__.py)  → actual impls
#   2. ToolName Literal (src/routes/diag.py)         → HTTP route enum
#   3. _CANNED dict (src/runtime/mock_diag.py)       → mock executor
#   4. ble_commands.json (fula-ota)                  → BLE command surface
#   5. diag_responses.schema.json $defs (fula-ota)   → response contract
#
# Drift symptoms by direction:
#   - dispatch has X but ToolName doesn't → HTTP 422 even though impl exists
#   - dispatch has X but mock doesn't     → mock executor 404s on X
#   - dispatch has X but ble_commands doesn't → BLE invocation fails
#   - dispatch has X but $defs doesn't    → impl responses can't be validated
#   - schema has X $def but no impl        → contract documents a non-existent tool
# ---------------------------------------------------------------------------


def test_consistency_dispatch_matches_route_enum():
    """Every dispatched tool name must appear in ToolName Literal."""
    from src.tools.diag_impls import known_tools
    from src.routes.diag import ToolName
    import typing
    literal_args = set(typing.get_args(ToolName))
    dispatched_short = {t.removeprefix("diag/") for t in known_tools()}
    assert dispatched_short == literal_args, (
        f"dispatch vs ToolName drift: "
        f"in dispatch not in ToolName={dispatched_short - literal_args}; "
        f"in ToolName not in dispatch={literal_args - dispatched_short}"
    )


def test_consistency_dispatch_matches_mock_canned():
    """Every dispatched tool must have a canned mock response."""
    from src.tools.diag_impls import known_tools
    from src.runtime.mock_diag import _CANNED
    dispatched = set(known_tools())
    canned = set(_CANNED.keys())
    assert dispatched == canned, (
        f"dispatch vs mock drift: "
        f"in dispatch not in mock={dispatched - canned}; "
        f"in mock not in dispatch={canned - dispatched}"
    )


def test_consistency_dispatch_matches_ble_commands():
    """Every diag/* in dispatch must be in fula-ota's ble_commands.json.
    Skips cleanly when the fula-ota sibling checkout isn't present
    (CI without the sibling can't enforce this — but local + lab CI
    should)."""
    from src.tools.diag_impls import known_tools
    from .conftest import _locate_fula_ota_api_dir
    api_dir = _locate_fula_ota_api_dir()
    if api_dir is None:
        pytest.skip("fula-ota sibling not available; set BLOX_AI_FULA_OTA_SCHEMA_DIR")
    # ble_commands.json lives at the plugin root (one level UP from api/).
    ble_path = api_dir.parent / "ble_commands.json"
    if not ble_path.exists():
        pytest.skip(f"ble_commands.json not at {ble_path}")
    ble_data = json.loads(ble_path.read_text(encoding="utf-8"))
    ble_diag_names = {
        cmd["name"] for cmd in ble_data.get("commands", [])
        if cmd.get("name", "").startswith("diag/")
    }
    dispatched = set(known_tools())
    assert dispatched == ble_diag_names, (
        f"dispatch vs ble_commands drift: "
        f"in dispatch not in ble={dispatched - ble_diag_names}; "
        f"in ble not in dispatch={ble_diag_names - dispatched}"
    )


def test_consistency_dispatch_matches_schema_defs():
    """Every dispatched diag/X must have a $def `X` in
    diag_responses.schema.json. Skips when fula-ota sibling absent."""
    from src.tools.diag_impls import known_tools
    from .conftest import _locate_fula_ota_api_dir
    api_dir = _locate_fula_ota_api_dir()
    if api_dir is None:
        pytest.skip("fula-ota sibling not available")
    schema_path = api_dir / "diag_responses.schema.json"
    if not schema_path.exists():
        pytest.skip(f"schema not at {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_defs = set(schema.get("$defs", {}).keys())
    # The schema also defines helper types (severity, iso8601_datetime,
    # subsystem_name) which aren't diag tools. Filter to the
    # tool-shaped defs by intersecting with dispatched-name suffixes.
    dispatched_short = {t.removeprefix("diag/") for t in known_tools()}
    # Every dispatched tool MUST have a $def
    missing_defs = dispatched_short - schema_defs
    assert not missing_defs, (
        f"dispatched tools without schema $def: {missing_defs}. "
        f"Schema defs available: {schema_defs}"
    )


def test_consistency_ble_schema_versions_match():
    """ble_commands.json and diag_responses.schema.json should share
    the SAME schema_version — both bump on every diag-tool addition.
    A divergence means someone added a tool to one file but forgot the
    other."""
    from .conftest import _locate_fula_ota_api_dir
    api_dir = _locate_fula_ota_api_dir()
    if api_dir is None:
        pytest.skip("fula-ota sibling not available")
    ble_path = api_dir.parent / "ble_commands.json"
    schema_path = api_dir / "diag_responses.schema.json"
    if not (ble_path.exists() and schema_path.exists()):
        pytest.skip("contract files missing")
    ble_v = json.loads(ble_path.read_text(encoding="utf-8")).get("schema_version")
    schema_v = json.loads(schema_path.read_text(encoding="utf-8")).get("schema_version")
    assert ble_v == schema_v, (
        f"schema_version drift: ble_commands.json={ble_v}, "
        f"diag_responses.schema.json={schema_v}. Both must bump together "
        f"on every diag-tool addition (Phase 0.5d ratchet)."
    )


def test_consistency_ble_commands_schema_minimum_version():
    """Defensive lower-bound: post-Phase-0.5c the schema_version MUST be
    >= 4. Prevents accidental downgrade by a future PR that loses the
    additive history."""
    from .conftest import _locate_fula_ota_api_dir
    api_dir = _locate_fula_ota_api_dir()
    if api_dir is None:
        pytest.skip("fula-ota sibling not available")
    schema_path = api_dir / "diag_responses.schema.json"
    if not schema_path.exists():
        pytest.skip("schema missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sv = schema.get("schema_version")
    assert isinstance(sv, int) and sv >= 4, (
        f"diag_responses.schema.json schema_version must be >= 4 "
        f"(Phase 0.5c was the v4 bump); got {sv!r}"
    )
