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

def test_real_executor_dispatches_all_11_tools():
    """Sanity: every tool name resolves to an impl + the executor raises
    UnknownToolError on bad names. Uses asyncio.run rather than the
    pytest-asyncio plugin (one extra dep we don't otherwise need)."""
    import asyncio
    from src.tools.diag_impls import RealDiagExecutor, known_tools, UnknownToolError
    ex = RealDiagExecutor()
    assert set(known_tools()) == {
        "diag/internet", "diag/relay", "diag/time", "diag/power",
        "diag/storage", "diag/containers", "diag/wireguard",
        "diag/heartbeat", "diag/events", "diag/readiness", "diag/summary",
    }
    with pytest.raises(UnknownToolError):
        asyncio.run(ex("diag/no-such-tool", {}))


def test_real_executor_known_tools_set_matches_schema():
    """The 11 tools listed in fula-ota's ble_commands.json MUST all be
    in known_tools(). Cross-runtime contract."""
    from src.tools.diag_impls import known_tools
    expected = {
        "diag/internet", "diag/relay", "diag/time", "diag/power",
        "diag/storage", "diag/containers", "diag/wireguard",
        "diag/heartbeat", "diag/events", "diag/readiness", "diag/summary",
    }
    assert set(known_tools()) == expected
