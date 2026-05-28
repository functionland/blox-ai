"""Mock diag/* executor for C2 (real implementations land in C3).

The bridge calls `executor(tool_name, args)` and gets back a payload dict
the bridge then wraps into a tool_result SSE event. C2 only needs the
plumbing to work; the payloads here are canned, NOT validated against
diag_responses.schema.json (that validation lives in C3 once the real
implementations exist).
"""
from __future__ import annotations

import asyncio


class UnknownToolError(KeyError):
    """Raised when the model emits a tool_call for a tool the executor
    doesn't know how to run. Distinct exception type so the bridge can
    map it to a tool_result with ok=False + error message."""


# Canned per-tool responses. Shapes are loosely aligned with what C3
# will emit, but C2 doesn't enforce against diag_responses.schema.json
# — these are placeholders.
_CANNED: dict[str, dict] = {
    "diag/internet": {
        "dns_ok": True, "https_google_ok": True, "https_discovery_ok": True,
        "latency_ms_avg": 42, "captive_portal_likely": False,
    },
    "diag/relay": {
        "relays": [], "reservation_count": 0,
    },
    "diag/time": {
        "synced": True, "offset_ms": 12, "service": "systemd-timesyncd",
    },
    "diag/power": {
        "undervoltage_events_24h": 0, "recent_reboots": 1, "max_temp_c": 54,
        "soc_voltage_ratio": 1.0, "uptime_s": 86400,
    },
    "diag/storage": {
        "disk_free_gb": 240.5, "ext4_errors": 0, "io_errors_1h": 0,
    },
    "diag/containers": {
        "containers": [],
    },
    "diag/wireguard": {
        "installed": True, "registered": True,
        "last_handshake_age_seconds": 12, "rx_bytes": 0, "tx_bytes": 0,
    },
    "diag/heartbeat": {
        "last_attempt_ts": "2026-05-24T19:00:00Z", "http_status": 200,
    },
    "diag/events": {
        "events": [],
    },
    "diag/readiness": {
        "log_tail": "fula-readiness-check: nominal",
    },
    "diag/summary": {
        "overall_severity": "green",
        "subsystems": {
            "internet": "green", "relay": "green", "time": "green",
            "power": "green", "storage": "green", "containers": "green",
            "wireguard": "green",
        },
    },
    "diag/discovery_state": {
        "ok": True, "last_check_ts": "2026-05-28T19:00:00Z", "latency_ms": 32,
    },
    "diag/systemd_services": {
        "services": [
            {"name": "fula.service", "active": True, "state": "active",
             "sub_state": "running", "result": "success"},
        ],
    },
    "diag/network_interface": {
        "interfaces": [
            {"name": "wlan0", "operstate": "UP", "mtu": 1500,
             "mac": "aa:bb:cc:dd:ee:ff",
             "ipv4": ["192.168.1.50"], "ipv6": [],
             "wifi_associated": True, "wifi_ssid": "FakeWifi",
             "wifi_signal_dbm": -55},
        ],
        "tools_present": {"ip": True, "iw": True},
    },
    "diag/uniondrive": {
        "mounted": True, "mergerfs_installed": True,
        "mergerfs_version": "2.33.5",
        "mount_source": "/media/pi/sda1", "mount_fstype": "fuse.mergerfs",
        "size_bytes": 1_000_000_000_000, "used_bytes": 200_000_000_000,
        "avail_bytes": 800_000_000_000, "use_percent": 20,
        "backing_device": "sda1", "ext4_errors_count": 0,
        "dmesg_io_errors_1h": 0,
    },
    "diag/identity_health": {
        "pool_member": True, "pool_member_reason": "ok",
        "online_recent": True, "online_recent_reason": "ok",
        "pool_id": 1, "chain": "skale",
        "cluster_peer_id": "12D3KooWE6gC66XWxKacdna5LX4ymwnCCMpaddBFkB8At3WedRaZ",
        "online_count": 24, "online_total_expected": 24,
        "online_window_s": 86400,
    },
}


class MockDiagExecutor:
    """Returns canned payloads keyed by tool name. C3 replaces with the
    real implementations that read /run/fula-*.state, shell out to docker,
    etc."""

    name: str = "mock"

    async def __call__(self, tool: str, args: dict) -> dict:
        # Cheap async-friendly delay to simulate I/O so concurrency tests
        # observe interleaving.
        await asyncio.sleep(0)
        if tool not in _CANNED:
            raise UnknownToolError(tool)
        return _CANNED[tool]
