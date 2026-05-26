"""diag/summary — parallel run of the read-only subset under a 5s budget.

Returns an overall severity + per-subsystem status. Free-form `key_metrics`
per subsystem gives the LLM the most-relevant 2-3 numbers without
re-running every individual diag/* in sequence.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable

from src.tools.diag_impls._helpers import now_iso
from src.tools.diag_impls.containers import diag_containers
from src.tools.diag_impls.heartbeat import diag_heartbeat
from src.tools.diag_impls.internet import diag_internet
from src.tools.diag_impls.power import diag_power
from src.tools.diag_impls.relay import diag_relay
from src.tools.diag_impls.storage import diag_storage
from src.tools.diag_impls.time_ import diag_time
from src.tools.diag_impls.wireguard import diag_wireguard


SUMMARY_BUDGET_S = 5.0


def diag_summary() -> dict:
    """Runs each subsystem in parallel under a global 5s budget. A subsystem
    that times out gets status='red' + a 'timeout' key_metric."""
    subsystems: dict[str, Callable[[], dict]] = {
        "internet":  diag_internet,
        "relay":     diag_relay,
        "time":      diag_time,
        "power":     diag_power,
        "storage":   diag_storage,
        "containers": diag_containers,
        "wireguard": diag_wireguard,
        "heartbeat": diag_heartbeat,
    }
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subsystems)) as pool:
        future_map = {pool.submit(fn): name for name, fn in subsystems.items()}
        try:
            for future in concurrent.futures.as_completed(
                future_map, timeout=SUMMARY_BUDGET_S,
            ):
                name = future_map[future]
                try:
                    raw = future.result(timeout=0.1)
                    results[name] = _scorecard(name, raw)
                except Exception as e:
                    results[name] = {
                        "status": "red",
                        "key_metrics": {"error": str(e)[:200]},
                    }
        except concurrent.futures.TimeoutError:
            # Mark every still-pending subsystem as red+timeout
            for fut, name in future_map.items():
                if not fut.done() and name not in results:
                    results[name] = {
                        "status": "red",
                        "key_metrics": {"error": "timeout"},
                    }
    overall = _overall_severity(results)
    return {
        "overall": overall,
        "generated_at": now_iso(),
        "subsystems": results,
    }


def _scorecard(name: str, raw: dict) -> dict:
    """Translate a per-subsystem diag dict into a small status+key_metrics
    summary."""
    if name == "internet":
        ok = raw.get("dns_ok") and raw.get("https_discovery_ok")
        return {
            "status": "green" if ok else "red",
            "key_metrics": {
                "discovery_https_ok": bool(raw.get("https_discovery_ok")),
                "captive_portal_likely": bool(raw.get("captive_portal_likely")),
                "latency_ms_avg": raw.get("latency_ms_avg", 0),
            },
        }
    if name == "relay":
        rc = raw.get("reservation_count", 0)
        return {
            "status": "green" if rc > 0 else "yellow",
            "key_metrics": {"reservation_count": rc, "peer_count": len(raw.get("relays") or [])},
        }
    if name == "time":
        return {
            "status": "green" if raw.get("synced") else "red",
            "key_metrics": {"synced": bool(raw.get("synced"))},
        }
    if name == "power":
        # Lab observed 2026-05-26: a SINGLE undervoltage event in 24h
        # would flip status=red and feed the AI a panic-level signal,
        # which then dominated the verdict. Real PSU failures show
        # repeated events; isolated transients (one bad cable wiggle,
        # one boot brownout) should be a yellow, not a red.
        #
        # Tiers:
        #   0   events  → green
        #   1-2 events  → yellow (note but don't panic; could be transient)
        #   3+  events  → red (real PSU issue, recommend power-cable check)
        ue = raw.get("undervoltage_events_24h", 0)
        if ue == 0:
            status = "green"
        elif ue <= 2:
            status = "yellow"
        else:
            status = "red"
        return {
            "status": status,
            "key_metrics": {
                "uptime_s": raw.get("uptime_s", 0),
                "undervoltage_events_24h": ue,
            },
        }
    if name == "storage":
        ext4 = raw.get("ext4_errors_count", 0)
        io = raw.get("dmesg_io_errors_1h", 0)
        return {
            "status": "red" if (ext4 > 0 or io > 0) else "green",
            "key_metrics": {"ext4_errors": ext4, "dmesg_io_errors_1h": io},
        }
    if name == "containers":
        running = [c for c in raw.get("containers") or []
                   if c.get("state") == "running"]
        oom = [c for c in raw.get("containers") or [] if c.get("oom_killed")]
        return {
            "status": "red" if oom else ("yellow" if not running else "green"),
            "key_metrics": {"running_count": len(running), "oom_count": len(oom)},
        }
    if name == "wireguard":
        age = raw.get("last_handshake_age_sec")
        status = "green"
        if not raw.get("active"):
            status = "yellow"
        elif isinstance(age, int) and age > 180:
            status = "red"
        return {
            "status": status,
            "key_metrics": {
                "active": bool(raw.get("active")),
                "last_handshake_age_sec": age if age is not None else -1,
            },
        }
    if name == "heartbeat":
        st = raw.get("http_status", 0)
        return {
            "status": "green" if 200 <= st < 300 else "yellow",
            "key_metrics": {"http_status": st},
        }
    return {"status": "green", "key_metrics": {}}


def _overall_severity(results: dict[str, dict]) -> str:
    """Max severity wins. Empty → green (degenerate case)."""
    severities = [r.get("status", "green") for r in results.values()]
    if "red" in severities:
        return "red"
    if "yellow" in severities:
        return "yellow"
    return "green"
