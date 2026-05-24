"""diag/time — NTP sync status (state file + timedatectl).

File named time_.py (trailing underscore) to avoid shadowing the
stdlib `time` module if anyone does `from src.tools.diag_impls import *`."""
from __future__ import annotations

from src.tools.diag_impls._helpers import read_state, run_subprocess


TIME_STATE_PATH = "/run/fula-time.state"


def diag_time() -> dict:
    """Prefers /run/fula-time.state written by Phase 1.3's check_ntp_sync.
    Falls back to a live `timedatectl` shell-out when the state file is
    absent (typical on a freshly-installed device before the first
    readiness-check cycle)."""
    state = read_state(TIME_STATE_PATH)
    if state:
        return {
            "synced": bool(state.get("synced", False)),
            "offset_ms": float(state.get("offset_ms", 0)),
            "service": _coerce_service(state.get("service")),
            **(
                {"last_sync_ts": state["last_sync_ts"]}
                if isinstance(state.get("last_sync_ts"), str)
                else {}
            ),
        }
    rc, out, _ = run_subprocess(
        ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
        timeout_s=3.0,
    )
    synced = rc == 0 and out.strip().lower() == "yes"
    return {
        "synced": synced,
        "offset_ms": 0.0,
        "service": "unknown",
    }


def _coerce_service(s) -> str:
    """Coerce arbitrary string to the closed enum."""
    if not isinstance(s, str):
        return "unknown"
    if s in {"chronyd", "systemd-timesyncd", "none", "unknown"}:
        return s
    return "unknown"
