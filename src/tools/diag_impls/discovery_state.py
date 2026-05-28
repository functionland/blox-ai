"""diag/discovery_state — wrap /run/fula-discovery.state.

Phase 1.2 of the macro plan added a `check_discovery_https_reachable()`
probe in readiness-check.py that writes
`/run/fula-discovery.state` on each cycle. This diag tool surfaces that
state to the deterministic tree (the tree's "internet ok but
discovery.fula.network unreachable" branch will read this).

Read-only — no probe is performed here. The probe runs in
readiness-check.py once per cycle; doing it here too would double the
network traffic and could race the cached value. If the state file
hasn't been written yet (fresh boot, or pre-Phase-1.2 firmware), we
return `ok=null` so trees can branch on the unknown explicitly.
"""
from __future__ import annotations

from src.tools.diag_impls._helpers import read_state


DISCOVERY_STATE_PATH = "/run/fula-discovery.state"


def diag_discovery_state() -> dict:
    state = read_state(DISCOVERY_STATE_PATH)
    out: dict = {
        # Tristate: True / False / None. JSON-encodes as bool or null.
        # Phase 1.2 hasn't shipped yet on every device, so None is the
        # honest answer when there's no file.
        "ok": state.get("ok") if isinstance(state.get("ok"), bool) else None,
    }
    last_ts = state.get("last_check_ts")
    if isinstance(last_ts, str) and last_ts:
        out["last_check_ts"] = last_ts
    err = state.get("error")
    if isinstance(err, str) and err:
        out["error"] = err[:500]
    latency = state.get("latency_ms")
    if isinstance(latency, (int, float)) and latency >= 0:
        out["latency_ms"] = float(latency)
    return out
