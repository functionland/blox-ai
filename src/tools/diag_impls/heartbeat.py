"""diag/heartbeat — last heartbeat attempt + reserved relay set.

Reads /run/fula-heartbeat.state written by Phase 1.8."""
from __future__ import annotations

from src.tools.diag_impls._helpers import now_iso, read_state


HEARTBEAT_STATE_PATH = "/run/fula-heartbeat.state"


def diag_heartbeat() -> dict:
    state = read_state(HEARTBEAT_STATE_PATH)
    # Schema requires last_attempt_ts. If state file is missing, return
    # a synthesized "never attempted" with now_iso() — alternative would
    # be a 404, but the AI is better served by a clearly-empty record.
    last_ts = state.get("last_attempt_ts")
    if not isinstance(last_ts, str):
        return {"last_attempt_ts": now_iso()}
    out: dict = {"last_attempt_ts": last_ts}
    for k in ("http_status", "last_circuit_count"):
        v = state.get(k)
        if isinstance(v, int) and v >= 0:
            out[k] = v
    err = state.get("error")
    if isinstance(err, str) and err:
        out["error"] = err[:1000]
    reserved = state.get("last_reserved_on")
    if isinstance(reserved, list) and all(isinstance(x, str) for x in reserved):
        out["last_reserved_on"] = reserved
    return out
