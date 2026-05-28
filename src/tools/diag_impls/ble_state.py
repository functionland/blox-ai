"""diag/ble_state — recent BLE session activity.

Reads /run/fula-ble.state, written by the planned Phase 1.9 BLE plugin
extension scanner on each registered-command invocation. Schema (per
the plan):
  {last_session_ts, last_command, last_command_ts, session_count_24h}

Tristate per the discovery_state pattern: `present=null` when the
state file hasn't been written yet (pre-Phase-1.9 firmware or no BLE
session has happened since boot). Trees should branch on null
explicitly rather than treating absence as "never used."
"""
from __future__ import annotations

from src.tools.diag_impls._helpers import read_state


BLE_STATE_PATH = "/run/fula-ble.state"


def diag_ble_state() -> dict:
    state = read_state(BLE_STATE_PATH)
    if not state:
        # Tristate null — file absent or unreadable.
        return {"present": None}
    out: dict = {"present": True}
    for key in ("last_session_ts", "last_command", "last_command_ts"):
        v = state.get(key)
        if isinstance(v, str) and v:
            out[key] = v[:200]
    count = state.get("session_count_24h")
    if isinstance(count, int) and count >= 0:
        out["session_count_24h"] = count
    return out
