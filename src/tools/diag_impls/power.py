"""diag/power — RK3588 power/undervoltage/temperature/uptime.

Reads /run/fula-power.state written by Phase 1.6's check_power_health.
Falls back to /proc/uptime + sysfs reads if the state file isn't present
(the schema's only required field is uptime_s)."""
from __future__ import annotations

from pathlib import Path

from src.tools.diag_impls._helpers import read_state


POWER_STATE_PATH = "/run/fula-power.state"


def diag_power() -> dict:
    state = read_state(POWER_STATE_PATH)
    if state:
        out = {"uptime_s": int(state.get("uptime_s", 0))}
        for k in ("undervoltage_events_24h", "recent_reboots"):
            v = state.get(k)
            if isinstance(v, int) and v >= 0:
                out[k] = v
        for k in ("max_temp_c", "soc_voltage_ratio"):
            v = state.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out
    # Fallback: just /proc/uptime
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            uptime_s = int(float(f.readline().split()[0]))
    except (OSError, ValueError, IndexError):
        uptime_s = 0
    return {"uptime_s": uptime_s}
