"""diag/systemd_services — `systemctl is-active` over the fula unit set.

Surfaces the active/inactive/failed state for the small fixed set of
units the deterministic tree branches on (fula.service +
uniondrive.service + wireguard-support.service + readiness-check +
recover + docker + blox-ai). Trees can map "fula not active" → "ask user
to run `sudo systemctl restart fula` or restart via the plugin's
action_executor."

We deliberately query ONE unit per subprocess call rather than the
batch form `systemctl is-active u1 u2 u3` so that:
  - a single 2s timeout caps the worst case (a hung systemctl on one
    unit doesn't poison the others)
  - the parse is trivial (one line of output per call)
  - units that don't exist on this device (e.g. blox-ai when the plugin
    isn't installed) come back as `unknown` rather than dragging the
    whole call to error

We do NOT call `systemctl status` (much heavier; renders the journal
tail; can spawn a pager); we only call `is-active` + `show
--property=Result`. Both are constant-time + side-effect-free.
"""
from __future__ import annotations

from src.tools.diag_impls._helpers import run_subprocess


# Units the deterministic tree references. Adding a unit here is the
# whole change — no schema bump (the response is a list, not enumerated
# at the schema level).
FULA_UNITS: tuple[str, ...] = (
    "fula.service",
    "uniondrive.service",
    "wireguard-support.service",
    "fula-readiness-check.service",
    "fula-readiness-check-recover.service",
    "docker.service",
    "blox-ai.service",
)

# Per-unit subprocess timeout. systemctl is normally near-instant; >2s
# means systemd itself is wedged, which we surface as `unknown`.
_PER_UNIT_TIMEOUT_S = 2.0


def diag_systemd_services() -> dict:
    services: list[dict] = []
    for unit in FULA_UNITS:
        services.append(_query_unit(unit))
    return {"services": services}


def _query_unit(unit: str) -> dict:
    # `is-active` returns 0 on active, 3 on inactive/failed, etc. The
    # text output is the key signal — `active`, `inactive`, `failed`,
    # `activating`, etc.
    rc, out, _ = run_subprocess(
        ["systemctl", "is-active", unit],
        timeout_s=_PER_UNIT_TIMEOUT_S,
    )
    if rc == -1:
        # systemctl not present (Windows dev / container without init),
        # OR per-unit timeout. Either way the deterministic tree should
        # branch on `unknown` rather than try to interpret a missing
        # answer.
        return {"name": unit, "active": None, "state": "unknown"}
    state_text = (out or "").strip() or "unknown"
    active = state_text == "active"
    result: dict = {"name": unit, "active": active, "state": state_text}

    # Pull Result + SubState in one show call — cheaper than two queries.
    # Failure mode: skip silently. The tree only needs `active`/`state`
    # for branching; sub_state + result are nice-to-have detail.
    rc2, show_out, _ = run_subprocess(
        ["systemctl", "show", unit,
         "--property=Result,SubState"],
        timeout_s=_PER_UNIT_TIMEOUT_S,
    )
    if rc2 == 0 and show_out:
        for line in show_out.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if not v:
                continue
            if k == "Result":
                result["result"] = v
            elif k == "SubState":
                result["sub_state"] = v
    return result
