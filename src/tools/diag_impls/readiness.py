"""diag/readiness — journalctl tail of fula-readiness-check.service."""
from __future__ import annotations

from src.tools.diag_impls._helpers import run_subprocess


READINESS_UNIT = "fula-readiness-check.service"
RECOVER_UNIT = "fula-readiness-check-recover.service"
LOG_LINES = 100


def diag_readiness() -> dict:
    rc, out, _ = run_subprocess(
        ["journalctl", "-u", READINESS_UNIT, "-n", str(LOG_LINES),
         "--no-pager", "-o", "short-iso"],
        timeout_s=5.0,
    )
    result: dict = {
        "recent_log": (out or "")[:60_000],  # bounded so the LLM context stays sane
    }
    # Latest failure timestamp if the unit failed recently
    rc2, st_out, _ = run_subprocess(
        ["systemctl", "show", READINESS_UNIT, "--property=InactiveExitTimestamp,Result"],
        timeout_s=3.0,
    )
    if rc2 == 0 and st_out:
        for line in st_out.splitlines():
            if line.startswith("InactiveExitTimestamp="):
                ts = line.split("=", 1)[1].strip()
                if ts and ts != "0":
                    # systemd timestamps aren't ISO; surface as is —
                    # the schema's `last_failure_ts` requires ISO so we
                    # skip rather than mislabel.
                    pass
    # Recover-state snapshot — minimal; future iteration can include
    # the sentinel file's existence.
    return result
