"""diag/wireguard — handshake age + transfer counters.

Calls `wg show support` (or a status script if present). Reads the
extended Phase 1.4 fields when available. Falls back to {"installed":
false, "registered": false, "active": false} when wg isn't present
(dev machine, container without --cap-add=NET_ADMIN).
"""
from __future__ import annotations

import re

from src.tools.diag_impls._helpers import run_subprocess


WG_INTERFACE = "support"


def diag_wireguard() -> dict:
    out: dict = {
        "installed": False,
        "registered": False,
        "active": False,
    }
    # 1. interface present?
    rc, _, _ = run_subprocess(["which", "wg"], timeout_s=2.0)
    if rc != 0:
        return out
    out["installed"] = True

    rc, ifaces, _ = run_subprocess(["wg", "show", "interfaces"], timeout_s=2.0)
    if rc != 0 or WG_INTERFACE not in (ifaces or "").split():
        return out
    out["registered"] = True
    out["active"] = True

    # 2. handshake age (epoch seconds → age)
    rc, hs_out, _ = run_subprocess(
        ["wg", "show", WG_INTERFACE, "latest-handshakes"],
        timeout_s=2.0,
    )
    if rc == 0:
        for line in (hs_out or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    hs_epoch = int(parts[1])
                    if hs_epoch > 0:
                        import time
                        out["last_handshake_age_sec"] = int(time.time()) - hs_epoch
                        break
                except ValueError:
                    continue

    # 3. transfer counters
    rc, tr_out, _ = run_subprocess(
        ["wg", "show", WG_INTERFACE, "transfer"],
        timeout_s=2.0,
    )
    if rc == 0:
        for line in (tr_out or "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                try:
                    out["rx_bytes"] = int(parts[1])
                    out["tx_bytes"] = int(parts[2])
                    break
                except ValueError:
                    continue

    # 4. persistent keepalive
    rc, pk_out, _ = run_subprocess(
        ["wg", "show", WG_INTERFACE, "persistent-keepalive"],
        timeout_s=2.0,
    )
    if rc == 0:
        for line in (pk_out or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                out["persistent_keepalive_sec"] = int(parts[1])
                break

    return out
