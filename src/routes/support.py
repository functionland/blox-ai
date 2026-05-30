"""POST /support/wireguard — start/restart the WireGuard support tunnel
over the LAN HTTP channel, and verify it actually came up.

Distinct from the AI action executor: this is a direct, user-initiated
action from the app's "Enable Remote Support" button, not an AI-proposed
recommended_action. So it does NOT go through the HMAC approval-token /
whitelist machinery. Instead it is gated by two checks:

  1. A custom request header `X-Fula-Support: enable`. Custom headers force
     a CORS preflight in every modern browser, so a drive-by page on the
     same LAN can't trigger this with a simple cross-origin POST — that is
     the materially-worse LAN failure mode the existing BLE button doesn't
     have. The core BLE proxy can't send custom headers, so this endpoint
     is deliberately LAN-only; the BLE "SUPPORT ON" button covers BLE.
  2. The tier-3 security code (same file the action executor reads), so
     enabling a remote-support tunnel is a deliberate, authenticated act.

Once both gates pass it does NOT blindly `systemctl restart` and trust the
exit code. `wireguard-support.service` is `Type=oneshot RemainAfterExit=yes`,
so `systemctl is-active` reports "active" even after the tunnel drops at the
protocol level — and a restart whose ExecStart short-circuits can report
success while the interface never appeared. So this endpoint mirrors
readiness-check.py's activate_wireguard_support() lifecycle:

  1. Pre-check install/setup state via the host's wireguard/status.sh
     (installed = wg binary + keys; registered = registration.state present;
     active = `ip link show support` succeeds).
  2. If NOT installed, run the idempotent host installer (install.sh) on
     demand — a restart alone exits 1 in start.sh when wg is absent.
  3. `systemctl reset-failed` to clear any latched start-limit lock (the
     unit is Restart=on-failure; repeated failures can latch it).
  4. `systemctl restart` (not start — restart forces a clean
     re-establishment even when a stale interface already exists).
  5. Re-run status.sh and treat its `active` (interface present) as the
     GROUND TRUTH for "did the tunnel come up", independent of the restart
     exit code.

All host commands run via nsenter into PID 1's namespaces (the container
runs as root, so no sudo is needed inside the host ns). The source IP is
logged for the on-device audit trail.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.tools.executor import read_security_code


router = APIRouter()
logger = logging.getLogger("blox-ai.support")


SUPPORT_HEADER = "x-fula-support"
SUPPORT_HEADER_VALUE = "enable"
WG_SUPPORT_UNIT = "wireguard-support.service"

# Host-side scripts (paths inside PID 1's mount namespace, which nsenter
# enters). status.sh emits the installed/registered/active JSON; install.sh
# is the idempotent one-time installer (apt wireguard-tools + keygen + unit).
WG_STATUS_SCRIPT = "/usr/bin/fula/wireguard/status.sh"
WG_INSTALL_SCRIPT = "/usr/bin/fula/wireguard/install.sh"

# nsenter into PID 1 — same namespace set the restart already used. --mount
# gives the host fs (scripts, python3, wg binary); --net gives the host's
# network namespace so `ip link show support` / `wg show` see the tunnel.
NSENTER_PREFIX = [
    "nsenter", "--target", "1", "--mount", "--uts", "--ipc", "--net", "--pid",
]

STATUS_TIMEOUT_S = 15.0
# install.sh apt-installs wireguard-tools on a cold device; generous because
# it only runs on the rare not-yet-installed path and the server is the
# authority on completion (if the app's client times out first, install
# still finishes server-side and the idempotent retry completes fast).
INSTALL_TIMEOUT_S = 150.0
RESET_FAILED_TIMEOUT_S = 10.0
# Was 30s — too tight: start.sh can call register_wireguard.sh (a network
# round-trip to the support server) before `wg-quick up`. The unit's own
# TimeoutStartSec is 240s; 60s comfortably covers a registration round-trip
# while still bounding a hung restart well under the unit cap.
RESTART_TIMEOUT_S = 60.0


def _truncate(s: str, n: int = 2048) -> str:
    if not s:
        return ""
    # Strip CR/LF before truncation (log-injection mitigation), mirroring
    # src/tools/executor.py._truncate. status.sh emits single-line JSON, so
    # this never corrupts the payload we parse downstream.
    safe = s.replace("\r", " ").replace("\n", " ")
    return safe if len(safe) <= n else safe[: n - 1] + "…"


async def _run(cmd: list[str], timeout: float) -> dict:
    """Run a subprocess off the event loop, mirroring executor._run."""
    loop = asyncio.get_event_loop()

    def _exec() -> dict:
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
            return {
                "success": cp.returncode == 0,
                "exit_code": cp.returncode,
                "stdout_excerpt": _truncate(cp.stdout),
                "stderr_excerpt": _truncate(cp.stderr),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "exit_code": -1,
                    "stderr_excerpt": f"timeout after {timeout}s"}
        except FileNotFoundError as e:
            return {"success": False, "exit_code": -1,
                    "stderr_excerpt": f"command not found: {e.filename or cmd[0]}"}
        except OSError as e:
            return {"success": False, "exit_code": -1,
                    "stderr_excerpt": f"OS error: {e}"}

    return await loop.run_in_executor(None, _exec)


def _nsenter(cmd: list[str]) -> list[str]:
    """Prefix a host command with nsenter into PID 1's namespaces."""
    return NSENTER_PREFIX + list(cmd)


async def _wg_status(client_ip: str) -> dict | None:
    """Run the host's wireguard status.sh and parse its JSON
    ({installed, registered, active, last_handshake_age_sec, ...}).

    Returns None if the script can't be run or its output isn't parseable —
    callers treat None as "unknown" and fall back to the restart exit code
    rather than hard-failing on a transient status-check hiccup."""
    res = await _run(_nsenter(["bash", WG_STATUS_SCRIPT]), timeout=STATUS_TIMEOUT_S)
    if not res.get("success"):
        logger.warning(
            "support/wireguard: status.sh exited %s (from %s)",
            res.get("exit_code"), client_ip,
        )
        return None
    raw = res.get("stdout_excerpt") or ""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning(
            "support/wireguard: status.sh output not JSON (from %s)", client_ip,
        )
        return None
    return parsed if isinstance(parsed, dict) else None


@router.post("/support/wireguard")
async def support_wireguard(request: Request) -> JSONResponse:
    client_ip = request.client.host if request.client else "unknown"

    # 1. Custom-header gate (forces CORS preflight; blocks browser drive-by).
    header_val = request.headers.get(SUPPORT_HEADER, "")
    if header_val.strip().lower() != SUPPORT_HEADER_VALUE:
        logger.warning(
            "support/wireguard rejected (missing/invalid %s header) from %s",
            SUPPORT_HEADER, client_ip,
        )
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "support_header_required"},
        )

    # 2. Parse body (tolerate empty/malformed body — security code lives here).
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    security_code = body.get("security_code")

    # 3. Security-code gate (same file + plain comparison as the executor's
    # tier-3 check, for consistency; hardening the hardcoded code is a
    # separately-tracked task).
    file_code = read_security_code()
    if file_code is None:
        logger.warning(
            "support/wireguard rejected (security code file missing) from %s",
            client_ip,
        )
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "security_code_file_missing"},
        )
    if not security_code or security_code != file_code:
        logger.warning(
            "support/wireguard rejected (security code invalid) from %s", client_ip,
        )
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": "security_code_invalid"},
        )

    # 4. Pre-flight: capture install/setup/active state. Lets the response
    # report the starting point and decide whether an on-demand install is
    # needed before a restart can possibly succeed.
    pre_status = await _wg_status(client_ip)

    installed_on_demand = False
    if pre_status is not None and pre_status.get("installed") is False:
        # wg binary / keys absent → start.sh exits 1 ("wg not installed") and
        # a bare restart can never bring the tunnel up. Run the idempotent
        # host installer on demand, mirroring activate_wireguard_support().
        logger.info(
            "support/wireguard: not installed; running install.sh on demand (from %s)",
            client_ip,
        )
        install_res = await _run(
            _nsenter(["bash", WG_INSTALL_SCRIPT]), timeout=INSTALL_TIMEOUT_S,
        )
        installed_on_demand = True
        if not install_res["success"]:
            logger.warning(
                "support/wireguard: install.sh failed exit=%s (from %s)",
                install_res.get("exit_code"), client_ip,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "wireguard_not_installed",
                    "exit_code": install_res.get("exit_code"),
                    "stderr_excerpt": install_res.get("stderr_excerpt", ""),
                    "status": pre_status,
                    "installed_on_demand": True,
                },
            )

    # 5. Clear any latched start-limit lock before the restart. The unit is
    # Restart=on-failure; repeated failures (e.g. support server unreachable
    # for a few minutes) can latch it "failed" and silently refuse a start.
    # reset-failed is best-effort — its own failure must not block the restart.
    await _run(
        _nsenter(["systemctl", "reset-failed", WG_SUPPORT_UNIT]),
        timeout=RESET_FAILED_TIMEOUT_S,
    )

    # 6. Restart the support tunnel on the host. restart (not start) forces a
    # clean re-establishment even when a stale `support` interface already
    # exists but the protocol has dropped (start.sh short-circuits on an
    # existing interface, so a plain start would NOT re-handshake).
    logger.info(
        "support/wireguard: restarting %s (requested from %s)",
        WG_SUPPORT_UNIT, client_ip,
    )
    result = await _run(
        _nsenter(["systemctl", "restart", WG_SUPPORT_UNIT]),
        timeout=RESTART_TIMEOUT_S,
    )

    # 7. Post-condition: independently verify the tunnel actually came up.
    # `systemctl is-active` lies (RemainAfterExit=yes) and the restart exit
    # code can be 0 even when start.sh's `wg-quick up` left no interface, so
    # status.sh's `active` (ip link show support) is the ground truth.
    post_status = await _wg_status(client_ip)
    logger.info(
        "support/wireguard: restart -> exit=%s; post active=%s registered=%s (from %s)",
        result.get("exit_code"),
        (post_status or {}).get("active"),
        (post_status or {}).get("registered"),
        client_ip,
    )

    if post_status is not None and post_status.get("active") is True:
        return JSONResponse(status_code=200, content={
            "success": True,
            "exit_code": result.get("exit_code", 0),
            "stdout_excerpt": result.get("stdout_excerpt", ""),
            "stderr_excerpt": result.get("stderr_excerpt", ""),
            "status": post_status,
            "installed_on_demand": installed_on_demand,
        })

    if post_status is None:
        # Couldn't verify (status.sh unavailable). Don't regress to a hard
        # failure on a transient status hiccup — fall back to the restart's
        # own exit code as the success signal.
        ok = bool(result["success"])
        return JSONResponse(status_code=200 if ok else 500, content={
            "success": ok,
            "exit_code": result.get("exit_code"),
            "stdout_excerpt": result.get("stdout_excerpt", ""),
            "stderr_excerpt": result.get("stderr_excerpt", ""),
            "status": None,
            "installed_on_demand": installed_on_demand,
        })

    # Verified inactive: the restart ran but the interface never appeared.
    # Distinct from a restart that errored outright so the app can tell the
    # user "we tried but the tunnel didn't come up" vs a gate rejection.
    logger.warning(
        "support/wireguard: restart completed but tunnel still inactive "
        "(registered=%s) (from %s)",
        post_status.get("registered"), client_ip,
    )
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "tunnel_inactive_after_restart",
        "exit_code": result.get("exit_code"),
        "stdout_excerpt": result.get("stdout_excerpt", ""),
        "stderr_excerpt": result.get("stderr_excerpt", ""),
        "status": post_status,
        "installed_on_demand": installed_on_demand,
    })
