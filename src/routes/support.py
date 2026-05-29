"""POST /support/wireguard — start/restart the WireGuard support tunnel
over the LAN HTTP channel.

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

On success it runs `systemctl restart wireguard-support.service` on the
host via nsenter (PID 1 namespace), mirroring the executor's dispatch. The
source IP is logged for the on-device audit trail.
"""
from __future__ import annotations

import asyncio
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
RESTART_TIMEOUT_S = 30.0


def _truncate(s: str, n: int = 2048) -> str:
    if not s:
        return ""
    # Strip CR/LF before truncation (log-injection mitigation), mirroring
    # src/tools/executor.py._truncate.
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

    # 4. Restart the support tunnel on the host (PID 1 namespace).
    logger.info(
        "support/wireguard: restarting %s (requested from %s)",
        WG_SUPPORT_UNIT, client_ip,
    )
    result = await _run(
        ["nsenter", "--target", "1", "--mount", "--uts", "--ipc",
         "--net", "--pid", "systemctl", "restart", WG_SUPPORT_UNIT],
        timeout=RESTART_TIMEOUT_S,
    )
    logger.info(
        "support/wireguard: restart of %s -> success=%s exit=%s (from %s)",
        WG_SUPPORT_UNIT, result["success"], result.get("exit_code"), client_ip,
    )
    status_code = 200 if result["success"] else 500
    return JSONResponse(status_code=status_code, content=result)
