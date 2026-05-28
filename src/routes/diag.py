"""C3 — GET /diag/{tool} routes.

Each route maps to the corresponding impl in src.tools.diag_impls. The
SAME impls back the tool-call loop in /troubleshoot — single source of
truth for what each tool returns.

Closed enum on the path parameter so a typo'd tool name surfaces as 404
immediately (not "endpoint accepts anything").
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request


router = APIRouter()
logger = logging.getLogger("blox-ai.routes.diag")


ToolName = Literal[
    "internet", "relay", "time", "power", "storage", "containers",
    "wireguard", "heartbeat", "events", "readiness", "summary",
    "discovery_state", "systemd_services", "network_interface",
    "uniondrive", "identity_health",
    "kubo_health", "fula_go_health", "image_versions", "ble_state", "plugins",
]


@router.get("/diag/{tool}")
async def diag(tool: ToolName, request: Request) -> dict:
    executor = request.app.state.tool_executor
    full_name = f"diag/{tool}"
    try:
        return await executor(full_name, {})
    except KeyError:
        # Executor didn't recognise the tool (e.g. someone wired a
        # mock that doesn't cover all tools). Surface as 404 rather
        # than 500 — the tool name was syntactically valid but not
        # supported by this executor.
        raise HTTPException(status_code=404, detail=f"unknown tool: {full_name}")
    except Exception as e:
        logger.exception("diag/%s impl raised", tool)
        raise HTTPException(status_code=500, detail=str(e)[:200])
