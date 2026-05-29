"""C3 — GET /diag/{tool} routes.

Each route maps to the corresponding impl in src.tools.diag_impls. The
SAME impls back the tool-call loop in /troubleshoot — single source of
truth for what each tool returns.

Closed enum on the path parameter so a typo'd tool name surfaces as 404
immediately (not "endpoint accepts anything").
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from src.tools.diag_impls import known_tools


router = APIRouter()
logger = logging.getLogger("blox-ai.routes.diag")


ToolName = Literal[
    "internet", "relay", "time", "power", "storage", "containers",
    "wireguard", "heartbeat", "events", "readiness", "summary",
    "discovery_state", "systemd_services", "network_interface",
    "uniondrive", "identity_health",
    "kubo_health", "fula_go_health", "image_versions", "ble_state", "plugins",
]


# /diag/bundle budgets. Each tool gets its own deadline; the overall wait
# caps total wall-clock so a single wedged tool can't hang the snapshot.
# Slowest single tool today is diag/relay (~15s swarm-connect), so 18s
# per-tool leaves headroom; 25s overall stays under the app's 30s client
# timeout with room for transport overhead.
BUNDLE_PER_TOOL_TIMEOUT_S = 18.0
BUNDLE_OVERALL_BUDGET_S = 25.0


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


@router.post("/diag/bundle")
async def diag_bundle(request: Request) -> dict:
    """Run every read-only diag tool concurrently and return one snapshot.

    Backs the app's "Raw diagnostics" card. POST (not GET) so the core BLE
    proxy — which always POSTs json={} — can reach it, and so the path
    doesn't collide with the GET /diag/{tool} Literal enum (a GET /bundle
    would 422 against that enum).

    diag/summary is excluded: it internally re-runs a subset of the read
    tools, so bundling it would duplicate work — the bundle is already a
    superset. Every tool is isolated: a per-tool timeout or exception
    becomes an {"error": ...} entry rather than failing the whole snapshot,
    and an overall budget guarantees a response even if a tool wedges.
    """
    executor = request.app.state.tool_executor
    tools = [t for t in known_tools() if t != "diag/summary"]

    async def run_one(tool: str) -> tuple[str, dict]:
        try:
            res = await asyncio.wait_for(
                executor(tool, {}), timeout=BUNDLE_PER_TOOL_TIMEOUT_S
            )
            return tool, res
        except asyncio.TimeoutError:
            return tool, {"error": "timeout", "timeout_s": BUNDLE_PER_TOOL_TIMEOUT_S}
        except Exception as e:  # noqa: BLE001
            logger.warning("diag/bundle: %s impl raised: %s", tool, e)
            return tool, {"error": str(e)[:200]}

    task_to_tool = {asyncio.ensure_future(run_one(t)): t for t in tools}
    done, pending = await asyncio.wait(
        task_to_tool.keys(), timeout=BUNDLE_OVERALL_BUDGET_S
    )

    results: dict[str, dict] = {}
    for task in done:
        tool, res = task.result()
        results[tool.removeprefix("diag/")] = res
    for task in pending:
        task.cancel()
        tool = task_to_tool[task]
        results[tool.removeprefix("diag/")] = {"error": "budget_exceeded"}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tools": results,
    }
