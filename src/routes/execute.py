"""C4 — POST /execute-action.

Thin route handler. All trust-boundary logic lives in ActionExecutor.
This route:
  1. Validates the request body against execute_action_request.schema.json
     (400 body_invalid on failure)
  2. Looks up the `recommended_action` event the action_id was issued for
     (the bridge / SessionState retains it; if missing → 401-equivalent
     bound to "approval_token_invalid" since the token wouldn't bind
     anyway)
  3. Calls executor.execute(...) which returns the HTTP status to use
     + the audit-log line (already written) + the SSE-shaped event body
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse


router = APIRouter()
logger = logging.getLogger("blox-ai.execute")


@router.post("/execute-action")
async def execute_action(request: Request) -> Response:
    schemas = request.app.state.schemas
    validator = schemas.validator_for("execute_action_request.schema.json")
    executor = getattr(request.app.state, "action_executor", None)
    if executor is None:
        return JSONResponse(
            status_code=500,
            content={"error": "executor_not_initialised"},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body_invalid"})

    try:
        validator.validate(body)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body_invalid"})

    # Look up the recommended_action the token was issued for. The bridge
    # tracks issued recommendations on the session; for sessions that have
    # been evicted, we treat as "token was issued but session is gone" —
    # the HMAC still verifies but we have no action_name/args to dispatch.
    # Fall back to reading them from the request body's optional extras
    # (not currently exposed in the schema). Practical resolution: include
    # action_name + args in the recommendation lookup off the SESSION's
    # issued list (set by the bridge when it emitted the recommendation).
    session_mgr = request.app.state.session_manager
    action_id = body["action_id"]
    action_name = None
    action_args: dict = {}
    for s in list(session_mgr._sessions.values()):  # noqa: SLF001
        for rec in getattr(s, "issued_recommendations", {}).items():
            rid, payload = rec
            if rid == action_id:
                action_name = payload["action_name"]
                action_args = payload.get("args", {})
                break
        if action_name is not None:
            break
    if action_name is None:
        # No session has this recommendation in memory. HMAC may verify
        # cleanly — but without action_name + args we can't dispatch.
        # Truthful audit reason is `recommendation_not_found` (NOT
        # approval_token_invalid — that would lie to the operator).
        from src.tools.audit import append as audit_append, now_iso
        audit_line = {
            "ts": now_iso(),
            "request_id": "no-session",
            "action_id": action_id,
            "action": "<unknown>",
            "args": {},
            "tier": 0,
            "approval_token_valid": False,
            "security_code_required": False,
            "executed": False,
            "rejected_reason": "recommendation_not_found",
            "approver_transport": "ble",
            "duration_ms": 0,
            "executor_version": "0.1.0",
            "whitelist_hash": executor.whitelist.sha256_hex,
            "error": "no in-memory recommendation matches this action_id; "
                     "session may have been evicted or container restarted",
        }
        audit_append(audit_line, path=executor.audit_path)
        return JSONResponse(
            status_code=409,
            content={"error": "recommendation_not_found"},
        )

    result = await executor.execute(
        action_id=action_id,
        approval_token=body["approval_token"],
        security_code=body.get("security_code"),
        action_name=action_name,
        action_args=action_args,
        approver_transport="ble",
    )
    return JSONResponse(
        status_code=result["http_status"],
        content=result["sse_event"],
    )
