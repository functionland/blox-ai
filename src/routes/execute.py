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
    # Phase 1.e (2026-05-28): also identify the OWNING session so we can
    # (a) check its execution_results cache before re-running, and
    # (b) store the result + append execution_result to the event buffer
    # so resume callers see the post-execute state.
    session_mgr = request.app.state.session_manager
    action_id = body["action_id"]
    action_name = None
    action_args: dict = {}
    owning_session = None
    for s in list(session_mgr._sessions.values()):  # noqa: SLF001
        for rec in getattr(s, "issued_recommendations", {}).items():
            rid, payload = rec
            if rid == action_id:
                action_name = payload["action_name"]
                action_args = payload.get("args", {})
                owning_session = s
                break
        if action_name is not None:
            break

    # Phase 1.e idempotency: if this action_id has a cached successful
    # execution_result, return it instead of re-running. Closes the
    # resume-then-tap-again replay loop (tree runs are deterministic
    # so the buffered recommended_action replays on every SSE reconnect;
    # without this, a re-tap would burn the token's nonce + fail at
    # verify with a confusing "approval_token_already_used" error
    # instead of "this action already ran").
    if owning_session is not None:
        cached = owning_session.execution_results.get(action_id)
        if cached is not None:
            logger.info(
                "execute_action replay: returning cached result for action_id=%s",
                action_id,
            )
            return JSONResponse(status_code=200, content=cached)

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
            # tier defaults to 1 (read; harmless) since we don't know
            # what the original recommendation was; the schema enum
            # forbids 0. rejected_reason carries the truthful cause.
            "tier": 1,
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

    # Phase 1.e: cache successful execution_result on the owning session
    # so a subsequent replay (resume → reducer auto-opens modal → user
    # re-taps approve) returns the cached result instead of burning
    # the token nonce. ALSO append the execution_result to the
    # session's event_buffer so the existing resume protocol replays
    # it naturally — the app's reducer is then aware that the action
    # already ran without needing to look at the cache directly.
    # Only cache success (200 + execution_result event); failures
    # let the user retry.
    if (
        owning_session is not None
        and result.get("http_status") == 200
        and isinstance(result.get("sse_event"), dict)
        and result["sse_event"].get("type") == "execution_result"
    ):
        owning_session.execution_results[action_id] = result["sse_event"]
        try:
            await owning_session.append_event(result["sse_event"])
        except Exception:
            # Buffer append failure shouldn't fail the HTTP request —
            # the cache + executor write are the durable parts.
            logger.exception(
                "failed to append execution_result to session buffer "
                "(action_id=%s); response still returned 200", action_id,
            )

    return JSONResponse(
        status_code=result["http_status"],
        content=result["sse_event"],
    )
