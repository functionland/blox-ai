"""C2+C5 — /troubleshoot SSE + conversational endpoints.

Wraps the bridge in `src.session.tool_call_loop.stream_troubleshoot` in
an SSE response. Each event from the bridge is serialized as a single
`data: <json>\\n\\n` SSE record (per the standard SSE wire format).

C5 adds:
  - POST /troubleshoot/user-reply (matches Phase 11 contract)
  - POST /troubleshoot/phone-context (matches Phase 11 contract)
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.session.manager import sanitize_for_log
from src.session.tool_call_loop import stream_troubleshoot


router = APIRouter()
logger = logging.getLogger("blox-ai.troubleshoot")


class TroubleshootRequest(BaseModel):
    """POST /troubleshoot body."""
    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class UserReplyRequest(BaseModel):
    """POST /troubleshoot/user-reply body. Mirrors fula-ota's
    user_reply_request.schema.json shape."""
    model_config = {"extra": "forbid"}

    session_id: str = Field(min_length=1, max_length=128)
    question_id: str = Field(min_length=1, max_length=128)
    reply_text: str = Field(min_length=1, max_length=4000)


class PhoneContextRequest(BaseModel):
    """POST /troubleshoot/phone-context body. The inner phone_context
    object is validated against the JSON Schema separately (we don't
    duplicate the full pydantic model for a payload-validating phase)."""
    model_config = {"extra": "forbid"}

    session_id: str = Field(min_length=1, max_length=128)
    phone_context: dict


def _error_body(code: str, detail: str = "") -> dict:
    """Standard 4xx body shape per fula-ota api/README.md Phase 11."""
    out = {"error": code}
    if detail:
        out["detail"] = detail
    return out


@router.post("/troubleshoot")
async def troubleshoot(req: TroubleshootRequest, request: Request) -> Response:
    backend = request.app.state.backend
    tool_executor = request.app.state.tool_executor
    validator = request.app.state.schemas.validator_for("sse_events.schema.json")
    session_mgr = request.app.state.session_manager

    # Resolve session: caller-supplied wins; else mint new.
    session = None
    if req.session_id:
        session = session_mgr.get(req.session_id)
        if session is None:
            # Caller passed an unknown session_id — create one with that
            # id so the SSE session_started event echoes back the same id
            # (matches Phase 11 caller-supplied behaviour).
            session = session_mgr.create(session_id=req.session_id)
    else:
        session = session_mgr.create()

    backend_events = backend.run_troubleshoot(
        prompt=req.prompt,
        session_id=session.session_id,
    )

    backend_handles_tools = getattr(backend, "consumes_tool_results", False)

    async def sse_stream():
        try:
            async for event in stream_troubleshoot(
                backend_events, tool_executor, validator,
                session=session,
                backend_handles_tools=backend_handles_tools,
            ):
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        except Exception:
            logger.exception("unexpected bridge failure")
            fallback = {
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": "unexpected bridge failure",
                "recoverable": False,
            }
            yield f"data: {json.dumps(fallback, separators=(',', ':'))}\n\n"
        finally:
            # Slide TTL on stream completion so a session that just
            # finished a turn doesn't expire instantly.
            session_mgr.touch(session.session_id)

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/troubleshoot/user-reply")
async def user_reply(req: UserReplyRequest, request: Request) -> Response:
    """Phase 11 contract: the app submits this when the user answers a
    user_question event. Container looks up session, validates
    question_id matches the most-recent pending, pushes reply into the
    bridge's queue, returns 200."""
    session_mgr = request.app.state.session_manager
    session = session_mgr.get(req.session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("session_not_found"),
        )
    # Phase 11 contract: question_id MUST match the currently-pending.
    # Idempotency: a repeat with the SAME id is a no-op + 200 (BLE retry
    # could resend the same reply).
    if session.pending_question_id is None:
        return JSONResponse(
            status_code=400,
            content=_error_body("question_id_mismatch",
                                "no pending question on this session"),
        )
    if req.question_id != session.pending_question_id:
        return JSONResponse(
            status_code=400,
            content=_error_body("question_id_mismatch",
                                f"expected {session.pending_question_id!r}"),
        )
    # Slide TTL on activity
    session_mgr.touch(session.session_id)
    # Push reply; the bridge's wait_for will unblock. If the queue is full
    # (caller retried while the bridge hasn't drained yet), don't block
    # the HTTP request — treat as idempotent 200.
    try:
        session.reply_queue.put_nowait({
            "question_id": req.question_id,
            "text": req.reply_text,
        })
    except Exception:
        logger.info("reply_queue full for session=%s (idempotent retry?)",
                    session.session_id)
    return JSONResponse(status_code=200, content={})


@router.post("/troubleshoot/phone-context")
async def phone_context(req: PhoneContextRequest, request: Request) -> Response:
    """Phase 11 contract: in-memory only. Validates body against
    phone_context.schema.json (the inner shape). On validation error,
    sanitize SSID/BSSID/IP before any log line."""
    schemas = request.app.state.schemas
    session_mgr = request.app.state.session_manager
    session = session_mgr.get(req.session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("session_not_found"),
        )
    pc_validator = schemas.validator_for("phone_context.schema.json")
    errors = list(pc_validator.iter_errors(req.phone_context))
    if errors:
        # PII-aware log: sanitize the first error's message before
        # touching the operator log line. The whole payload is NOT logged.
        first = errors[0]
        logger.warning(
            "phone_context_invalid session=%s path=%s msg=%s",
            session.session_id,
            list(first.path),
            sanitize_for_log(first.message, max_len=200),
        )
        return JSONResponse(
            status_code=400,
            content=_error_body("phone_context_invalid"),
        )
    # In-memory only — replaces any prior snapshot.
    session.phone_context = req.phone_context
    session_mgr.touch(session.session_id)
    return JSONResponse(status_code=200, content={})
