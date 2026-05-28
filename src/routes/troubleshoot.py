"""C2+C5 — /troubleshoot SSE + conversational endpoints.

Wraps the bridge in `src.session.tool_call_loop.stream_troubleshoot` in
an SSE response. Each event from the bridge is serialized as a single
`data: <json>\\n\\n` SSE record (per the standard SSE wire format).

C5 adds:
  - POST /troubleshoot/user-reply (matches Phase 11 contract)
  - POST /troubleshoot/phone-context (matches Phase 11 contract)

2026-05-28 resume support:
  - POST /troubleshoot now starts the generator as a detached
    `asyncio.create_task`; the SSE handler is just a buffer reader.
    Disconnecting the SSE consumer no longer cancels the model run.
  - GET /troubleshoot/resume?session_id=X&from=N reattaches to the
    same task's buffered output. App-side persistence (lastEventSeq
    in AsyncStorage) plus an AppState foreground subscriber makes
    background-then-resume seamless.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.session.manager import SessionState, sanitize_for_log
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


async def _drive_generator_into_buffer(
    session: SessionState,
    backend,
    tool_executor,
    validator,
    prompt: str,
) -> None:
    """Run the bridge generator detached from any SSE consumer, writing
    each event into the session's buffer. SSE consumers come and go via
    `_stream_from_buffer`; this task is the SOLE writer. Survives
    consumer disconnect — that's the whole point of the resume feature.

    On exception we still try to append a synthetic error event so the
    consumer can render the failure instead of staring at a hanging
    chat. `mark_done` is in the finally so even hard exceptions release
    the consumers' `cond.wait()`.
    """
    backend_handles_tools = getattr(backend, "consumes_tool_results", False)
    try:
        backend_events = backend.run_troubleshoot(
            prompt=prompt,
            session_id=session.session_id,
        )
        async for event in stream_troubleshoot(
            backend_events, tool_executor, validator,
            session=session,
            backend_handles_tools=backend_handles_tools,
        ):
            await session.append_event(event)
    except asyncio.CancelledError:
        # Container shutdown or explicit cancel — propagate after
        # marking done so consumers exit.
        await session.mark_done()
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("background generator failed session=%s", session.session_id)
        try:
            await session.append_event({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": str(e)[:200],
                "recoverable": False,
            })
        except Exception:
            pass  # last-ditch; buffer write itself shouldn't crash
    finally:
        await session.mark_done()


async def _stream_from_buffer(
    session: SessionState,
    from_seq: int,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings from the session's event buffer
    starting at `from_seq`, blocking on `session.cond` for new events
    until the generator marks itself done OR a newer consumer claims
    this session (last-wins policy).

    SSE `id:` field carries the seq number so the client's
    EventSource records it on lastEventId; the client persists this to
    AsyncStorage and supplies it as `?from=` on the next reconnect.

    Truncation marker: if `from_seq` is less than the oldest buffered
    seq (the consumer was away long enough for events to fall off the
    cap), inject a synthetic `thought` event with the dropped count.
    Per advisor input we don't grow the SSE schema for a flow-control
    concern — a `thought` event with the marker text is enough; the
    chat surface renders it as italic gray prose."""
    my_generation = session.consumer_generation
    last_yielded_seq = from_seq - 1

    # Truncation detection — fires when the consumer is asking for an
    # event older than what's still in the buffer. Includes the
    # from_seq=0 case (fresh resume after the head of the buffer
    # already overflowed).
    if (
        session.event_buffer
        and session.event_buffer[0][0] > from_seq
    ):
        gap = session.event_buffer[0][0] - from_seq
        marker = {
            "type": "thought",
            "payload": (
                f"[resume] {gap} earlier event(s) dropped from the on-device "
                f"buffer (cap 500). Newer events from this session continue below."
            ),
        }
        # Emit the marker with a special id=-1 (no real seq) so the
        # client's lastEventId tracker doesn't try to use it as a
        # resume offset later.
        yield f"id: -1\ndata: {json.dumps(marker, separators=(',', ':'))}\n\n"

    while True:
        # Yield any buffered events strictly newer than last_yielded.
        # Materialize so we don't hold the buffer reference across the
        # await (the buffer is mutated by the producer task).
        new_events = [
            (s, e) for s, e in session.event_buffer if s > last_yielded_seq
        ]
        for s, e in new_events:
            yield f"id: {s}\ndata: {json.dumps(e, separators=(',', ':'))}\n\n"
            last_yielded_seq = s

        # Check exit conditions BEFORE waiting (covers the case where
        # mark_done already fired or a new consumer already took over).
        if session.generator_done:
            return
        if session.consumer_generation != my_generation:
            # Last-wins: another /troubleshoot or /resume call took
            # over this session. Quietly exit so the network frame
            # stream isn't doubled. The newer consumer reads the same
            # buffer and picks up from from_seq.
            return

        async with session.cond:
            await session.cond.wait()


@router.post("/troubleshoot")
async def troubleshoot(req: TroubleshootRequest, request: Request) -> Response:
    backend = request.app.state.backend
    tool_executor = request.app.state.tool_executor
    validator = request.app.state.schemas.validator_for("sse_events.schema.json")
    session_mgr = request.app.state.session_manager

    # Resolve session: caller-supplied wins; else mint new.
    session: SessionState | None = None
    if req.session_id:
        session = session_mgr.get(req.session_id)
        if session is None:
            # Caller passed an unknown session_id — create one with that
            # id so the SSE session_started event echoes back the same id
            # (matches Phase 11 caller-supplied behaviour).
            session = session_mgr.create(session_id=req.session_id)
    else:
        session = session_mgr.create()

    # If this session already has a running generator (caller retried
    # POST on the same session_id while a previous task is still
    # active), reject. Resume via GET /troubleshoot/resume instead so
    # we don't spawn duplicate writers into one buffer.
    if (
        session.generator_task is not None
        and not getattr(session.generator_task, "done", lambda: True)()
        and not session.generator_done
    ):
        return JSONResponse(
            status_code=409,
            content=_error_body(
                "session_already_active",
                "use GET /troubleshoot/resume?session_id=...&from=N to reattach",
            ),
        )

    # Reset per-session buffer state for the new generator. We keep the
    # SessionState (so phone_context, reply_queue, etc. survive) but
    # wipe the conversation buffer so the new prompt starts at seq 0.
    session.event_buffer = []
    session.next_seq = 0
    session.dropped_count = 0
    session.generator_done = False
    session.consumer_generation += 1
    my_generation = session.consumer_generation

    session.generator_task = asyncio.create_task(
        _drive_generator_into_buffer(
            session, backend, tool_executor, validator, req.prompt,
        ),
        name=f"blox-ai-generator-{session.session_id}",
    )

    async def sse_stream():
        # consumer_generation was bumped above; capture it for the
        # last-wins check in _stream_from_buffer.
        session.consumer_generation = my_generation
        try:
            async for chunk in _stream_from_buffer(session, from_seq=0):
                yield chunk
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


@router.get("/troubleshoot/resume")
async def troubleshoot_resume(
    request: Request,
    session_id: str = Query(min_length=1, max_length=128),
    from_seq: int = Query(alias="from", default=0, ge=0),
) -> Response:
    """Reattach to a /troubleshoot session's existing generator output.
    Returns 404 if the session was evicted (TTL, LRU, or container
    restart); the client clears its persisted state on 404 + offers
    Start-new-chat.

    Replays buffered events newer than `from_seq`, injects a
    truncation marker if `from_seq` is older than the oldest buffered
    event, then blocks on the session's cond for new events until the
    generator marks itself done or a newer consumer takes over.

    Idempotent: multiple consumers can call /resume; last-wins kicks
    the older consumer (it exits cleanly without re-emitting events).
    """
    session_mgr = request.app.state.session_manager
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("session_not_found"),
        )
    # Slide TTL — resume IS user activity.
    session_mgr.touch(session.session_id)
    # Last-wins: bump generation so any prior consumer's loop exits.
    session.consumer_generation += 1

    async def sse_stream():
        try:
            async for chunk in _stream_from_buffer(session, from_seq=from_seq):
                yield chunk
        finally:
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
    sanitize SSID/BSSID/IP before any log line.

    Auto-creates the session if the supplied session_id doesn't exist
    (idempotent prime-then-troubleshoot pattern). This handles two
    real-world cases observed in production:
      1. User taps "Share my phone's context" BEFORE starting a
         /troubleshoot session — the app has generated a UUID and
         wants to attach the phone snapshot to it; the next
         /troubleshoot with that same session_id will find the
         pre-primed session with phone_context already loaded.
      2. Container restarted between sessions — the app's cached
         session_id was valid before the restart but the in-memory
         session map is now empty. Auto-create avoids a confusing
         "Session not found" error in the UI; the user gets the same
         outcome as starting fresh, just labelled with the existing
         session_id.
    """
    schemas = request.app.state.schemas
    session_mgr = request.app.state.session_manager
    session = session_mgr.get(req.session_id)
    if session is None:
        # Don't 404 — create the session with the supplied id. Slide
        # TTL via touch() at the end so the auto-created session has
        # a full 30 min before pruning.
        logger.info(
            "session_auto_created_via_phone_context session=%s",
            req.session_id,
        )
        session = session_mgr.create(session_id=req.session_id)
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
