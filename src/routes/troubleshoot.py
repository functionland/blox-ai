"""C2 — POST /troubleshoot SSE skeleton.

Wraps the bridge in `src.session.tool_call_loop.stream_troubleshoot` in
an SSE response. Each event from the bridge is serialized as a single
`data: <json>\\n\\n` SSE record (per the standard SSE wire format).

C5 will add /troubleshoot/user-reply + /troubleshoot/phone-context and
introduce session-state look-up; this file gets those routes appended.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.session.tool_call_loop import stream_troubleshoot


router = APIRouter()
logger = logging.getLogger("blox-ai.troubleshoot")


class TroubleshootRequest(BaseModel):
    """POST /troubleshoot body.

    Closed (extra=forbid) so a typo'd field surfaces immediately rather
    than being silently dropped server-side.
    """
    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


@router.post("/troubleshoot")
async def troubleshoot(req: TroubleshootRequest, request: Request) -> Response:
    backend = request.app.state.backend
    tool_executor = request.app.state.tool_executor
    validator = request.app.state.schemas.validator_for("sse_events.schema.json")

    backend_events = backend.run_troubleshoot(
        prompt=req.prompt,
        session_id=req.session_id,
    )

    async def sse_stream():
        try:
            async for event in stream_troubleshoot(
                backend_events, tool_executor, validator,
            ):
                # Standard SSE wire format: data: <json>\n\n
                # JSON serialization uses compact separators so a single
                # event fits one network frame in the common case.
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        except Exception as e:
            # Last-resort guard. If the bridge itself raises (it shouldn't
            # for documented failure modes), emit one final error event
            # so the client doesn't hang on a half-closed stream.
            logger.exception("unexpected bridge failure")
            fallback = {
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": "unexpected bridge failure",
                "recoverable": False,
            }
            yield f"data: {json.dumps(fallback, separators=(',', ':'))}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent any intermediate proxy from buffering the stream
            # (kills the live-progress UX in the app's chat transcript).
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
