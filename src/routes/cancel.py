"""C6 — POST /cancel.

Per fula-ota /troubleshoot contract: terminates an in-flight session.
The bridge in tool_call_loop.py awaits `session.reply_queue.get()` when
the model is paused on a user_question. /cancel puts a sentinel value
onto the queue + removes the session from the manager — the bridge
sees the sentinel, emits an error event, and returns.

Sessions that aren't currently paused on a user_question simply get
removed from the manager (in-flight tool_call evaluation will complete
naturally; the next /user-reply for the cancelled session would 404).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


router = APIRouter()
logger = logging.getLogger("blox-ai.cancel")


# Sentinel value the bridge recognises as "cancelled". Cannot collide
# with a real user reply because real replies are dicts; this is a class
# instance we compare with `is`.
class _CancelSentinel:
    pass


CANCEL_SENTINEL = _CancelSentinel()


class CancelRequest(BaseModel):
    model_config = {"extra": "forbid"}
    session_id: str = Field(min_length=1, max_length=128)


@router.post("/cancel")
async def cancel(req: CancelRequest, request: Request) -> Response:
    mgr = request.app.state.session_manager
    session = mgr.get(req.session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"error": "session_not_found"},
        )
    # If the bridge is paused on a user_question, push the sentinel so
    # it wakes up. If it's not paused, the put_nowait fills the queue
    # (size=1) for a hypothetical future await; the session is removed
    # anyway so no real harm.
    try:
        session.reply_queue.put_nowait(CANCEL_SENTINEL)
    except Exception:
        pass
    mgr.remove(req.session_id)
    return JSONResponse(status_code=200, content={})
