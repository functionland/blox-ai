"""The model↔tool-executor bridge.

C2 ships a simple async-iterator bridge that:
  1. Pulls events from the backend's async generator.
  2. Validates every event against sse_events.schema.json before yielding
     it to the SSE stream. Schema-invalid events become a synthetic
     `error` event with code='SCHEMA_VIOLATION', recoverable=false.
  3. Intercepts `tool_call` events, runs the supplied executor, then
     yields a matching `tool_result` event with the same call_id.
  4. Executor failures map cleanly to tool_result.ok=false + error
     field (so the model can read the error and continue reasoning).

C5 extends the bridge to handle `user_question` events:
  - records pending_question_id on the session
  - PAUSES the SSE stream (awaits the session's reply_queue)
  - on reply, yields a `user_reply_received` event then resumes

In C5 the backend is still the MockBackend's scripted sequence. C7
replaces that with a real RKLLM-driven backend; this bridge stays
unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Awaitable, Callable

import jsonschema

from src.session.manager import SessionState


logger = logging.getLogger("blox-ai.bridge")


# Max wall-clock the bridge will wait for a /user-reply before giving up.
# Generous because some humans take a while to read + type. Aligns with
# the parent plan's 30-min session TTL but caps a single question well
# under that so a forgotten chat doesn't hold an SSE connection open
# for the full TTL.
USER_REPLY_WAIT_SEC = 10 * 60


# ToolExecutor signature: (tool_name, args) → awaitable[payload dict]
ToolExecutor = Callable[[str, dict], Awaitable[dict]]


async def stream_troubleshoot(
    backend_events: AsyncIterator[dict],
    tool_executor: ToolExecutor,
    validator: jsonschema.Draft202012Validator,
    session: SessionState | None = None,
) -> AsyncIterator[dict]:
    """Bridge generator. Yields validated event dicts ready to wrap as SSE.

    Schema-invalid events are swallowed and replaced with a synthetic
    error event; this protects the SSE renderer from a backend bug that
    would otherwise crash mid-stream.

    When `session` is None, user_question events are passed through
    without waiting — callers that don't use SessionManager get the
    pre-C5 behaviour (model asks, no reply machinery).
    """
    async for event in backend_events:
        if not _validate(event, validator):
            yield _schema_violation_error(event)
            return

        yield event

        evtype = event.get("type")

        if evtype == "tool_call":
            try:
                payload = await tool_executor(
                    event["payload"]["tool"],
                    event["payload"]["args"],
                )
                tool_result = {
                    "type": "tool_result",
                    "call_id": event["call_id"],
                    "ok": True,
                    "payload": payload,
                }
            except Exception as e:
                tool_result = {
                    "type": "tool_result",
                    "call_id": event["call_id"],
                    "ok": False,
                    "payload": None,
                    "error": _truncate(str(e), 2000),
                }
            if not _validate(tool_result, validator):
                yield _schema_violation_error(tool_result)
                return
            yield tool_result

        elif evtype == "recommended_action" and session is not None:
            # C4: stash for /execute-action lookup. The HMAC token
            # binds to action_id; the executor dispatcher needs the
            # full {action_name, args} pair to actually run anything.
            session.issued_recommendations[event["action_id"]] = {
                "action_name": event["action_name"],
                "args": event.get("args") or {},
                "tier": event["tier"],
            }

        elif evtype == "user_question" and session is not None:
            # Phase 11 contract: PAUSE the stream until /user-reply lands.
            session.pending_question_id = event["question_id"]
            try:
                reply = await asyncio.wait_for(
                    session.reply_queue.get(),
                    timeout=USER_REPLY_WAIT_SEC,
                )
            except asyncio.TimeoutError:
                yield {
                    "type": "error",
                    "code": "USER_REPLY_TIMEOUT",
                    "message": (
                        f"timed out waiting for user reply to question "
                        f"{event['question_id']!r} "
                        f"after {USER_REPLY_WAIT_SEC}s"
                    ),
                    "recoverable": False,
                }
                return
            # C6: /cancel pushes a sentinel onto the queue. Identify it
            # via duck-typing (avoids a circular import on the route
            # module).
            if reply.__class__.__name__ == "_CancelSentinel":
                yield {
                    "type": "error",
                    "code": "SESSION_CANCELLED",
                    "message": "session cancelled by /cancel",
                    "recoverable": False,
                }
                return
            ack = {
                "type": "user_reply_received",
                "question_id": event["question_id"],
                "session_id": session.session_id,
            }
            if not _validate(ack, validator):
                yield _schema_violation_error(ack)
                return
            yield ack
            session.pending_question_id = None
            _ = reply


def _validate(event: dict, validator: jsonschema.Draft202012Validator) -> bool:
    """Try-validate. Log + return False on failure (so the bridge can
    convert into an error event without letting the exception escape)."""
    try:
        validator.validate(event)
        return True
    except jsonschema.ValidationError as e:
        logger.warning("SSE event failed schema validation: type=%s err=%s",
                       event.get("type"), str(e)[:200])
        return False


def _schema_violation_error(offending: dict) -> dict:
    return {
        "type": "error",
        "code": "SCHEMA_VIOLATION",
        "message": (
            f"backend emitted invalid event "
            f"(type={offending.get('type', '<missing>')!s})"
        ),
        "recoverable": False,
    }


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else (s[: n - 1] + "…")
