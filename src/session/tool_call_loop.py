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

In C2 the backend is the MockBackend's scripted sequence. C7 replaces
that with a real RKLLM-driven backend; this bridge stays unchanged.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Awaitable, Callable

import jsonschema


logger = logging.getLogger("blox-ai.bridge")


# ToolExecutor signature: (tool_name, args) → awaitable[payload dict]
ToolExecutor = Callable[[str, dict], Awaitable[dict]]


async def stream_troubleshoot(
    backend_events: AsyncIterator[dict],
    tool_executor: ToolExecutor,
    validator: jsonschema.Draft202012Validator,
) -> AsyncIterator[dict]:
    """Bridge generator. Yields validated event dicts ready to wrap as SSE.

    Schema-invalid events are swallowed and replaced with a synthetic
    error event; this protects the SSE renderer from a backend bug that
    would otherwise crash mid-stream.
    """
    async for event in backend_events:
        if not _validate(event, validator):
            yield _schema_violation_error(event)
            return

        yield event

        if event.get("type") == "tool_call":
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
                # The tool_result we constructed somehow doesn't validate
                # (would mean the executor returned a payload shape we
                # can't carry, OR our error truncation made it invalid).
                # Surface to client as a stream-terminating error.
                yield _schema_violation_error(tool_result)
                return
            yield tool_result


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
