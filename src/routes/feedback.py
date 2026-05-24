"""C6 — POST /feedback.

Per fula-ota/.../api/README.md Phase 16 contract:
  - body validates against feedback_request.schema.json
  - session-detached acceptance: 200 even if session was evicted
  - one JSONL line per request to /var/log/fula/ai-feedback.jsonl
  - comment CR/LF stripped before write (log-injection defense)
  - anonymized_transcript_uploaded defaults to false on every line
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse


router = APIRouter()
logger = logging.getLogger("blox-ai.feedback")


DEFAULT_FEEDBACK_LOG_PATH = "/var/log/fula/ai-feedback.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _sanitize_comment(s: str) -> str:
    """Strip CR/LF (log-injection defense) and cap at the schema's 2000
    char ceiling. Returns empty string if nothing useful is left."""
    out = s.replace("\r", " ").replace("\n", " ").strip()
    return out[:2000]


def _feedback_log_path(request: Request) -> str:
    """Allow tests to override via app.state.feedback_log_path."""
    return getattr(
        request.app.state,
        "feedback_log_path",
        DEFAULT_FEEDBACK_LOG_PATH,
    )


def _verdict_summary_from_session(session) -> str:
    """Pull verdict.payload.summary from the session if the bridge stored
    one. C5 doesn't currently capture this; C6+ can extend SessionState
    to remember the last verdict event. Empty string is the valid
    'session detached / no verdict yet' value per the log schema."""
    if session is None:
        return ""
    summary = getattr(session, "last_verdict_summary", None)
    return summary if isinstance(summary, str) else ""


def _actions_taken_from_session(session) -> list:
    """C6 doesn't yet capture executed actions per session (that's C4's
    audit log surface). Empty list is schema-valid."""
    return []


@router.post("/feedback")
async def feedback(request: Request) -> Response:
    schemas = request.app.state.schemas
    session_mgr = request.app.state.session_manager
    req_validator = schemas.validator_for("feedback_request.schema.json")
    log_validator = schemas.validator_for("feedback_log_line.schema.json")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body_invalid"})

    try:
        req_validator.validate(body)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body_invalid"})

    session_id = body["session_id"]
    rating = body["rating"]
    comment_in = body.get("comment")

    # Session may have been evicted; per Phase 16 contract still accept.
    session = session_mgr.get(session_id)
    if session is not None:
        session_mgr.touch(session_id)

    line: dict = {
        "ts": _now_iso(),
        "session_id": session_id,
        "user_rating": rating,
        "verdict_summary": _verdict_summary_from_session(session),
        "actions_taken": _actions_taken_from_session(session),
        "anonymized_transcript_uploaded": False,
    }
    if isinstance(comment_in, str):
        sanitized = _sanitize_comment(comment_in)
        if sanitized:
            line["comment"] = sanitized

    try:
        log_validator.validate(line)
    except Exception as e:
        logger.error("feedback log line failed our own schema: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error"},
        )

    log_path = _feedback_log_path(request)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError as e:
        logger.error("feedback append failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error"},
        )

    return JSONResponse(status_code=200, content={})
