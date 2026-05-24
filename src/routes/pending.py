"""C6 — GET /pending.

Returns the most recent line from /var/log/fula/ai-pending-actions.jsonl
(written by Phase 14's isolation_mode.py on the host). The fula-ota
plugin's BLE proxy exposes this as `ai/pending`, which the apps/box
PendingActionsPanel (Phase 15) reads.

Returns {} when the log file doesn't exist OR is empty — the app's
parsePendingResponse helper treats this as "no pending recommendations".
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse


router = APIRouter()
logger = logging.getLogger("blox-ai.pending")


DEFAULT_PENDING_LOG_PATH = "/var/log/fula/ai-pending-actions.jsonl"


def _pending_log_path(request: Request) -> str:
    return getattr(
        request.app.state,
        "pending_log_path",
        DEFAULT_PENDING_LOG_PATH,
    )


@router.get("/pending")
async def pending(request: Request) -> Response:
    path = _pending_log_path(request)
    p = Path(path)
    if not p.is_file():
        return JSONResponse(status_code=200, content={})
    try:
        # Read last 4KB and find the last newline-terminated line. Each
        # JSONL entry is rare (isolation mode fires every 6h max) and
        # small; bounded read keeps us cheap.
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - 4096))
            chunk = f.read().decode("utf-8", errors="replace")
        # Last non-empty line
        last = ""
        for line in reversed(chunk.splitlines()):
            if line.strip():
                last = line
                break
        if not last:
            return JSONResponse(status_code=200, content={})
        try:
            obj = json.loads(last)
        except json.JSONDecodeError:
            logger.warning("pending log last line is malformed JSON")
            return JSONResponse(status_code=200, content={})
        return JSONResponse(status_code=200, content=obj)
    except OSError as e:
        logger.warning("pending log read failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "io_error"})
