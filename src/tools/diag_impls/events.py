"""diag/events — tail of /var/log/fula/events.jsonl."""
from __future__ import annotations

import json
import logging
from pathlib import Path


logger = logging.getLogger("blox-ai.diag.events")


EVENTS_LOG_PATH = "/var/log/fula/events.jsonl"
DEFAULT_TAIL_N = 50


def diag_events(tail_n: int = DEFAULT_TAIL_N) -> dict:
    """Returns the last N parseable JSON lines from events.jsonl. Lines
    that fail to parse are SKIPPED (not surfaced) — the schema requires
    every event to have ts+category+detail."""
    p = Path(EVENTS_LOG_PATH)
    if not p.is_file():
        return {"events": []}
    try:
        # Read whole file then take tail. events.jsonl rotates at 50 MB
        # per Phase 1.8 + we typically tail far less; cost is bounded.
        with p.open(encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except OSError as e:
        logger.warning("events.jsonl read failed: %s", e)
        return {"events": []}
    events: list[dict] = []
    for raw in raw_lines[-tail_n * 3:]:  # over-read for malformed-line headroom
        if len(events) >= tail_n:
            break
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(ev, dict)
            and isinstance(ev.get("ts"), str)
            and isinstance(ev.get("category"), str)
            and isinstance(ev.get("detail"), str)
        ):
            events.append({
                "ts": ev["ts"],
                "category": ev["category"][:64],
                "detail": ev["detail"][:2000],
            })
    # Final cap (schema doesn't actually cap items, but be defensive)
    return {"events": events[-tail_n:]}
