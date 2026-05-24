"""Runbook loader — loads + reloads runbook.md via SIGHUP.

Reads from the bind-mounted /usr/bin/fula/ai/runbook.md by default
(configurable via BLOX_AI_RUNBOOK_PATH for dev). Parses the frontmatter
via the vendored runbook_frontmatter parser. Refuses SIGHUP swaps that
violate:
  - downgrade-protection (new runbook_version must be strictly greater)
  - schema-version bumps (forces full container restart)

Emits events on every reload outcome to /var/log/fula/events.jsonl so
operators can see whether their OTA-pushed runbook landed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from src.runtime.runbook_frontmatter import (
    RunbookFrontmatter,
    RunbookFrontmatterError,
    parse,
)


logger = logging.getLogger("blox-ai.runbook")


DEFAULT_RUNBOOK_PATH = "/usr/bin/fula/ai/runbook.md"
DEFAULT_EVENTS_LOG_PATH = "/var/log/fula/events.jsonl"


class RunbookLoader:
    """Holds the in-memory runbook text + metadata. Thread-safe (the
    SIGHUP handler runs on the main thread; reads can come from any
    request handler async-task)."""

    def __init__(
        self,
        path: str = DEFAULT_RUNBOOK_PATH,
        events_log_path: str = DEFAULT_EVENTS_LOG_PATH,
    ):
        self.path = path
        self.events_log_path = events_log_path
        self._lock = threading.Lock()
        self._text: str = ""
        self._frontmatter: Optional[RunbookFrontmatter] = None

    def load_initial(self) -> bool:
        """Load runbook at container start. Returns False if the file
        is missing — the container can still run (mock backend doesn't
        need a runbook), but logs WARNING."""
        try:
            text = Path(self.path).read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("runbook not present at startup: %s", e)
            return False
        try:
            fm = parse(text)
        except RunbookFrontmatterError as e:
            logger.error("runbook present but malformed at startup: %s", e)
            return False
        with self._lock:
            self._text = text
            self._frontmatter = fm
        logger.info(
            "runbook loaded: runbook_version=%d schema_version=%d",
            fm.runbook_version, fm.schema_version,
        )
        self._emit_event("runbook_loaded", {
            "runbook_version": fm.runbook_version,
            "schema_version": fm.schema_version,
        })
        return True

    def reload(self) -> dict:
        """SIGHUP handler. Re-reads runbook + validates. Returns a result
        dict with `outcome` in {accepted, refused_malformed,
        refused_schema, refused_downgrade}. Logged to events.jsonl
        regardless of outcome."""
        try:
            text = Path(self.path).read_text(encoding="utf-8")
        except OSError as e:
            return self._refuse("refused_missing", str(e))
        try:
            fm = parse(text)
        except RunbookFrontmatterError as e:
            return self._refuse("refused_malformed", str(e))
        with self._lock:
            current = self._frontmatter
        try:
            is_newer = fm.is_newer_than(current)
        except RunbookFrontmatterError as e:
            return self._refuse("refused_schema", str(e))
        if not is_newer:
            return self._refuse(
                "refused_downgrade",
                f"new runbook_version {fm.runbook_version} <= "
                f"loaded {current.runbook_version if current else 0}",
            )
        # Accepted — swap atomically
        with self._lock:
            old_version = current.runbook_version if current else 0
            self._text = text
            self._frontmatter = fm
        result = {
            "outcome": "accepted",
            "old_runbook_version": old_version,
            "new_runbook_version": fm.runbook_version,
        }
        logger.info("runbook reloaded: %d -> %d",
                    old_version, fm.runbook_version)
        self._emit_event("runbook_reload", result)
        return result

    def get_text(self) -> str:
        with self._lock:
            return self._text

    def get_version(self) -> int:
        with self._lock:
            return self._frontmatter.runbook_version if self._frontmatter else 0

    # ------------------------------------------------------------------

    def _refuse(self, outcome: str, detail: str) -> dict:
        result = {"outcome": outcome, "detail": detail[:500]}
        logger.warning("runbook reload %s: %s", outcome, detail[:200])
        self._emit_event("runbook_reload", result)
        return result

    def _emit_event(self, category: str, detail_obj: dict) -> None:
        """Append to events.jsonl. Best-effort — never raises."""
        try:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            line = json.dumps({
                "ts": ts,
                "category": category,
                "detail": json.dumps(detail_obj, separators=(",", ":"))[:2000],
            })
            os.makedirs(os.path.dirname(self.events_log_path), exist_ok=True)
            with open(self.events_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning("could not write events.jsonl: %s", e)
