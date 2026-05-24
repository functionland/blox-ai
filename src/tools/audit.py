"""C4 — append-only audit log for /execute-action.

Per fula-ota audit_log_line.schema.json:
  - one JSONL line per /execute-action request (executed OR rejected)
  - 50 MB primary + 5 backup rotation
  - O_APPEND only — never truncate, never seek
  - Schema enforces conditional invariants (executed=true requires
    `result` + empty `rejected_reason`; executed=false requires
    non-empty `rejected_reason` + no `result`)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger("blox-ai.audit")


# 50 MB primary + 5 backups (matches plan's events.jsonl convention).
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5


DEFAULT_AUDIT_LOG_PATH = os.environ.get(
    "BLOX_AI_AUDIT_LOG_PATH",
    "/var/log/fula/ai-actions.jsonl",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _rotate_if_needed(path: Path, max_bytes: int, backup_count: int) -> None:
    """Roll the primary file out to .1, shifting .1→.2, etc., when it
    exceeds max_bytes. Mirrors the standard logrotate pattern but
    in-process (we don't depend on host logrotate running)."""
    try:
        if not path.exists():
            return
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    # Shift backups down
    for i in range(backup_count, 0, -1):
        old = path.with_suffix(path.suffix + f".{i}")
        if i == backup_count:
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        newer = path.with_suffix(path.suffix + f".{i+1}")
        if old.exists():
            try:
                old.replace(newer)
            except OSError:
                pass
    # Move primary → .1
    try:
        path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError as e:
        logger.warning("audit log rotation rename failed: %s", e)


def append(line: dict, path: str | None = None,
           max_bytes: int = DEFAULT_MAX_BYTES,
           backup_count: int = DEFAULT_BACKUP_COUNT) -> bool:
    """Write a single audit line. Returns True on success, False on
    any I/O error (caller MUST map False → HTTP 500 + emit error event)."""
    target = Path(path or DEFAULT_AUDIT_LOG_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("audit dir mkdir failed: %s", e)
        return False
    _rotate_if_needed(target, max_bytes, backup_count)
    try:
        # O_APPEND only — no seek, no truncate. The file handle is
        # opened fresh per call so the kernel's append semantics
        # serialize concurrent writers under O_APPEND (POSIX guarantee
        # for writes <= PIPE_BUF, which 1 JSONL line of <2 KB easily is).
        fd = os.open(str(target),
                     os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(line, separators=(",", ":")) + "\n")
                     .encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except OSError as e:
        logger.error("audit append failed: %s", e)
        return False
