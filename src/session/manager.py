"""SessionManager — in-memory session state for /troubleshoot conversations.

Per parent plan Phase 11:
  - 30-min SLIDING TTL (refreshed on each /user-reply or /phone-context)
  - 50-session cap with LRU eviction
  - LOST on container restart (matches the Phase 10 HMAC-rotation discipline)
  - phone_context held in-memory ONLY — never persisted, never logged raw
  - Per-session reply queue lets the SSE generator await /user-reply

The manager is process-local; container restart wipes everything, which
is the right semantic (the HMAC approval-token secret rotates per
container start; reusing stale session_ids across restarts would be a
cross-restart replay surface).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("blox-ai.session")


DEFAULT_TTL_SEC = 30 * 60  # 30 minutes
DEFAULT_MAX_SESSIONS = 50
# Resume support — per-session event buffer cap. ~1 KB/event × 500 cap ×
# 50 sessions ≈ 25 MB worst-case; comfortable on a device with 7-8 GB
# RAM. On overflow we drop the oldest event AND inject a synthetic
# `thought` truncation marker (see SessionState.append_event) so the
# resuming consumer sees a plain-text note about the gap. Reusing
# `thought` avoids touching the SSE schema for a flow-control concern
# (advisor input: don't grow the schema for transport metadata).
DEFAULT_BUFFER_CAP = 500


@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_active: float
    # Pending question_id from the most-recent user_question event the
    # backend emitted. /user-reply MUST match this; None = no pending
    # question.
    pending_question_id: str | None = None
    # Most-recent phone_context payload (REPLACES on each /phone-context;
    # the model only reasons over the latest snapshot — phone state
    # changes fast). NEVER appears in any log file.
    phone_context: dict | None = None
    # Bridge's reply queue. /user-reply pushes; the bridge awaits.
    # asyncio.Queue is bounded (maxsize=1) so a duplicate /user-reply
    # blocks server-side rather than queueing stale replies.
    reply_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    # C4: recommended_action events the bridge has emitted on this
    # session's stream. Keyed by action_id → {action_name, args}. The
    # /execute-action route reads this to resolve action_name/args
    # for the executor; the HMAC token binds to action_id, the
    # dispatch table needs action_name + args.
    issued_recommendations: dict[str, dict] = field(default_factory=dict)
    # Phase 1.e: successful execution_result events keyed by action_id.
    # The /execute-action route consults this BEFORE calling the
    # executor — if a result is cached, return it (200 + cached sse_event)
    # without re-running. Closes the resume-then-tap-again replay loop
    # that trees expose (LLM didn't trigger this because it doesn't
    # deterministically re-emit recommended_action on replay).
    # Only successful (http_status=200) executions are cached; failures
    # let the user retry.
    execution_results: dict[str, dict] = field(default_factory=dict)
    # ---- Resume support (added 2026-05-28) -----------------------------
    # event_buffer holds (seq, event) tuples in arrival order. seq is
    # monotonic per-session — even on truncation we keep counting up so
    # gaps in seq let the consumer detect dropped events.
    event_buffer: list[tuple[int, dict]] = field(default_factory=list)
    next_seq: int = 0
    # Total events dropped from the head of the buffer due to cap.
    # Cumulative across the session lifetime. Surfaced to the consumer
    # as the marker payload when a resume's `from` is older than the
    # oldest buffered seq.
    dropped_count: int = 0
    # Set True once the generator task finishes (verdict reached, error,
    # or session_cancelled). Consumers exit their wait loop on this.
    generator_done: bool = False
    # The detached generator task. Created on the first /troubleshoot
    # POST for this session; subsequent /resume calls reattach to the
    # same task's output (via the buffer). NEVER awaited from the SSE
    # handler — that's the whole point of the resume feature (SSE
    # consumer disconnect must NOT cancel the generator).
    generator_task: object | None = None
    # Last-wins consumer policy: each new SSE consumer increments this
    # token; consumers loop while their captured token equals the
    # current value, exit when a newer consumer has taken over. Avoids
    # multiple SSE streams writing duplicate frames to the network.
    consumer_generation: int = 0
    # Producer/consumer wake — asyncio.Condition (not Event) per
    # advisor input: Event has the "lost wake between clear and wait"
    # race. Condition's notify_all + acquire pattern is race-free for
    # multiple consumers, and the migration from single-consumer to
    # last-wins doesn't require touching wake logic.
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def append_event(self, event: dict) -> None:
        """Append an event to the buffer + wake any waiting consumer.
        Drops the oldest event on overflow (incrementing dropped_count
        so the next resume can synthesize a truncation marker)."""
        seq = self.next_seq
        self.next_seq += 1
        self.event_buffer.append((seq, event))
        if len(self.event_buffer) > DEFAULT_BUFFER_CAP:
            self.event_buffer.pop(0)
            self.dropped_count += 1
        async with self.cond:
            self.cond.notify_all()

    async def mark_done(self) -> None:
        """Mark the generator as finished + wake all consumers so they
        exit their stream loop. Idempotent — safe to call from the
        generator wrapper's try/except/finally branches."""
        self.generator_done = True
        async with self.cond:
            self.cond.notify_all()


class SessionManager:
    """Process-local session registry. Thread-unsafe by design — uvicorn
    runs one event loop per worker, and the parent plan caps workers at 1."""

    def __init__(
        self,
        ttl_sec: int = DEFAULT_TTL_SEC,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ):
        self.ttl_sec = ttl_sec
        self.max_sessions = max_sessions
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str | None = None) -> SessionState:
        """Mint a fresh session. If session_id is provided (client-supplied),
        use it; else generate a UUIDv4. If we're at cap, evict the LRU
        first."""
        self._prune_expired()
        if len(self._sessions) >= self.max_sessions:
            self._evict_lru()
        sid = session_id or str(uuid.uuid4())
        now = time.monotonic()
        s = SessionState(session_id=sid, created_at=now, last_active=now)
        self._sessions[sid] = s
        return s

    def get(self, session_id: str) -> SessionState | None:
        """Lookup. Returns None if unknown OR expired (expired entries are
        purged on access to keep _sessions tidy)."""
        s = self._sessions.get(session_id)
        if s is None:
            return None
        if self._is_expired(s):
            self._sessions.pop(session_id, None)
            return None
        return s

    def touch(self, session_id: str) -> bool:
        """Slide the TTL forward. Returns True if found + slid, False
        if session is unknown / expired."""
        s = self.get(session_id)
        if s is None:
            return False
        s.last_active = time.monotonic()
        return True

    def remove(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        self._prune_expired()
        return len(self._sessions)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _is_expired(self, s: SessionState) -> bool:
        return (time.monotonic() - s.last_active) > self.ttl_sec

    def _prune_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
        for sid in expired:
            del self._sessions[sid]

    def _evict_lru(self) -> None:
        if not self._sessions:
            return
        lru_sid = min(self._sessions, key=lambda k: self._sessions[k].last_active)
        logger.info("session_evict lru=%s count=%d", lru_sid, len(self._sessions))
        del self._sessions[lru_sid]


# ---------------------------------------------------------------------------
# PII sanitization for log lines (phone_context contains SSID/BSSID/IP)
# ---------------------------------------------------------------------------

import re

_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_BSSID = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")


def sanitize_for_log(s: str, max_len: int = 200) -> str:
    """Aggressively redact IP-shaped + BSSID-shaped substrings before any
    log line that might echo phone_context fields. Defense in depth — the
    server's documented contract is 'never log raw phone_context', and
    this is the last-line guard before a logging.warning() call.

    SSIDs aren't pattern-matched (impossible in general). Field-name
    callers MUST always pass `<redacted>` instead of the raw value when
    the field is wifi_ssid / wifi_bssid / etc.
    """
    out = _IPV4.sub("<ip>", s)
    out = _BSSID.sub("<bssid>", out)
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out
