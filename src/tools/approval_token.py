"""C4 — HMAC approval-token sign + verify.

Per fula-ota api/README.md Phase 10 contract:

Token wire format:
    base64url(json({action_id, expires_at, nonce, hmac}))

Where:
    - action_id: matches the recommended_action event's action_id
    - expires_at: ISO-8601 UTC, now + 300 s (5-min window)
    - nonce: secrets.token_urlsafe(16) (~22 chars; >128 bits entropy)
    - hmac: hex SHA-256 HMAC of canonical-json({action_id, expires_at, nonce})
            using a 32-byte secret from /run/fula-ai/approval-secret

Secret discipline:
    - Generated at container start (32 bytes from os.urandom).
    - Written to /run/fula-ai/approval-secret mode 0600.
    - LOST on container restart (matches Phase 11 session-state TTL).
    - The mode is 0600 owned by the container's running uid — the
      api/README spec says "root:root" but a non-root container can't
      satisfy that; per advisor consensus, mode 0600 + container-uid
      is what matters (no other user on the host can read the file
      because /run/fula-ai/ is also 0700).

Nonce LRU:
    - In-memory dict (capped at 10_000 entries; 5-min TTL).
    - Consumed ONLY after HMAC + expiry pass (per api/README).
      Reverse order would let an attacker who knows the action_id
      format pre-burn legitimate nonces by submitting invalid HMACs.
"""
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path


logger = logging.getLogger("blox-ai.token")


# Wire-format constants
TOKEN_TTL_SEC = 300                 # 5 minutes
NONCE_TTL_SEC = 300                 # match token TTL; expired tokens
                                     # could never re-validate anyway
NONCE_CACHE_MAX = 10_000             # cheap memory cap


# File-system layout — matches docker-compose bind mounts
DEFAULT_SECRET_DIR = "/run/fula-ai"
DEFAULT_SECRET_PATH = "/run/fula-ai/approval-secret"


# Validation outcome codes — match audit_log_line.schema.json's
# rejected_reason enum.
class TokenError(Exception):
    """Raised on any validation failure. The `reason` attribute matches
    the audit log's `rejected_reason` enum so call sites just propagate."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Secret bootstrap
# ---------------------------------------------------------------------------

def ensure_secret(path: str | None = None) -> bytes:
    """Generate (or re-use) the HMAC secret. Returns 32 raw bytes.

    On container start we always WRITE a fresh secret (matches the
    Phase 11 session-state-LOST discipline; cross-restart token replay
    is then impossible). The on-disk file is so anything else in the
    container can read it without being passed the bytes in memory.
    """
    p = Path(path or os.environ.get("BLOX_AI_APPROVAL_SECRET_PATH",
                                    DEFAULT_SECRET_PATH))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Mode 0700 on dir prevents other users on the host from listing
        os.chmod(p.parent, 0o700)
    except OSError as e:
        logger.warning("approval-secret dir setup failed: %s", e)
    secret = secrets.token_bytes(32)
    try:
        # umask+open with O_CREAT|O_TRUNC|O_WRONLY mode 0600
        fd = os.open(str(p), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            os.write(fd, secret)
        finally:
            os.close(fd)
    except OSError as e:
        logger.warning("approval-secret write failed: %s (running in-memory only)", e)
    return secret


# ---------------------------------------------------------------------------
# Signer / verifier
# ---------------------------------------------------------------------------

def _canonical_payload_bytes(action_id: str, expires_at: str, nonce: str) -> bytes:
    """Bytes that we HMAC over. JSON, sorted keys, separator-tight."""
    return json.dumps(
        {"action_id": action_id, "expires_at": expires_at, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hmac_hex(secret: bytes, payload: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


class ApprovalTokenSigner:
    """Issues approval tokens for `recommended_action` events. Owns the
    nonce LRU on the verify side too — one signer per container lifetime."""

    def __init__(self, secret: bytes | None = None):
        self.secret = secret if secret is not None else ensure_secret()
        # OrderedDict for cheap LRU; threadsafe with self._lock
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    # ----- sign -----

    def sign(self, action_id: str) -> str:
        """Mint a token for `action_id`. Returns the base64url wire string."""
        if not action_id or len(action_id) > 128:
            raise ValueError("action_id must be 1..128 chars")
        expires_at = (datetime.now(timezone.utc)
                      + timedelta(seconds=TOKEN_TTL_SEC)
                      ).isoformat(timespec="seconds").replace("+00:00", "Z")
        nonce = secrets.token_urlsafe(16)
        body = _canonical_payload_bytes(action_id, expires_at, nonce)
        hmac_hex = _hmac_hex(self.secret, body)
        wire_dict = {
            "action_id": action_id,
            "expires_at": expires_at,
            "nonce": nonce,
            "hmac": hmac_hex,
        }
        wire_json = json.dumps(wire_dict, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(wire_json).rstrip(b"=").decode("ascii")

    # ----- verify -----

    def verify(self, token: str, expected_action_id: str) -> None:
        """Validate `token` for `expected_action_id`. Raises TokenError on
        any failure; reason field carries the audit log's rejected_reason.

        Verification order (per advisor / api/README):
          1. Decode + parse
          2. expires_at > now
          3. HMAC matches
          4. action_id in token == expected_action_id
          5. ONLY THEN: nonce check + consume
        """
        # 1. Decode + parse
        try:
            padded = token + "=" * (-len(token) % 4)
            raw = base64.urlsafe_b64decode(padded.encode("ascii"))
            wire = json.loads(raw)
        except Exception as e:
            raise TokenError("approval_token_invalid", f"decode: {e}")
        if not isinstance(wire, dict):
            raise TokenError("approval_token_invalid", "wire is not an object")
        for k in ("action_id", "expires_at", "nonce", "hmac"):
            if k not in wire or not isinstance(wire[k], str):
                raise TokenError("approval_token_invalid", f"missing {k}")

        # 2. Expiry
        try:
            expires = datetime.fromisoformat(wire["expires_at"].replace("Z", "+00:00"))
        except ValueError:
            raise TokenError("approval_token_invalid", "expires_at not ISO")
        if expires <= datetime.now(timezone.utc):
            raise TokenError("approval_token_expired")

        # 3. HMAC
        recomputed = _hmac_hex(
            self.secret,
            _canonical_payload_bytes(wire["action_id"], wire["expires_at"], wire["nonce"]),
        )
        if not hmac.compare_digest(recomputed, wire["hmac"]):
            raise TokenError("approval_token_invalid", "hmac mismatch")

        # 4. action_id binding
        if wire["action_id"] != expected_action_id:
            raise TokenError(
                "approval_token_invalid",
                f"action_id mismatch (token claims {wire['action_id']!r}, "
                f"request wants {expected_action_id!r})",
            )

        # 5. Nonce — CONSUME ONLY AFTER everything above passed
        with self._lock:
            self._prune_nonces_locked()
            if wire["nonce"] in self._nonces:
                raise TokenError("approval_token_replayed")
            self._nonces[wire["nonce"]] = time.monotonic() + NONCE_TTL_SEC
            # LRU cap
            while len(self._nonces) > NONCE_CACHE_MAX:
                self._nonces.popitem(last=False)

    # ----- internal -----

    def _prune_nonces_locked(self) -> None:
        now = time.monotonic()
        # OrderedDict insertion order doesn't track expiry directly,
        # so scan: cheap because expired nonces hang around at most
        # NONCE_TTL_SEC.
        expired = [k for k, v in self._nonces.items() if v < now]
        for k in expired:
            del self._nonces[k]

    # Test-only helper
    def _nonce_count(self) -> int:
        return len(self._nonces)
