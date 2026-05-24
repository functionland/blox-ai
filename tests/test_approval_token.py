"""C4 — ApprovalTokenSigner tests.

Covers:
  - sign + verify happy path
  - reject expired
  - reject replayed (nonce LRU)
  - reject HMAC mismatch (wrong secret)
  - reject action_id mismatch in token vs request
  - reject malformed wire format (bad base64, bad json, missing fields)
  - NONCE CONSUMPTION ORDER: HMAC + expiry checked BEFORE nonce burn
    (replay-protection sanity: a token with bad HMAC must NOT
    consume the nonce, so a later valid token with the same nonce
    would still verify)
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.tools.approval_token import (
    ApprovalTokenSigner,
    TOKEN_TTL_SEC,
    TokenError,
)


def _patch_secret_path(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH",
                       str(tmp_path / "approval-secret"))


def test_sign_verify_happy_path(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    token = s.sign("act-1")
    # No exception
    s.verify(token, expected_action_id="act-1")


def test_sign_returns_base64url_no_padding(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    token = s.sign("act-1")
    # Standard base64 padding is '='; urlsafe + rstrip removes it.
    assert "=" not in token
    # Reasonable length (json with action_id + ISO ts + 22-char nonce + 64-char hex hmac ≈ 180 chars; base64'd ≈ 240)
    assert 100 < len(token) < 600


def test_verify_rejects_replayed_token(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    token = s.sign("act-1")
    s.verify(token, expected_action_id="act-1")
    with pytest.raises(TokenError) as exc:
        s.verify(token, expected_action_id="act-1")
    assert exc.value.reason == "approval_token_replayed"


def test_verify_rejects_expired_token(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    # Manually craft an expired token using the same secret
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=60)) \
        .isoformat(timespec="seconds").replace("+00:00", "Z")
    from src.tools.approval_token import _canonical_payload_bytes, _hmac_hex
    body = _canonical_payload_bytes("act-1", expired_at, "nonce-x")
    h = _hmac_hex(s.secret, body)
    wire = base64.urlsafe_b64encode(json.dumps({
        "action_id": "act-1", "expires_at": expired_at,
        "nonce": "nonce-x", "hmac": h,
    }, separators=(",", ":")).encode()).rstrip(b"=").decode()
    with pytest.raises(TokenError) as exc:
        s.verify(wire, expected_action_id="act-1")
    assert exc.value.reason == "approval_token_expired"


def test_verify_rejects_hmac_mismatch_wrong_secret(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s1 = ApprovalTokenSigner()
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH",
                       str(tmp_path / "other-secret"))
    s2 = ApprovalTokenSigner()
    token = s1.sign("act-1")
    with pytest.raises(TokenError) as exc:
        s2.verify(token, expected_action_id="act-1")
    assert exc.value.reason == "approval_token_invalid"


def test_verify_rejects_action_id_mismatch(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    token = s.sign("act-1")
    with pytest.raises(TokenError) as exc:
        s.verify(token, expected_action_id="act-DIFFERENT")
    assert exc.value.reason == "approval_token_invalid"


def test_verify_rejects_bad_base64(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    with pytest.raises(TokenError) as exc:
        s.verify("!!!not-valid-base64!!!", expected_action_id="x")
    assert exc.value.reason == "approval_token_invalid"


def test_verify_rejects_bad_json(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    bad = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
    with pytest.raises(TokenError) as exc:
        s.verify(bad, expected_action_id="x")
    assert exc.value.reason == "approval_token_invalid"


def test_verify_rejects_missing_fields(monkeypatch, tmp_path):
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    bad = base64.urlsafe_b64encode(json.dumps({"action_id": "x"}).encode()).rstrip(b"=").decode()
    with pytest.raises(TokenError) as exc:
        s.verify(bad, expected_action_id="x")
    assert exc.value.reason == "approval_token_invalid"


def test_nonce_only_consumed_after_full_validation(monkeypatch, tmp_path):
    """SECURITY-CRITICAL: a token with the wrong HMAC but a real nonce
    MUST NOT burn that nonce. Otherwise an attacker who knows the nonce
    format could pre-burn legitimate nonces."""
    _patch_secret_path(monkeypatch, tmp_path)
    s = ApprovalTokenSigner()
    # Mint a real token (will consume nonce N on verify)
    real_token = s.sign("act-1")
    # Craft a TAMPERED version with a different hmac but same nonce
    # to attempt to burn the nonce ahead of time
    padded = real_token + "=" * (-len(real_token) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    nonce = decoded["nonce"]
    # Tamper the hmac
    tampered = dict(decoded)
    tampered["hmac"] = "0" * 64
    bad_token = base64.urlsafe_b64encode(
        json.dumps(tampered, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    # The tampered submission should fail on HMAC mismatch (not consume nonce)
    with pytest.raises(TokenError) as exc:
        s.verify(bad_token, expected_action_id="act-1")
    assert exc.value.reason == "approval_token_invalid"
    # And the legitimate token with the SAME nonce should still verify
    s.verify(real_token, expected_action_id="act-1")
    # Now THAT one consumed the nonce; replay would fail
    with pytest.raises(TokenError):
        s.verify(real_token, expected_action_id="act-1")


def test_ttl_constant_matches_plan_5min():
    assert TOKEN_TTL_SEC == 300


def test_signer_isolated_secret_per_instance(monkeypatch, tmp_path):
    """Two signer instances with different secrets must not cross-verify."""
    _patch_secret_path(monkeypatch, tmp_path)
    a = ApprovalTokenSigner()
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH",
                       str(tmp_path / "other"))
    b = ApprovalTokenSigner()
    t = a.sign("x")
    with pytest.raises(TokenError):
        b.verify(t, expected_action_id="x")
