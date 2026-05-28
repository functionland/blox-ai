"""On-chain helpers for diag/identity_health.

Read-only eth_call against Fula's PoolStorage + RewardEngine contracts
on base + skale. No signing, no gas, no keys — pure view-function reads.

Why no web3.py: that lib's transitive deps are ~80MB. We only need
eth_call (no transactions, no wallet) and `eth_abi` is the only piece
that handles padding/encoding cleanly. JSON-RPC over the existing
stdlib `urllib`-backed `http_post_json` helper.

bytes32(peerId) conversion ported faithfully from
`mainnet-claim-web/app.js:peerIdToBytes32`. TWO paths:
  - CIDv1 (Ed25519): leading bytes [0x00, 0x24, 0x08, 0x01, 0x12],
    total length >= 37, take the LAST 32 bytes (the raw pubkey).
  - Legacy multihash: leading bytes [0x12, 0x20], total length == 34,
    take bytes [2:] (the sha256 digest).
Gemini's "always take last 32" guess gets the legacy path wrong; advisor
caught it; algorithm is verified against the JS reference.

Tristate contract for chain-derived facts (per codex + gemini):
  - True / False — definitive answer from the chain
  - 'unknown' (string) with `unknown_reason` — RPC unreachable, chain
    revert, malformed peerId, etc. Trees branch explicitly on unknown.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from src.tools.diag_impls._helpers import http_post_json


logger = logging.getLogger("blox-ai.chain")


# Default public RPC endpoints per chain. Trees can branch on the
# `rpc_reachable` fact when these are blocked; in 0.6 we may add a
# config.yaml override field.
DEFAULT_RPC_URLS: dict[str, str] = {
    "base": "https://mainnet.base.org",
    "skale": "https://mainnet.skalenodes.com/v1/elated-tan-skat",
}

# Fula contract addresses, per user-provided spec 2026-05-28.
CONTRACTS: dict[str, dict[str, str]] = {
    "base": {
        "PoolStorage":  "0xb093fF4B3B3B87a712107B26566e0cCE5E752b4D",
        "RewardEngine": "0x31029f90405fd3D9cB0835c6d21b9DFF058Df45A",
    },
    "skale": {
        "PoolStorage":  "0xf9176Ffde541bF0aa7884298Ce538c471Ad0F015",
        "RewardEngine": "0xF7c64248294C45Eb3AcdD282b58675F1831fb047",
    },
}


# ---------------------------------------------------------------------------
# bytes32(peerId)
# ---------------------------------------------------------------------------


# Multibase btc-alphabet (used by libp2p peerId "z..." encoding). Stdlib
# `base64.b58decode` doesn't exist; we hand-roll because adding a base58
# pip dep for one function is excessive.
_BASE58_BTC_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58_decode(s: str) -> bytes:
    """Bitcoin-alphabet base58 decode. Matches the JS multibase 'z' prefix
    decoder used in mainnet-claim-web."""
    n = 0
    for c in s:
        try:
            n = n * 58 + _BASE58_BTC_ALPHABET.index(c)
        except ValueError:
            raise ValueError(f"invalid base58 character: {c!r}")
    # Convert int to bytes, then prepend a leading-zero byte per leading
    # '1' in the input (base58 maps leading zeros to '1').
    out = bytearray()
    while n > 0:
        out.append(n & 0xFF)
        n >>= 8
    leading_zeros = 0
    for c in s:
        if c == "1":
            leading_zeros += 1
        else:
            break
    out.extend(b"\x00" * leading_zeros)
    return bytes(reversed(out))


def peer_id_to_bytes32(peer_id: str) -> str:
    """Port of `mainnet-claim-web/app.js:peerIdToBytes32`.

    Accepts the libp2p peerId string (with or without the leading 'z'
    multibase prefix). Returns a 0x-prefixed 64-hex-char string suitable
    for passing as a `bytes32` parameter to PoolStorage / RewardEngine.

    Raises ValueError when the decoded length doesn't match either
    expected format — callers should treat this as `unknown_reason
    = 'invalid_peerid_format'`.
    """
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer_id must be a non-empty string")

    # Multibase 'z' prefix (base58btc) — JS code prepends if missing.
    stripped = peer_id[1:] if peer_id.startswith("z") else peer_id
    decoded = _b58_decode(stripped)

    # CIDv1 (Ed25519) — header [0x00, 0x24, 0x08, 0x01, 0x12], total >= 37
    cidv1_header = (0x00, 0x24, 0x08, 0x01, 0x12)
    if (
        len(decoded) >= 37
        and tuple(decoded[:5]) == cidv1_header
    ):
        pubkey = decoded[-32:]
        return "0x" + pubkey.hex()

    # Legacy multihash — header [0x12, 0x20], total == 34
    if len(decoded) == 34 and decoded[0] == 0x12 and decoded[1] == 0x20:
        digest = decoded[2:]
        return "0x" + digest.hex()

    raise ValueError(
        f"unsupported peerId format (decoded length {len(decoded)})"
    )


# ---------------------------------------------------------------------------
# JSON-RPC eth_call
# ---------------------------------------------------------------------------


def _keccak256(data: bytes) -> bytes:
    """keccak256 — needed for Ethereum function selectors. hashlib has
    sha3_256 (NIST SHA-3) which is DIFFERENT from keccak256 (the
    pre-standardization variant Ethereum uses). On Python 3.6+ we get
    keccak via `Cryptodome.Hash.keccak` OR `eth_utils`. Both are heavy.
    pysha3 was the standard but is unmaintained.

    Workaround: most function selectors are SHORT + KNOWN. Precompute
    them at module load so we never need keccak at runtime. This dict
    is the entire selector surface for our two contracts' view methods
    we use.
    """
    raise NotImplementedError(
        "keccak256 not implemented; use precomputed FUNCTION_SELECTORS"
    )


# Precomputed Ethereum 4-byte function selectors for the view methods
# we call. Each selector is the first 4 bytes of keccak256(signature).
# Generated offline via `cast sig "<signature>"` (foundry) and pinned
# here so we have ZERO runtime keccak dependency. When adding a new
# method: compute via `cast sig` + paste below.
#
# Signatures match the canonical ABI types — peerId is bytes32, poolId
# is uint32 (per Fula contract source).
FUNCTION_SELECTORS: dict[str, str] = {
    # PoolStorage view methods
    "isMemberOfPool(uint32,bytes32)": "0x00000000",  # PLACEHOLDER
    "members(uint32,bytes32)":        "0x00000000",  # PLACEHOLDER
    # RewardEngine view methods
    "isOnline(uint32,bytes32)":       "0x00000000",  # PLACEHOLDER
    "isPeerOnline(uint32,bytes32)":   "0x00000000",  # PLACEHOLDER
}
# IMPORTANT: the selectors above are PLACEHOLDERS. Phase 0.5b must:
#   1. Read RewardEngine.json + PoolStorage ABI from mainnet-claim-web
#      to confirm the EXACT function names + signatures we should call
#   2. Compute selectors offline via `cast sig`
#   3. Paste real values here BEFORE the diag tool ships
# The function above is structured so finishing this step is a one-line
# change per selector with no algorithm risk.


def encode_uint32(value: int) -> bytes:
    """ABI-encode a uint32 as 32 bytes (left-padded)."""
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"uint32 out of range: {value}")
    return value.to_bytes(32, byteorder="big")


def encode_bytes32(hex_value: str) -> bytes:
    """Decode a 0x-prefixed 32-byte hex string to raw bytes for ABI."""
    s = hex_value[2:] if hex_value.startswith("0x") else hex_value
    if len(s) != 64:
        raise ValueError(f"bytes32 must be 32 bytes hex; got len={len(s)}")
    return bytes.fromhex(s)


def encode_call(selector: str, *args: bytes) -> str:
    """Build a `data` payload for eth_call: selector || ABI-encoded args.
    Returns 0x-prefixed hex string."""
    sel = bytes.fromhex(selector[2:] if selector.startswith("0x") else selector)
    if len(sel) != 4:
        raise ValueError(f"selector must be 4 bytes; got {len(sel)}")
    return "0x" + (sel + b"".join(args)).hex()


# ---------------------------------------------------------------------------
# RPC client + cache
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    """Tristate result of an eth_call. `value` is None when `state` is
    'unknown' or 'error'; tree evaluator should branch on state, NOT
    on the value being None."""
    state: str   # 'ok' | 'unknown' | 'error'
    value: Any = None
    reason: str | None = None


# Per-call cache: key = (chain, contract_address, data) → (CallResult, expires_at).
# 60s TTL per gemini recommendation; troubleshoot sessions are short and
# membership/online status don't change second-to-second.
_CACHE_TTL_S = 60.0
_call_cache: dict[tuple, tuple[CallResult, float]] = {}
_call_cache_lock = threading.Lock()


def eth_call(
    chain: str,
    to_address: str,
    data: str,
    *,
    timeout_s: float = 2.0,
) -> CallResult:
    """JSON-RPC eth_call against `chain`'s default RPC endpoint.

    Returns a CallResult. `state == 'ok'` carries `value` as the
    0x-prefixed hex response. `'unknown'` for RPC unreachable / timeout.
    `'error'` for chain-side revert OR malformed RPC response.

    60s in-memory cache keyed on (chain, to, data).
    """
    if chain not in DEFAULT_RPC_URLS:
        return CallResult(state="error", reason=f"unknown_chain:{chain}")

    cache_key = (chain, to_address.lower(), data.lower())
    now = time.monotonic()
    with _call_cache_lock:
        cached = _call_cache.get(cache_key)
        if cached and cached[1] > now:
            return cached[0]

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [
            {"to": to_address, "data": data},
            "latest",
        ],
        "id": 1,
    }
    rpc_url = DEFAULT_RPC_URLS[chain]
    resp = http_post_json(rpc_url, payload, timeout_s=timeout_s)
    if resp is None:
        result = CallResult(state="unknown", reason="rpc_unreachable")
    elif "error" in resp:
        msg = resp["error"].get("message", "unknown_chain_error")
        result = CallResult(state="error", reason=str(msg)[:200])
    elif "result" in resp and isinstance(resp["result"], str):
        result = CallResult(state="ok", value=resp["result"])
    else:
        result = CallResult(state="error", reason="malformed_rpc_response")

    with _call_cache_lock:
        _call_cache[cache_key] = (result, now + _CACHE_TTL_S)
    return result


def decode_bool(hex_value: str) -> bool:
    """Decode a 32-byte hex eth_call result as a bool (0x0...0 => false,
    anything else => true)."""
    s = hex_value[2:] if hex_value.startswith("0x") else hex_value
    return any(c != "0" for c in s)


def clear_cache_for_tests() -> None:
    """Test-only: reset the call cache so cases don't leak across runs."""
    with _call_cache_lock:
        _call_cache.clear()
