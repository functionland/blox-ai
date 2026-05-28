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
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# NOTE: chain.py is intentionally self-contained — it does NOT import
# from src.tools.diag_impls because Phase 0.5b added
# diag_impls/identity_health.py which imports from this module, so
# importing diag_impls helpers at module-load time creates a circular
# dep. http_post_json is duplicated here (~10 LoC); we keep the
# diag_impls/_helpers.py copy as-is for callers that already use it.


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


# Precomputed Ethereum 4-byte function selectors for the view methods we
# call. Each selector is the first 4 bytes of keccak256(signature).
# Computed OFFLINE so we have zero runtime keccak dependency. Reproduction:
#
#   from Crypto.Hash import keccak
#   k = keccak.new(digest_bits=256); k.update(b'<signature>')
#   print('0x' + k.hexdigest()[:8])
#
# Sanity check: 'transfer(address,uint256)' must yield 0xa9059cbb (the
# canonical ERC20 transfer selector). Verified 2026-05-28.
#
# Signatures verified against:
#   PoolStorage:  E:/GitHub/fula-chain/contracts/core/StoragePool.sol:478,531
#   RewardEngine: E:/GitHub/mainnet-claim-web/abi.js lines 2104, 2451 (the
#                 JS ABI is extracted from contracts/RewardEngine.json)
#
# CRITICAL — param orders differ across methods, do not assume:
#   isPeerIdMemberOfPool(uint32 poolId, bytes32 peerId)
#   isPeerOnlineAtTimestamp(uint32 poolId, uint256 timestamp, bytes32 peerId)
#   getOnlineStatusSince(bytes32 peerId, uint32 poolId, uint256 sinceTime)
#                       ^^^^^^^ peerId is FIRST in this one
FUNCTION_SELECTORS: dict[str, str] = {
    # PoolStorage view methods
    # → returns (bool isMember, address memberAddress) — decode 1st word only
    "isPeerIdMemberOfPool(uint32,bytes32)":             "0xb098a605",
    # → returns (address member, uint256 lockedTokens) — useful for diag detail
    "getPeerIdInfo(uint32,bytes32)":                    "0x16f4c1d9",
    # RewardEngine view methods
    # → returns bool isOnline (single bool, 32-byte padded)
    "isPeerOnlineAtTimestamp(uint32,uint256,bytes32)":  "0x6ce7c477",
    # → returns (uint256 onlineCount, uint256 totalExpected) — better recency
    #   signal than the point-in-time check: did the peer report online at
    #   ANY of the expected periods since `sinceTime`?
    "getOnlineStatusSince(bytes32,uint32,uint256)":     "0x428373d6",
}


def encode_uint32(value: int) -> bytes:
    """ABI-encode a uint32 as 32 bytes (left-padded)."""
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"uint32 out of range: {value}")
    return value.to_bytes(32, byteorder="big")


# uint256 max — used for range checks. Solidity's uint256 is 2^256 - 1.
_UINT256_MAX = (1 << 256) - 1


def encode_uint256(value: int) -> bytes:
    """ABI-encode a uint256 as 32 bytes (left-padded). Used for unix
    timestamps and other large unsigned values."""
    if not isinstance(value, int) or value < 0 or value > _UINT256_MAX:
        raise ValueError(f"uint256 out of range: {value}")
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


def decode_bool_and_address(hex_value: str) -> tuple[bool, str]:
    """Decode `(bool, address)` ABI tuple return — two 32-byte words.
    Used for isPeerIdMemberOfPool which returns (bool isMember, address
    memberAddress).

    Layout (per Solidity ABI):
      [0..32)  bool  — low byte; 0x00..00 == False, anything else == True
      [32..64) address — low 20 bytes; high 12 bytes are zero padding

    Returns the address as a 0x-prefixed 40-hex-char lowercase string
    (matches eth checksum-stripped form). Raises ValueError on
    malformed input.
    """
    s = hex_value[2:] if hex_value.startswith("0x") else hex_value
    if len(s) != 128:
        raise ValueError(
            f"expected (bool,address) tuple = 64 bytes hex; got len={len(s)}"
        )
    bool_word = s[:64]
    addr_word = s[64:128]
    is_true = any(c != "0" for c in bool_word)
    # Address occupies the LOW 20 bytes (last 40 hex chars) of the word.
    # High 12 bytes (first 24 hex chars) are zero padding.
    address = "0x" + addr_word[-40:].lower()
    return is_true, address


def decode_uint256_pair(hex_value: str) -> tuple[int, int]:
    """Decode `(uint256, uint256)` ABI tuple — two 32-byte big-endian ints.
    Used for getOnlineStatusSince which returns (uint256 onlineCount,
    uint256 totalExpected)."""
    s = hex_value[2:] if hex_value.startswith("0x") else hex_value
    if len(s) != 128:
        raise ValueError(
            f"expected (uint256,uint256) tuple = 64 bytes hex; got len={len(s)}"
        )
    a = int(s[:64], 16)
    b = int(s[64:128], 16)
    return a, b


def clear_cache_for_tests() -> None:
    """Test-only: reset the call cache so cases don't leak across runs."""
    with _call_cache_lock:
        _call_cache.clear()


# ---------------------------------------------------------------------------
# Inlined http helper (avoid circular import with diag_impls)
# ---------------------------------------------------------------------------


def http_post_json(url: str, body: dict, timeout_s: float = 5.0) -> Any:
    """POST JSON body, parse JSON response. Returns None on any error.

    Duplicated from src.tools.diag_impls._helpers because chain.py must
    not import diag_impls (Phase 0.5b's identity_health imports back
    from chain.py, creating a circular dep at module-load time).
    Keeping the helper inline costs ~12 LoC and keeps chain.py
    self-contained."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, json.JSONDecodeError, TimeoutError):
        return None
