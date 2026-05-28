"""diag/identity_health — chain-grounded pool membership + online recency.

Read-only check that answers two questions the deterministic tree's
"not earning" branch needs:
  1. Is this blox's ipfs-cluster peerID actually a member of the
     configured pool on chain? (PoolStorage.isPeerIdMemberOfPool)
  2. Has the cluster reported online at least once in the recent window?
     (RewardEngine.getOnlineStatusSince)

Critical nuance the user surfaced: the APP connects to the blox via
the KUBO peerID; the CHAIN uses the IPFS-CLUSTER peerID. The two are
distinct. A user can see "app connected" yet have no rewards because
ipfs-cluster is wedged / not a member / not online — all chain-visible
states the app does not surface.

Tristate per advisor (codex+gemini): every chain-derived field is
bool|null with a parallel `*_reason` string when null. Trees branch
on `unknown` explicitly rather than treating absence as `false`.

Privacy: this tool reads PUBLIC chain state for a PUBLIC peerID. No
secrets touched. The config.yaml read is for poolName/chainName/
authorizer (all public identifiers); the identity field is NOT
read or surfaced. The identity.json read is for `id` (the public
peerID) — `private_key` is NOT touched.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.tools.chain import (
    CONTRACTS,
    DEFAULT_RPC_URLS,
    FUNCTION_SELECTORS,
    decode_bool_and_address,
    decode_uint256_pair,
    encode_bytes32,
    encode_call,
    encode_uint256,
    encode_uint32,
    eth_call,
    peer_id_to_bytes32,
)


logger = logging.getLogger("blox-ai.diag.identity_health")


# Defaults — tunable later if we move to per-call config.
CONFIG_YAML_PATH = "/home/pi/.internal/config.yaml"
CLUSTER_IDENTITY_PATH = "/uniondrive/ipfs-cluster/identity.json"

# "Recent" window for getOnlineStatusSince. The on-chain ledger expects
# online-status submissions roughly hourly; 24h gives 24 expected
# windows so a single missed submission isn't treated as offline.
ONLINE_WINDOW_S = 86400


def diag_identity_health() -> dict:
    out: dict = {
        # Tristate signals trees branch on (required by schema)
        "pool_member":         None,
        "pool_member_reason":  "not_attempted",
        "online_recent":       None,
        "online_recent_reason": "not_attempted",
    }

    # 1. Read config.yaml — get poolName + chainName + authorizer.
    cfg = _read_config_yaml(CONFIG_YAML_PATH)
    if cfg.get("authorizer"):
        out["authorizer_peer_id"] = cfg["authorizer"]

    pool_id = cfg.get("poolName_int")
    chain = cfg.get("chainName")
    if pool_id is None:
        out["pool_member_reason"] = "missing_pool_id"
        out["online_recent_reason"] = "missing_pool_id"
        return out
    out["pool_id"] = pool_id

    if not chain:
        out["pool_member_reason"] = "missing_chain"
        out["online_recent_reason"] = "missing_chain"
        return out
    out["chain"] = chain

    if chain not in CONTRACTS or chain not in DEFAULT_RPC_URLS:
        out["pool_member_reason"] = f"unknown_chain:{chain}"
        out["online_recent_reason"] = f"unknown_chain:{chain}"
        return out

    # 2. Read cluster identity.json — get the on-chain peerID.
    cluster_peer = _read_cluster_peer_id(CLUSTER_IDENTITY_PATH)
    if not cluster_peer:
        out["pool_member_reason"] = "missing_cluster_peer_id"
        out["online_recent_reason"] = "missing_cluster_peer_id"
        return out
    out["cluster_peer_id"] = cluster_peer

    # 3. Encode peerID → bytes32 for ABI.
    try:
        peer_b32 = peer_id_to_bytes32(cluster_peer)
    except ValueError as e:
        out["pool_member_reason"] = f"invalid_peer_id:{e}"[:200]
        out["online_recent_reason"] = f"invalid_peer_id:{e}"[:200]
        return out
    out["cluster_peer_id_bytes32"] = peer_b32

    # 4. PoolStorage.isPeerIdMemberOfPool(poolId, peerId) → (bool, address)
    member_call_data = encode_call(
        FUNCTION_SELECTORS["isPeerIdMemberOfPool(uint32,bytes32)"],
        encode_uint32(pool_id),
        encode_bytes32(peer_b32),
    )
    pool_storage_addr = CONTRACTS[chain]["PoolStorage"]
    member_result = eth_call(chain, pool_storage_addr, member_call_data, timeout_s=3.0)
    if member_result.state == "ok":
        try:
            is_member, member_addr = decode_bool_and_address(member_result.value)
            out["pool_member"] = is_member
            out["pool_member_reason"] = "ok"
            # Surface the contract-side member address — useful for
            # debugging cases where chain says "member" but the local
            # config has a different wallet. Don't store if zero (the
            # not-a-member sentinel).
            if member_addr != "0x" + "0" * 40:
                out["pool_member_address"] = member_addr
        except ValueError as e:
            out["pool_member_reason"] = f"decode_failed:{e}"[:200]
    elif member_result.state == "unknown":
        out["pool_member_reason"] = member_result.reason or "rpc_unreachable"
    else:
        out["pool_member_reason"] = f"chain_error:{member_result.reason}"[:200]

    # 5. RewardEngine.getOnlineStatusSince(peerId, poolId, sinceTime)
    #                                     → (uint256 onlineCount, uint256 totalExpected)
    # NOTE param order — peerId is FIRST in this method, unlike
    # isPeerOnlineAtTimestamp which puts poolId first.
    since = int(time.time()) - ONLINE_WINDOW_S
    online_call_data = encode_call(
        FUNCTION_SELECTORS["getOnlineStatusSince(bytes32,uint32,uint256)"],
        encode_bytes32(peer_b32),
        encode_uint32(pool_id),
        encode_uint256(since),
    )
    reward_engine_addr = CONTRACTS[chain]["RewardEngine"]
    online_result = eth_call(chain, reward_engine_addr, online_call_data, timeout_s=3.0)
    if online_result.state == "ok":
        try:
            online_count, total_expected = decode_uint256_pair(online_result.value)
            out["online_count"] = online_count
            out["online_total_expected"] = total_expected
            out["online_window_s"] = ONLINE_WINDOW_S
            # Recent-online iff at least one submission in the window.
            # Trees can also branch on the ratio for "mostly online" vs
            # "rarely online" — we surface the raw counts and let the
            # tree decide.
            out["online_recent"] = online_count > 0
            out["online_recent_reason"] = "ok"
        except ValueError as e:
            out["online_recent_reason"] = f"decode_failed:{e}"[:200]
    elif online_result.state == "unknown":
        out["online_recent_reason"] = online_result.reason or "rpc_unreachable"
    else:
        out["online_recent_reason"] = f"chain_error:{online_result.reason}"[:200]

    return out


# ---------------------------------------------------------------------------
# config.yaml minimal parser
# ---------------------------------------------------------------------------
#
# fula's config.yaml is a controlled, simple flat YAML. We do NOT take a
# pyyaml dep — the file shape is constrained and the failure mode of a
# wrong parse is "field comes back None and tree branches on unknown",
# not silent acceptance of wrong values. Keys we read:
#   - poolName:    string OR number (sometimes "1", sometimes 1)
#   - chainName:   string
#   - authorizer:  string (peer id)


def _read_config_yaml(path: str) -> dict:
    """Parse the controlled fula config.yaml subset. Returns a dict with
    keys among {poolName_int, chainName, authorizer}, or empty when
    file can't be read."""
    out: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t", "-")):
            # Skip nested list/dict — we only care about top-level scalars.
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "poolName":
            try:
                out["poolName_int"] = int(value)
            except ValueError:
                logger.warning("poolName=%r is not an int; skipping", value)
        elif key == "chainName":
            if value:
                out["chainName"] = value
        elif key == "authorizer":
            if value:
                out["authorizer"] = value
    return out


def _read_cluster_peer_id(path: str) -> str | None:
    """Read /uniondrive/ipfs-cluster/identity.json and return its `id`.
    Returns None when file missing / malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    peer = data.get("id") if isinstance(data, dict) else None
    return peer if isinstance(peer, str) and peer else None
