"""diag/relay — libp2p relay reservation status.

Calls the kubo HTTP API via BLOX_AI_KUBO_API_URL (docker-compose sets it to
`http://ipfs_host:5001/api/v0` — the kubo container on the shared
fula_default network). The blox-ai container is NOT network_mode: host, so
127.0.0.1:5001 is its own empty loopback, not the host's kubo.

What "reservation_count" means
------------------------------
A blox behind NAT becomes reachable from outside its LAN by holding a
circuit-relay-v2 *reservation* with a relay. kubo (AutoRelay) only
ANNOUNCES a `/p2p-circuit` address for a relay once it actually holds a
reservation with that relay — so the node's own announced address set
(`/api/v0/id` → Addresses) is the ground truth for "do I have a relay
reservation?".

Bug fix 2026-05-29: this probe previously scanned `/api/v0/swarm/peers`
for `/p2p-circuit` and counted those PEERS. That measures inbound peers
reached *via* a circuit, NOT the node's own outbound reservations — two
different things. A healthy blox holding a valid reservation typically
has ZERO circuit-bearing swarm peers, so the old logic returned
reservation_count=0 and the disconnected tree fired a false
"your Blox has no relay reservations" verdict. Lab device
(12D3KooWCnRuQF…) was announcing 3 circuit addresses via
relay.fula.network in its own `ipfs id` yet the probe reported 0.

A single reservation is announced once per relay transport (tcp /
quic-v1 / webtransport), so we dedupe by the relay's peer id:
reservation_count = number of distinct relays we're reserved through.
"""
from __future__ import annotations

import os

from src.tools.diag_impls._helpers import http_post_json, now_iso


# kubo API base. Mirror kubo_health.py: read BLOX_AI_KUBO_API_URL, which
# docker-compose sets to `http://ipfs_host:5001/api/v0` (the kubo container on
# the shared fula_default network). Default to that same in-container hostname
# so a missing env var still points somewhere real; host-side smoke scripts
# override to 127.0.0.1.
#
# Bug fix 2026-05-29: this previously hardcoded `http://127.0.0.1:5001`, which
# from inside the (non-host-networked) blox-ai container is Connection refused
# on every call -> http_post_json returns None -> reservation_count was
# structurally 0 on a perfectly healthy blox, firing a false "no relay
# reservation" verdict every run. The lab device held 3 circuit reservations
# via relay.fula.network that this probe could never see. POST, no body, per
# kubo's RPC convention.
KUBO_API_BASE = os.environ.get(
    "BLOX_AI_KUBO_API_URL",
    "http://ipfs_host:5001/api/v0",
)
KUBO_ID_URL = KUBO_API_BASE.rstrip("/") + "/id"

_CIRCUIT = "/p2p-circuit"


def _relay_peer_id(relay_prefix: str) -> str:
    """Relay peer id = the last `/p2p/<id>` in the address segment that
    precedes `/p2p-circuit`. Empty string if the address has no explicit
    relay peer id (then the caller dedupes on the raw prefix instead)."""
    parts = relay_prefix.split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "p2p":
            return parts[i + 1]
    return ""


def _relay_dns_name(relay_prefix: str) -> str:
    """Human-facing relay host: the /dns* name if present, else the
    /ip4 or /ip6 literal. Empty string if neither is present."""
    parts = relay_prefix.split("/")
    for i, part in enumerate(parts):
        if part in ("dns", "dnsaddr", "dns4", "dns6") and i + 1 < len(parts):
            return parts[i + 1]
    for i, part in enumerate(parts):
        if part in ("ip4", "ip6") and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def diag_relay() -> dict:
    """Returns the node's own relay reservations + a count. Defensive:
    kubo unreachable → empty list, count=0 (the disconnected tree checks
    kubo_health BEFORE relay_check, so a 0 here on a wedged kubo is never
    the verdict that reaches the user)."""
    id_resp = http_post_json(KUBO_ID_URL, {}, timeout_s=8.0)
    relays: list[dict] = []
    seen_relays: set[str] = set()

    if isinstance(id_resp, dict):
        for addr in id_resp.get("Addresses") or []:
            if not isinstance(addr, str) or _CIRCUIT not in addr:
                continue
            # A /p2p-circuit address in our OWN announced set == a held
            # reservation. Everything before /p2p-circuit identifies the relay.
            relay_prefix = addr.split(_CIRCUIT, 1)[0]
            relays.append({
                "addr": addr,
                "dns_name": _relay_dns_name(relay_prefix),
                "has_circuit_reservation": True,
            })
            seen_relays.add(_relay_peer_id(relay_prefix) or relay_prefix)

    return {
        "relays": relays,
        "reservation_count": len(seen_relays),
        "checked_at": now_iso(),
    }
