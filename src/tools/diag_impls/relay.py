"""diag/relay — libp2p relay reachability + circuit reservation status.

Calls the kubo HTTP API on 127.0.0.1:5001. The container needs network
access to the host's kubo (per docker-compose: `network_mode: host` OR
the host's 5001 bound on a reachable address)."""
from __future__ import annotations

from src.tools.diag_impls._helpers import http_post_json, now_iso


# kubo API: POST (no body) per kubo's RPC convention.
KUBO_ID_URL = "http://127.0.0.1:5001/api/v0/id"
KUBO_PEERS_URL = "http://127.0.0.1:5001/api/v0/swarm/peers"


def diag_relay() -> dict:
    """Returns the list of currently-connected relays + circuit reservation
    count. Defensive: kubo unreachable → empty list, count=0."""
    peers_resp = http_post_json(KUBO_PEERS_URL, {}, timeout_s=8.0)
    relays = []
    reservation_count = 0
    if isinstance(peers_resp, dict) and "Peers" in peers_resp:
        for p in peers_resp["Peers"]:
            if not isinstance(p, dict):
                continue
            addr = p.get("Addr", "")
            if not addr:
                continue
            # Heuristic: relays surface via /p2p-circuit suffix in their
            # advertised addr OR the explicit Direction == "Outbound" + relay
            # role. Be conservative — anything circuit-bearing counts.
            has_circuit = "/p2p-circuit" in addr or "Circuit" in str(p.get("Streams") or "")
            if has_circuit:
                reservation_count += 1
            relays.append({
                "addr": addr,
                "swarm_connect_ok": True,
                "has_circuit_reservation": has_circuit,
            })
    return {
        "relays": relays,
        "reservation_count": reservation_count,
        "checked_at": now_iso(),
    }
