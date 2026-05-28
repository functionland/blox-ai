"""diag/kubo_health — kubo daemon ID + version + swarm peer count.

Direct HTTP to the kubo API at 127.0.0.1:5001. Trees branch on:
  - daemon_reachable=false → kubo wedged or container crashed (different
    from the systemd_services check, which only sees the container
    state — kubo can be "running" but its API can be hung)
  - swarm_peer_count < 5 → libp2p connectivity poor; explains many
    "device disconnected" complaints
  - version mismatch with the bundled-target → OTA needed

kubo HTTP API uses POST for everything per the docs. No GET fallback.
"""
from __future__ import annotations

from src.tools.diag_impls._helpers import http_post_json


KUBO_API = "http://127.0.0.1:5001/api/v0"
_TIMEOUT_S = 3.0


def diag_kubo_health() -> dict:
    out: dict = {"daemon_reachable": False}

    id_resp = http_post_json(f"{KUBO_API}/id", body={}, timeout_s=_TIMEOUT_S)
    if not isinstance(id_resp, dict) or "ID" not in id_resp:
        return out
    out["daemon_reachable"] = True
    out["peer_id"] = id_resp.get("ID", "")
    out["agent_version"] = id_resp.get("AgentVersion", "")[:128]
    # Addresses can be a long list of multiaddrs (relay + LAN + WAN).
    # Don't dump them all — trees only need the COUNT for "have at
    # least one reachable address" branching.
    addrs = id_resp.get("Addresses") or []
    if isinstance(addrs, list):
        out["addresses_count"] = len(addrs)

    ver_resp = http_post_json(f"{KUBO_API}/version", body={}, timeout_s=_TIMEOUT_S)
    if isinstance(ver_resp, dict):
        out["version"] = ver_resp.get("Version", "")[:32]
        out["commit"] = ver_resp.get("Commit", "")[:32]

    swarm_resp = http_post_json(f"{KUBO_API}/swarm/peers", body={}, timeout_s=_TIMEOUT_S)
    if isinstance(swarm_resp, dict):
        peers = swarm_resp.get("Peers") or []
        if isinstance(peers, list):
            out["swarm_peer_count"] = len(peers)

    return out
