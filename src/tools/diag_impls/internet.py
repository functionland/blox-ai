"""diag/internet — DNS + HTTPS reachability from the device."""
from __future__ import annotations

import time

from src.tools.diag_impls._helpers import dns_lookup, https_head, now_iso


# Targets: google.com is the "the internet itself works" canary; the
# discovery host is the Fula-specific reachability check (per Phase 1.2
# in fula-ota). We deliberately do BOTH because corp firewalls / regional
# blocks frequently let google through but kill discovery.
GOOGLE_HOST = "www.google.com"
DISCOVERY_HOST = "discovery.fula.network"


def diag_internet() -> dict:
    dns_ok = dns_lookup(GOOGLE_HOST) and dns_lookup(DISCOVERY_HOST)
    g_ok, _, g_lat = https_head(f"https://{GOOGLE_HOST}", timeout_s=5.0)
    d_ok, _, d_lat = https_head(f"https://{DISCOVERY_HOST}/relays", timeout_s=5.0)
    avg_lat = (g_lat + d_lat) / 2 if (g_lat + d_lat) > 0 else 0.0
    # Captive-portal heuristic: DNS works AND google https returns OK
    # AND latency is suspiciously low (a portal usually intercepts at the
    # router with sub-50ms RTT) AND discovery is blocked. False positives
    # acceptable — the AI cites this as one signal among many.
    captive = dns_ok and g_ok and not d_ok and avg_lat < 50
    return {
        "dns_ok": dns_ok,
        "https_google_ok": g_ok,
        "https_discovery_ok": d_ok,
        "latency_ms_avg": round(avg_lat, 1),
        "captive_portal_likely": captive,
        "checked_at": now_iso(),
    }
