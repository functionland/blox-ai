"""diag/network_interface — IP + WiFi state per interface.

Parses `ip -j addr` (link + ipv4 + ipv6 — JSON, no scraping) and `iw
dev <iface> link` (WiFi association detail) for each interface. Used by
the deterministic tree to branch on:
  - phone reports "Disconnected" but the blox actually has no WiFi
    association (router restart, password rotation, captive portal)
  - blox connected to a different SSID than the phone (multi-AP house)
  - signal too weak for reliable libp2p (< -75 dBm threshold)
  - blox got NO ipv4 (DHCP server outage; static config broken)

We do NOT call `nmcli` because:
  - it's not present on every fula-ota build (depends on which Pi OS
    image was flashed; some are Raspbian Lite which uses dhcpcd
    directly)
  - `ip -j addr` is part of iproute2 which IS present everywhere
  - `iw` is the canonical WiFi tool and was part of the original probe
    helper user wrote

Graceful fallback: missing tools → `tools_present` lists what's present;
trees can branch on `wifi_supported: false` instead of treating a
missing `wifi_ssid` field as "user disconnected from WiFi" (false
positive).
"""
from __future__ import annotations

import json

from src.tools.diag_impls._helpers import run_subprocess


# Per-call subprocess timeout. ip + iw both return near-instant; >2s
# means the netlink path is hung, which we surface as missing data.
_TIMEOUT_S = 2.0


def diag_network_interface() -> dict:
    out: dict = {"interfaces": [], "tools_present": {}}
    out["tools_present"]["ip"] = _have("ip")
    out["tools_present"]["iw"] = _have("iw")

    if not out["tools_present"]["ip"]:
        # No iproute2 → return early with empty interfaces. Trees should
        # branch on `tools_present.ip == false` (broken build) instead
        # of misreading the empty list as "no interfaces detected".
        return out

    rc, ip_json, _ = run_subprocess(
        ["ip", "-j", "addr"],
        timeout_s=_TIMEOUT_S,
    )
    if rc != 0 or not ip_json:
        return out
    try:
        raw_links = json.loads(ip_json)
    except json.JSONDecodeError:
        return out

    for link in raw_links:
        ifname = link.get("ifname")
        if not isinstance(ifname, str) or ifname == "lo":
            continue
        iface = _summarize_link(link)
        if out["tools_present"]["iw"] and _looks_like_wifi(ifname):
            iface.update(_iw_link(ifname))
        out["interfaces"].append(iface)
    return out


def _have(cmd: str) -> bool:
    """True iff the tool is on PATH. `which` returns 0 on found, 1 on
    not-found. Treats subprocess errors (FileNotFoundError of `which`
    itself on Windows) as 'tool not present' — accurate for the dev
    host where neither `which` nor `iw` exist."""
    rc, _, _ = run_subprocess(["which", cmd], timeout_s=1.5)
    return rc == 0


def _summarize_link(link: dict) -> dict:
    """Pluck the fields trees actually branch on, drop the rest."""
    out: dict = {
        "name": link.get("ifname", ""),
        # `operstate` is the kernel's view: UP / DOWN / UNKNOWN / DORMANT.
        # Trees should branch on this for "link layer is up", NOT on
        # whether the interface has an IP — DHCP failure leaves UP+no-ip.
        "operstate": link.get("operstate", "UNKNOWN"),
        "mtu":  link.get("mtu") if isinstance(link.get("mtu"), int) else None,
        "mac":  link.get("address", "") if isinstance(link.get("address"), str) else "",
        "ipv4": [],
        "ipv6": [],
    }
    for info in link.get("addr_info") or []:
        family = info.get("family")
        local = info.get("local")
        if not isinstance(local, str):
            continue
        if family == "inet":
            out["ipv4"].append(local)
        elif family == "inet6":
            # Skip link-local (fe80::*) — noise unless trees explicitly
            # need them, which today they don't.
            if not local.lower().startswith("fe80"):
                out["ipv6"].append(local)
    return out


def _looks_like_wifi(ifname: str) -> bool:
    """Heuristic: only call `iw` for likely-WiFi names. Avoids spurious
    "Device or resource busy" errors when calling iw on bridge or
    docker interfaces."""
    return ifname.startswith(("wlan", "wlp", "wlx"))


def _iw_link(ifname: str) -> dict:
    """Parse `iw dev <iface> link`. Returns {} on any error or when
    not associated (output starts with 'Not connected')."""
    rc, link_out, _ = run_subprocess(
        ["iw", "dev", ifname, "link"],
        timeout_s=_TIMEOUT_S,
    )
    if rc != 0 or not link_out:
        return {}
    text = link_out.strip()
    if text.lower().startswith("not connected"):
        return {"wifi_associated": False}

    out: dict = {"wifi_associated": True}
    for raw in text.splitlines():
        line = raw.strip()
        # SSID
        if line.startswith("SSID:"):
            out["wifi_ssid"] = line.split(":", 1)[1].strip()
        # Signal: '-55 dBm'
        elif line.startswith("signal:"):
            tail = line.split(":", 1)[1].strip()
            for tok in tail.split():
                try:
                    out["wifi_signal_dbm"] = int(tok)
                    break
                except ValueError:
                    continue
        # tx bitrate: '130.0 MBit/s ...'
        elif line.startswith("tx bitrate:"):
            tail = line.split(":", 1)[1].strip()
            for tok in tail.split():
                try:
                    out["wifi_tx_bitrate_mbps"] = float(tok)
                    break
                except ValueError:
                    continue
        # freq: '5180'
        elif line.startswith("freq:"):
            tail = line.split(":", 1)[1].strip()
            try:
                out["wifi_freq_mhz"] = int(tail.split()[0])
            except (ValueError, IndexError):
                pass
    return out
