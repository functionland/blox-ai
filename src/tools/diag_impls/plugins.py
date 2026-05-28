"""diag/plugins — list installed + active plugins with version info.

Reads two sources:
  - /home/pi/.internal/plugins/active-plugins.txt — one plugin name
    per line; written by the plugins manager when a plugin is
    install/start. Empty/missing => no active plugins.
  - /home/pi/.internal/plugins/<name>/status.txt — short status string
    ("Installed", "Running", "Stopped", etc.) per installed plugin.
  - /usr/bin/fula/plugins/<name>/info.json — version + display_name +
    description metadata, sourced from the OTA bundle (NOT the
    runtime install dir, which may not have it).

Trees branch on:
  - blox-ai not in active list → user hasn't installed the plugin
  - status != "Installed" / "Running" → install failed mid-way
  - active vs installed mismatch → state diverged
"""
from __future__ import annotations

import json

from src.tools.diag_impls._helpers import read_state, run_subprocess


ACTIVE_PLUGINS_PATH = "/home/pi/.internal/plugins/active-plugins.txt"
RUNTIME_PLUGINS_DIR = "/home/pi/.internal/plugins"
SOURCE_PLUGINS_DIR = "/usr/bin/fula/plugins"
_TIMEOUT_S = 2.0


def diag_plugins() -> dict:
    out: dict = {
        "active": _read_active_plugins(),
        "installed": [],
    }

    # ls the runtime dir to find every currently-installed plugin
    # (which may differ from `active` if the plugin manager hasn't
    # picked up a fresh install yet).
    rc, ls_out, _ = run_subprocess(
        ["ls", "-1", RUNTIME_PLUGINS_DIR], timeout_s=_TIMEOUT_S,
    )
    if rc != 0 or not ls_out:
        return out

    for entry in ls_out.splitlines():
        name = entry.strip()
        # Skip the manifest text files and dotfiles.
        if not name or name.endswith(".txt") or name.startswith("."):
            continue
        # Confirm it's a directory (a real plugin install).
        info = _read_plugin_info(name)
        info["name"] = name
        info["status"] = _read_plugin_status(name)
        info["active"] = name in out["active"]
        out["installed"].append(info)

    return out


def _read_active_plugins() -> list[str]:
    try:
        with open(ACTIVE_PLUGINS_PATH, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _read_plugin_status(name: str) -> str:
    path = f"{RUNTIME_PLUGINS_DIR}/{name}/status.txt"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()[:64] or "unknown"
    except OSError:
        return "unknown"


def _read_plugin_info(name: str) -> dict:
    """Read info.json from the SOURCE plugins dir (where the OTA-shipped
    metadata lives), since runtime dirs may not include it. Returns
    {} when not found — tree handles."""
    path = f"{SOURCE_PLUGINS_DIR}/{name}/info.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    # Surface only the fields trees care about.
    for key in ("display_name", "version", "description"):
        v = data.get(key)
        if isinstance(v, str) and v:
            out[key] = v[:500]
    return out
