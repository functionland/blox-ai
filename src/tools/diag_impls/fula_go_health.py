"""diag/fula_go_health — fula_go container state + recent pool-related events.

fula_go is the wap / authorizer / chain-bridge container. It doesn't
expose an HTTP status endpoint, so the only signals are:
  - docker inspect for container state (running, restarts, started_at)
  - recent docker logs filtered for pool / authorizer / mDNS-config
    events (the actually-useful signal is buried in mDNS spam, ~95% of
    the log volume; we read 500 lines and filter)

Trees branch on:
  - container not running → catastrophic; restart needed
  - restart_count > 3 → fula_go in a crash loop
  - last_mdns_info_loaded_ts > N seconds old → wap is wedged (the mDNS
    server normally re-broadcasts every 5s per the 08fe98f LoadConfig
    fix; absence > 30s means something's wrong)
  - last_pool_event present + recent → registered with pool (cross-check
    against identity_health.pool_member from chain)
"""
from __future__ import annotations

import json
import re


CONTAINER = "fula_go"
_LOG_TAIL_LINES = 500


# docker SDK is the deployed container's path to docker.sock. The docker
# CLI is NOT installed in the blox-ai container image (would add ~50MB).
# Importing lazily so tests can patch `docker.from_env` per-test.
def _docker_client():
    try:
        import docker
        return docker.from_env(timeout=5)
    except Exception:
        return None

# Keywords that mark a pool-relevant event in fula_go logs. We're
# conservative — better to surface a few false-positives (e.g., a
# debug line containing "pool") than miss the actual registration.
_POOL_EVENT_KEYWORDS = re.compile(
    r"\b(authorizer|joined|registered|pool|chain|membership)\b",
    re.IGNORECASE,
)
# mDNS loaded-from-config line — proves wap is alive and re-reading
# config every 5s per go-fula 08fe98f.
_MDNS_LOADED_RE = re.compile(r"mdns info loaded from config file", re.IGNORECASE)


def diag_fula_go_health() -> dict:
    out: dict = {"container_running": False}

    client = _docker_client()
    if client is None:
        return out

    # 1. inspect — container state + uptime + restarts.
    try:
        container = client.containers.get(CONTAINER)
    except Exception:
        # Container doesn't exist, docker.sock unreachable, etc.
        return out

    attrs = container.attrs or {}
    state_dict = attrs.get("State") or {}
    state = state_dict.get("Status") or ""
    out["container_state"] = state
    out["container_running"] = state == "running"
    restart_count = attrs.get("RestartCount")
    if isinstance(restart_count, int):
        out["restart_count"] = restart_count
    started = state_dict.get("StartedAt")
    if isinstance(started, str) and started and started != "0001-01-01T00:00:00Z":
        out["container_started_at"] = started

    if not out["container_running"]:
        return out

    # 2. log tail — find last mDNS loaded line + last pool event.
    # The SDK's logs() includes BOTH stdout and stderr by default
    # (controllable via stdout=/stderr= kwargs); fula_go's zap logger
    # writes to stderr so we explicitly request both.
    try:
        log_bytes = container.logs(
            tail=_LOG_TAIL_LINES, stdout=True, stderr=True,
        )
        log_out = log_bytes.decode("utf-8", errors="replace") if log_bytes else ""
    except Exception:
        log_out = ""
    if not log_out:
        return out

    mdns_count = 0
    last_mdns_ts: str | None = None
    last_pool_ts: str | None = None
    last_pool_excerpt: str | None = None
    last_mdns_info: dict | None = None

    for raw_line in log_out.splitlines():
        # fula_go uses zap with ISO 8601 timestamps at the start of
        # each line. Pull the timestamp prefix for ordering.
        ts_match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)", raw_line)
        ts = ts_match.group(1) if ts_match else None

        if _MDNS_LOADED_RE.search(raw_line):
            mdns_count += 1
            if ts:
                last_mdns_ts = ts
            # Try to extract the embedded infoSlice JSON for the
            # latest mDNS broadcast — it carries BloxPeerIdString,
            # IpfsClusterID, PoolName, Authorizer. Useful even when
            # config.yaml read fails for some reason.
            info_match = re.search(r'"infoSlice":\s*(\{[^}]+\})', raw_line)
            if info_match:
                try:
                    last_mdns_info = json.loads(info_match.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
        elif _POOL_EVENT_KEYWORDS.search(raw_line) and not _MDNS_LOADED_RE.search(raw_line):
            if ts:
                last_pool_ts = ts
            # Bound the excerpt — full lines can be 500+ chars.
            last_pool_excerpt = raw_line[:300]

    out["mdns_broadcasts_in_tail"] = mdns_count
    if last_mdns_ts:
        out["last_mdns_loaded_ts"] = last_mdns_ts
    if last_pool_ts:
        out["last_pool_event_ts"] = last_pool_ts
    if last_pool_excerpt:
        out["last_pool_event_excerpt"] = last_pool_excerpt
    if isinstance(last_mdns_info, dict):
        # Surface only the four known infoSlice keys; drop anything
        # else to avoid leaking unexpected fields if the format
        # changes upstream.
        info_out = {}
        for key in ("BloxPeerIdString", "IpfsClusterID", "PoolName",
                    "Authorizer", "HardwareID"):
            v = last_mdns_info.get(key)
            if isinstance(v, str):
                info_out[key] = v
        if info_out:
            out["last_mdns_info"] = info_out

    return out
