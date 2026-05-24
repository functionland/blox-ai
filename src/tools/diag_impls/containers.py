"""diag/containers — docker ps + per-container inspect via docker-py.

Talks to the host docker daemon via the bind-mounted /var/run/docker.sock.
docker-py auto-detects the socket from the standard environment when
DOCKER_HOST isn't set.

Returns empty list when docker is unreachable (dev box, sock unmounted)."""
from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("blox-ai.diag.containers")


# Container names we care about. Returning only these keeps the response
# shape predictable and small (the host runs other containers we don't
# need to surface to the AI).
WATCHED_CONTAINERS = (
    "ipfs_host",
    "ipfs_cluster",
    "fula_go",
    "fula_pinning",
    "fula_gateway",
    "fula_fxsupport",
    "blox-ai",
)


def diag_containers() -> dict:
    try:
        import docker  # imported lazily to keep dev-machine boot path quiet
        from docker.errors import DockerException
    except ImportError:
        logger.warning("docker-py not installed; diag/containers returns empty")
        return {"containers": []}
    try:
        client = docker.from_env(timeout=5)
    except DockerException as e:
        logger.warning("docker daemon unreachable: %s", e)
        return {"containers": []}
    out: list[dict] = []
    try:
        # all=True so we see exited/dead/restarting too — that's the
        # interesting state for troubleshooting.
        for c in client.containers.list(all=True, ignore_removed=True):
            if c.name not in WATCHED_CONTAINERS:
                continue
            entry = _build_entry(c)
            if entry is not None:
                out.append(entry)
    except DockerException as e:
        logger.warning("docker list failed: %s", e)
    finally:
        try:
            client.close()
        except Exception:
            pass
    return {"containers": out}


def _build_entry(c: Any) -> dict | None:
    """Map a docker.containers.Container to our schema shape. Returns None
    if the inspect data is malformed (defensive — docker-py occasionally
    returns partial dicts during container lifecycle transitions)."""
    try:
        state = c.attrs.get("State", {})
        status = state.get("Status", "unknown")
        # Map docker statuses to our closed enum.
        valid_states = {"running", "restarting", "exited", "paused", "dead", "created"}
        if status not in valid_states:
            return None
        entry: dict = {"name": c.name, "state": status}
        oom = state.get("OOMKilled")
        if isinstance(oom, bool):
            entry["oom_killed"] = oom
        restart_count = c.attrs.get("RestartCount")
        if isinstance(restart_count, int) and restart_count >= 0:
            entry["restart_count"] = restart_count
        image = c.attrs.get("Config", {}).get("Image")
        if isinstance(image, str):
            entry["image"] = image
        started_at = state.get("StartedAt")
        if isinstance(started_at, str) and started_at and not started_at.startswith("0001"):
            # Docker returns "2026-05-24T19:00:00.000000000Z"; trim
            # subsecond precision to milliseconds for schema regex match.
            entry["started_at"] = _trim_to_ms(started_at)
        return entry
    except Exception as e:
        logger.warning("inspect for %s failed: %s", getattr(c, "name", "?"), e)
        return None


def _trim_to_ms(ts: str) -> str:
    """Trim 'YYYY-MM-DDTHH:MM:SS.ffffffffZ' to 'YYYY-MM-DDTHH:MM:SS.fffZ'.

    The schema's iso8601_datetime pattern accepts any sub-second precision
    but Python's strict pattern matching can be picky on nanosecond
    forms in some validator versions. Trimming is harmless + safer.
    """
    if "." in ts and ts.endswith("Z"):
        head, _, tail = ts[:-1].partition(".")
        return f"{head}.{tail[:3]}Z"
    return ts
