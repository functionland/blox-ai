"""Shared helpers for the diag/* impls.

Stdlib-only on purpose: the impls run in tight per-request latency
budgets (and diag/summary in a 5s wall-clock budget across all of
them), so we keep import overhead minimal.
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("blox-ai.diag")


def now_iso() -> str:
    """ISO 8601 UTC with Z suffix, millisecond precision (matches the
    iso8601_datetime $def regex in diag_responses.schema.json)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def read_state(path: str | Path, default: dict | None = None) -> dict:
    """Read a /run/fula-*.state JSON file. Returns `default` (or {})
    on any error — missing file is treated as 'subsystem hasn't run yet'
    rather than an exception that propagates."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else (default or {})
    except OSError:
        return default or {}
    except json.JSONDecodeError as e:
        logger.warning("state file %s is malformed: %s", path, e)
        return default or {}


def run_subprocess(
    cmd: list[str],
    timeout_s: float = 5.0,
    text: bool = True,
) -> tuple[int, str, str]:
    """Run a subprocess with a strict timeout. Returns (rc, stdout, stderr).
    On timeout / FileNotFoundError, returns (-1, "", <reason>) — never raises."""
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=text,
            timeout=timeout_s,
            check=False,
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"timeout after {timeout_s}s"
    except FileNotFoundError as e:
        return -1, "", f"command not found: {e.filename or cmd[0]}"
    except OSError as e:
        return -1, "", f"OS error: {e}"


def https_head(url: str, timeout_s: float = 5.0) -> tuple[bool, int | None, float]:
    """HEAD request. Returns (ok, http_status, latency_ms).

    `ok` is True iff the request completed AND status is 2xx/3xx. Latency
    is wall-clock from request start to response. Stdlib urllib — no
    runtime requests/httpx dependency.
    """
    import time
    start = time.monotonic()
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            latency = (time.monotonic() - start) * 1000
            ok = 200 <= resp.status < 400
            return ok, resp.status, latency
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - start) * 1000
        return False, e.code, latency
    except (urllib.error.URLError, OSError, TimeoutError):
        latency = (time.monotonic() - start) * 1000
        return False, None, latency


def dns_lookup(host: str, timeout_s: float = 3.0) -> bool:
    """True iff `host` resolves. socket.gethostbyname has no per-call
    timeout option in the stdlib, but it's bounded by the system resolver."""
    socket.setdefaulttimeout(timeout_s)
    try:
        socket.gethostbyname(host)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)


def http_get_json(url: str, timeout_s: float = 5.0) -> Any:
    """GET that returns parsed JSON. Returns None on any error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, json.JSONDecodeError, TimeoutError):
        return None


def http_post_json(url: str, body: dict, timeout_s: float = 5.0) -> Any:
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
