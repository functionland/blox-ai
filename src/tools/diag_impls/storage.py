"""diag/storage — df + ext4 errors + dmesg I/O errors + smartctl health."""
from __future__ import annotations

import re
from pathlib import Path

from src.tools.diag_impls._helpers import run_subprocess


# Mounts we care about for blox storage triage. The schema's `df` value
# is free-form so we emit a per-mount dict with size+free+pct.
WATCHED_MOUNTS = ("/", "/uniondrive", "/var/log/fula")


def diag_storage() -> dict:
    df = _df_for_mounts(WATCHED_MOUNTS)
    ext4 = _ext4_errors_count()
    dmesg = _dmesg_io_errors_recent()
    smart = _smartctl_health()
    out = {
        "df": df,
        "ext4_errors_count": ext4,
        "dmesg_io_errors_1h": dmesg,
    }
    if smart in ("PASSED", "FAILED", "unknown"):
        out["smartctl_health"] = smart
    return out


def _df_for_mounts(mounts: tuple[str, ...]) -> dict:
    rc, out, _ = run_subprocess(["df", "-PB1", *mounts], timeout_s=5.0)
    if rc != 0 or not out:
        return {}
    by_mount: dict[str, dict] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        size, used, avail = parts[1], parts[2], parts[3]
        mount = parts[5]
        try:
            by_mount[mount] = {
                "size_bytes": int(size),
                "used_bytes": int(used),
                "avail_bytes": int(avail),
            }
        except ValueError:
            continue
    return by_mount


def _ext4_errors_count() -> int:
    """Sum errors_count across all ext4 sysfs entries. Returns 0 if sysfs
    unreadable (e.g. running in a container that masked /sys)."""
    total = 0
    for p in Path("/sys/fs/ext4").glob("*/errors_count"):
        try:
            total += int(p.read_text().strip())
        except (OSError, ValueError):
            continue
    return total


def _dmesg_io_errors_recent() -> int:
    """Count I/O error lines from the last hour. dmesg requires CAP_SYSLOG
    or the kernel's permissive dmesg_restrict — best-effort."""
    rc, out, _ = run_subprocess(
        ["dmesg", "--ctime", "--since", "1 hour ago"],
        timeout_s=5.0,
    )
    if rc != 0:
        return 0
    return len(re.findall(r"i/o error", out, flags=re.IGNORECASE))


def _smartctl_health() -> str:
    rc, out, _ = run_subprocess(["smartctl", "-H", "/dev/sda"], timeout_s=5.0)
    if rc < 0:
        return "unknown"
    if "PASSED" in out:
        return "PASSED"
    if "FAILED" in out:
        return "FAILED"
    return "unknown"
