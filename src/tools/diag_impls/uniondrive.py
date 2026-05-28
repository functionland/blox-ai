"""diag/uniondrive — mergerfs mount state + backing filesystem health.

The fula stack stores everything under /uniondrive. On RK3588 lab
devices this is typically `/media/pi/sda1` mounted via mergerfs (a
union FS) so user data + container state can be moved across disks
without container reconfig. Trees branch on:
  - /uniondrive not mounted at all → all containers will fail
  - mergerfs not installed (vanilla install missing dep) → unsupported
  - backing FS ext4 errors_count > 0 → silent corruption brewing
  - dmesg I/O errors in the last hour → disk going bad
  - disk usage > 95% → containers will start failing soon

Read-only; never mutates. Output is the union of:
  - mount-table parse for /uniondrive
  - `df -B1 /uniondrive` for byte-accurate sizes (so trees can compute
    a percentage without re-parsing human-readable suffixes)
  - /sys/fs/ext4/<dev>/errors_count for the backing block device
  - `dmesg --since 1h | grep -c i/o error` (best-effort; needs CAP_SYSLOG
    or root — falls back to None when denied)
"""
from __future__ import annotations

import os

from src.tools.diag_impls._helpers import read_state, run_subprocess


UNIONDRIVE_PATH = "/uniondrive"


def diag_uniondrive() -> dict:
    out: dict = {
        # Tristate per advisor pattern: True / False / None.
        "mounted": False,
        "mergerfs_installed": _have("mergerfs"),
    }
    mount_info = _parse_mount_for(UNIONDRIVE_PATH)
    if mount_info:
        out["mounted"] = True
        out["mount_source"] = mount_info["source"]
        out["mount_fstype"] = mount_info["fstype"]
        out["mount_options"] = mount_info["options"][:500]

    if out["mergerfs_installed"]:
        rc, ver_out, _ = run_subprocess(["mergerfs", "--version"], timeout_s=2.0)
        if rc == 0 and ver_out:
            # Example line: "mergerfs version: 2.33.5"
            for line in ver_out.splitlines():
                if "version" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        out["mergerfs_version"] = parts[-1].strip()[:64]
                        break

    if out["mounted"]:
        sizes = _df_bytes(UNIONDRIVE_PATH)
        if sizes:
            out.update(sizes)
        # Backing block device for ext4 errors_count. mergerfs source
        # is the underlying mount (`/media/pi/sda1`); resolve the
        # device behind that mount.
        backing_dev = _backing_device_for(mount_info["source"] if mount_info else None)
        if backing_dev:
            out["backing_device"] = backing_dev
            errs = _ext4_errors_count(backing_dev)
            if errs is not None:
                out["ext4_errors_count"] = errs

    io_errs = _dmesg_io_errors_last_hour()
    if io_errs is not None:
        out["dmesg_io_errors_1h"] = io_errs

    return out


def _have(cmd: str) -> bool:
    rc, _, _ = run_subprocess(["which", cmd], timeout_s=1.5)
    return rc == 0


def _parse_mount_for(path: str) -> dict | None:
    """Find the FIRST mount entry whose target is `path`. Returns
    {source, fstype, options} or None when not mounted."""
    rc, out, _ = run_subprocess(["mount"], timeout_s=2.0)
    if rc != 0 or not out:
        return None
    # mount output: "<source> on <target> type <fstype> (<options>)"
    for line in out.splitlines():
        parts = line.split(" ")
        if len(parts) < 6:
            continue
        try:
            on_idx = parts.index("on")
            type_idx = parts.index("type")
        except ValueError:
            continue
        target = " ".join(parts[on_idx + 1:type_idx])
        if target != path:
            continue
        source = " ".join(parts[:on_idx])
        fstype = parts[type_idx + 1] if type_idx + 1 < len(parts) else ""
        # Options are inside parens after the fstype.
        opts = ""
        line_after_fstype = " ".join(parts[type_idx + 2:])
        if line_after_fstype.startswith("(") and line_after_fstype.endswith(")"):
            opts = line_after_fstype[1:-1]
        return {"source": source, "fstype": fstype, "options": opts}
    return None


def _df_bytes(path: str) -> dict | None:
    """`df -B1 <path>` → {size_bytes, used_bytes, avail_bytes,
    use_percent}. Returns None on any error."""
    rc, out, _ = run_subprocess(["df", "-B1", path], timeout_s=2.0)
    if rc != 0 or not out:
        return None
    lines = out.splitlines()
    if len(lines) < 2:
        return None
    # Header: Filesystem 1B-blocks Used Available Use% Mounted on
    # On long filesystem names df wraps to two lines; the data is
    # ALWAYS in the LAST line.
    parts = lines[-1].split()
    if len(parts) < 6:
        return None
    try:
        size = int(parts[-5])
        used = int(parts[-4])
        avail = int(parts[-3])
        pct_str = parts[-2].rstrip("%")
        pct = int(pct_str) if pct_str.isdigit() else None
    except (ValueError, IndexError):
        return None
    out_dict: dict = {
        "size_bytes": size,
        "used_bytes": used,
        "avail_bytes": avail,
    }
    if pct is not None:
        out_dict["use_percent"] = pct
    return out_dict


def _backing_device_for(source: str | None) -> str | None:
    """Resolve a mount source (which may be a path like /media/pi/sda1
    for mergerfs, or a block device like /dev/sda1) to the leaf block
    device name (e.g. 'sda1'). Used to look up ext4 errors_count under
    /sys/fs/ext4/."""
    if not source:
        return None
    # If source is itself a path that is ALSO mounted, recurse into the
    # mount table to find the actual block device behind it. This is
    # the common mergerfs case where source='/media/pi/sda1' and the
    # block device is '/dev/sda1' mounted at '/media/pi/sda1'.
    if source.startswith("/dev/"):
        return os.path.basename(source)
    sub = _parse_mount_for(source)
    if sub and sub["source"].startswith("/dev/"):
        return os.path.basename(sub["source"])
    return None


def _ext4_errors_count(device: str) -> int | None:
    path = f"/sys/fs/ext4/{device}/errors_count"
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read().strip()
            return int(txt) if txt.isdigit() else None
    except (OSError, ValueError):
        return None


def _dmesg_io_errors_last_hour() -> int | None:
    """Count case-insensitive 'i/o error' lines in dmesg since 1h ago.
    Returns None when dmesg returned no output (denied or empty)."""
    rc, out, _ = run_subprocess(
        ["dmesg", "--since", "1 hour ago"],
        timeout_s=3.0,
    )
    if rc != 0:
        return None
    if not out:
        return 0
    return sum(1 for line in out.splitlines() if "i/o error" in line.lower())
