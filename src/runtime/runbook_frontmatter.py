"""Vendored from fula-ota.

Source-of-truth lives at:
  fula-ota/docker/fxsupport/linux/plugins/blox-ai/runbook_frontmatter.py

Kept byte-for-byte equivalent in semantics. A CI test asserts the hash
matches the upstream file when both are checked out side-by-side; drift
causes the container to refuse a SIGHUP swap (the schema_version bump
or downgrade-protection would catch it at runtime, but the test catches
it earlier).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_REQUIRED_KEYS = {"runbook_version", "schema_version", "last_updated"}
_KEY_RE = re.compile(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+?)\s*$")
_FENCE = "---"


class RunbookFrontmatterError(ValueError):
    pass


@dataclass(frozen=True)
class RunbookFrontmatter:
    runbook_version: int
    schema_version: int
    last_updated: str

    def is_newer_than(self, other: Optional["RunbookFrontmatter"]) -> bool:
        if other is None:
            return True
        if self.schema_version != other.schema_version:
            raise RunbookFrontmatterError(
                f"refusing to compare versions across schema_version "
                f"{other.schema_version} → {self.schema_version}; "
                f"container must restart, not SIGHUP-reload"
            )
        return self.runbook_version > other.runbook_version


def parse(text: str) -> RunbookFrontmatter:
    if not text.startswith(_FENCE + "\n") and not text.startswith(_FENCE + "\r\n"):
        raise RunbookFrontmatterError(
            "runbook.md must begin with a '---' fence on its own line"
        )
    lines = text.splitlines()
    if not lines or lines[0] != _FENCE:
        raise RunbookFrontmatterError("missing opening '---' fence")
    closing_idx = None
    for i in range(1, len(lines)):
        if lines[i] == _FENCE:
            closing_idx = i
            break
    if closing_idx is None:
        raise RunbookFrontmatterError("missing closing '---' fence")
    if closing_idx == 1:
        raise RunbookFrontmatterError("empty frontmatter block")

    found: dict[str, str] = {}
    for raw in lines[1:closing_idx]:
        if not raw.strip():
            continue
        m = _KEY_RE.match(raw)
        if not m:
            raise RunbookFrontmatterError(
                f"unparseable frontmatter line: {raw!r}"
            )
        key, value = m.group(1), m.group(2)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        found[key] = value

    missing = _REQUIRED_KEYS - found.keys()
    if missing:
        raise RunbookFrontmatterError(
            f"missing required frontmatter keys: {sorted(missing)}"
        )

    try:
        rv = int(found["runbook_version"])
        sv = int(found["schema_version"])
    except ValueError as e:
        raise RunbookFrontmatterError(
            f"runbook_version / schema_version must be integers: {e}"
        )
    if rv < 1:
        raise RunbookFrontmatterError(
            f"runbook_version must be >= 1, got {rv}"
        )
    if sv < 1:
        raise RunbookFrontmatterError(
            f"schema_version must be >= 1, got {sv}"
        )

    return RunbookFrontmatter(
        runbook_version=rv,
        schema_version=sv,
        last_updated=found["last_updated"],
    )


def parse_file(path: str) -> RunbookFrontmatter:
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
