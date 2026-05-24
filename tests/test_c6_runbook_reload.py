"""C6 — RunbookLoader tests.

Covers:
  - load_initial happy path + missing + malformed
  - reload happy path
  - reload refuses on missing / malformed / schema-bump / downgrade
  - events.jsonl emission per reload outcome
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime.runbook_loader import RunbookLoader


def _runbook(rv: int = 1, sv: int = 1) -> str:
    return (
        "---\n"
        f"runbook_version: {rv}\n"
        f"schema_version: {sv}\n"
        f"last_updated: 2026-05-24\n"
        "---\n"
        "# body\n"
    )


def test_load_initial_happy_path(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text(_runbook(rv=3))
    events = tmp_path / "events.jsonl"
    loader = RunbookLoader(path=str(rb), events_log_path=str(events))
    assert loader.load_initial() is True
    assert loader.get_version() == 3
    assert "body" in loader.get_text()
    # Event emitted
    lines = events.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["category"] == "runbook_loaded"


def test_load_initial_returns_false_when_file_missing(tmp_path):
    loader = RunbookLoader(
        path=str(tmp_path / "no-such-file"),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    assert loader.load_initial() is False
    assert loader.get_version() == 0


def test_load_initial_returns_false_when_malformed(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text("no frontmatter at all\n")
    loader = RunbookLoader(
        path=str(rb),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    assert loader.load_initial() is False


def test_reload_accepts_higher_version(tmp_path):
    rb = tmp_path / "runbook.md"
    events = tmp_path / "events.jsonl"
    rb.write_text(_runbook(rv=1))
    loader = RunbookLoader(path=str(rb), events_log_path=str(events))
    loader.load_initial()
    rb.write_text(_runbook(rv=2))
    r = loader.reload()
    assert r["outcome"] == "accepted"
    assert r["old_runbook_version"] == 1
    assert r["new_runbook_version"] == 2
    assert loader.get_version() == 2


def test_reload_refuses_downgrade(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text(_runbook(rv=5))
    loader = RunbookLoader(
        path=str(rb),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    loader.load_initial()
    rb.write_text(_runbook(rv=3))
    r = loader.reload()
    assert r["outcome"] == "refused_downgrade"
    assert loader.get_version() == 5  # unchanged


def test_reload_refuses_same_version(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text(_runbook(rv=3))
    loader = RunbookLoader(
        path=str(rb),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    loader.load_initial()
    rb.write_text(_runbook(rv=3))  # same version on disk
    r = loader.reload()
    assert r["outcome"] == "refused_downgrade"


def test_reload_refuses_schema_bump(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text(_runbook(rv=1, sv=1))
    loader = RunbookLoader(
        path=str(rb),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    loader.load_initial()
    rb.write_text(_runbook(rv=2, sv=2))  # schema bump
    r = loader.reload()
    assert r["outcome"] == "refused_schema"
    assert loader.get_version() == 1


def test_reload_refuses_malformed_runbook(tmp_path):
    rb = tmp_path / "runbook.md"
    rb.write_text(_runbook(rv=1))
    loader = RunbookLoader(
        path=str(rb),
        events_log_path=str(tmp_path / "events.jsonl"),
    )
    loader.load_initial()
    rb.write_text("broken file no fence\n")
    r = loader.reload()
    assert r["outcome"] == "refused_malformed"


def test_every_reload_emits_event(tmp_path):
    rb = tmp_path / "runbook.md"
    events = tmp_path / "events.jsonl"
    rb.write_text(_runbook(rv=1))
    loader = RunbookLoader(path=str(rb), events_log_path=str(events))
    loader.load_initial()  # 1 event
    rb.write_text(_runbook(rv=2))
    loader.reload()  # 1 event
    rb.write_text(_runbook(rv=1))  # downgrade
    loader.reload()  # 1 event
    lines = events.read_text().splitlines()
    assert len(lines) == 3
    outcomes = []
    for L in lines:
        rec = json.loads(L)
        if rec["category"] == "runbook_loaded":
            outcomes.append("loaded")
        else:
            detail = json.loads(rec["detail"])
            outcomes.append(detail.get("outcome"))
    assert outcomes == ["loaded", "accepted", "refused_downgrade"]
