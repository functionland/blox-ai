"""C4 — audit log writer tests."""
from __future__ import annotations

import json

import pytest

from src.tools.audit import append, _rotate_if_needed


def test_append_writes_jsonl(tmp_path):
    p = tmp_path / "audit.jsonl"
    assert append({"ts": "t", "x": 1}, path=str(p)) is True
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"ts": "t", "x": 1}


def test_append_is_append_only(tmp_path):
    p = tmp_path / "audit.jsonl"
    append({"a": 1}, path=str(p))
    append({"a": 2}, path=str(p))
    append({"a": 3}, path=str(p))
    lines = p.read_text().splitlines()
    assert [json.loads(L)["a"] for L in lines] == [1, 2, 3]


def test_append_creates_parent_dir(tmp_path):
    p = tmp_path / "new" / "dir" / "audit.jsonl"
    assert append({"x": 1}, path=str(p)) is True
    assert p.exists()


def test_rotate_when_over_threshold(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text("x" * 200)
    # tiny threshold to force rotation
    _rotate_if_needed(p, max_bytes=100, backup_count=3)
    # primary moved to .1
    assert not p.exists() or p.stat().st_size == 0
    assert (tmp_path / "audit.jsonl.1").exists()


def test_no_rotation_under_threshold(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text("small")
    _rotate_if_needed(p, max_bytes=1000, backup_count=3)
    assert p.read_text() == "small"
    assert not (tmp_path / "audit.jsonl.1").exists()


def test_rotation_evicts_oldest_backup(tmp_path):
    p = tmp_path / "audit.jsonl"
    # Stage existing rotations 1..3 (limit)
    (tmp_path / "audit.jsonl.1").write_text("backup1")
    (tmp_path / "audit.jsonl.2").write_text("backup2")
    (tmp_path / "audit.jsonl.3").write_text("backup3")
    p.write_text("x" * 200)
    _rotate_if_needed(p, max_bytes=100, backup_count=3)
    # backup3 should have been deleted (was the oldest); but in our
    # implementation we delete .N (backup_count) FIRST then shift down,
    # so .3 is gone; .2 → .3; .1 → .2; primary → .1
    assert not (tmp_path / "audit.jsonl.4").exists()
    assert (tmp_path / "audit.jsonl.1").exists()
    # The original backup3 content should NOT survive
    assert "backup3" not in (tmp_path / "audit.jsonl.3").read_text()
