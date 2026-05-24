"""C1 — SchemaRegistry tests.

Hard contract: the container CANNOT start with a missing or malformed
schema. Any drift in fula-ota's plugin api/ files must surface here as
a load-time failure, not as a permissive runtime "endpoint accepts
anything" silent regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas import REQUIRED_SCHEMAS, SchemaLoadError, SchemaRegistry


def test_load_succeeds_with_all_schemas(schema_dir_with_all_required):
    reg = SchemaRegistry.load(str(schema_dir_with_all_required))
    assert len(reg) == len(REQUIRED_SCHEMAS)
    for name in REQUIRED_SCHEMAS:
        assert name in reg


def test_load_refuses_when_dir_missing(tmp_path):
    bogus = tmp_path / "no-such-dir"
    with pytest.raises(SchemaLoadError) as exc:
        SchemaRegistry.load(str(bogus))
    assert "not found" in str(exc.value)


def test_load_refuses_when_required_schema_missing(tmp_path, schema_dir_with_all_required):
    # Remove one of the required files; load must fail with a clear message
    # naming what's missing.
    victim = "sse_events.schema.json"
    (schema_dir_with_all_required / victim).unlink()
    with pytest.raises(SchemaLoadError) as exc:
        SchemaRegistry.load(str(schema_dir_with_all_required))
    assert victim in str(exc.value)


def test_load_refuses_when_schema_is_invalid_json(tmp_path, schema_dir_with_all_required):
    (schema_dir_with_all_required / "sse_events.schema.json").write_text("{ this is not json")
    with pytest.raises(SchemaLoadError) as exc:
        SchemaRegistry.load(str(schema_dir_with_all_required))
    assert "invalid JSON" in str(exc.value)


def test_load_refuses_when_schema_violates_draft_2020_12(tmp_path, schema_dir_with_all_required):
    # 'type' must be a string or array of strings, not an int.
    (schema_dir_with_all_required / "sse_events.schema.json").write_text(
        json.dumps({"type": 42})
    )
    with pytest.raises(SchemaLoadError) as exc:
        SchemaRegistry.load(str(schema_dir_with_all_required))
    assert "not a valid Draft 2020-12 schema" in str(exc.value)


def test_validator_for_returns_usable_validator(schema_dir_with_all_required):
    reg = SchemaRegistry.load(str(schema_dir_with_all_required))
    v = reg.validator_for("sse_events.schema.json")
    # Just verify it has the validator interface — actual SSE event
    # validation lives in C2's tests.
    assert hasattr(v, "validate")
    assert hasattr(v, "iter_errors")


def test_registry_dict_like_api(schema_dir_with_all_required):
    reg = SchemaRegistry.load(str(schema_dir_with_all_required))
    # Indexing
    s = reg["sse_events.schema.json"]
    assert isinstance(s, dict)
    # Membership
    assert "sse_events.schema.json" in reg
    assert "no-such-schema.json" not in reg


def test_registry_records_schema_dir(schema_dir_with_all_required):
    reg = SchemaRegistry.load(str(schema_dir_with_all_required))
    assert reg.schema_dir == str(schema_dir_with_all_required)
