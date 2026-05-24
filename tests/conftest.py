"""Shared pytest fixtures."""
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make `src` importable without packaging.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def schema_dir_with_all_required(tmp_path) -> Path:
    """Stage a directory with every schema the SchemaRegistry insists on.

    Sources from fula-ota when the sibling repo is present (preferred:
    keeps tests honest against the live contract). Otherwise falls back
    to a minimal valid stub per filename so the test suite is
    self-contained when fula-ota isn't checked out next door.
    """
    fula_ota_api = (
        _REPO_ROOT.parent / "fula-ota" / "docker" / "fxsupport" / "linux"
        / "plugins" / "blox-ai" / "api"
    )
    if fula_ota_api.is_dir():
        for p in fula_ota_api.glob("*.schema.json"):
            shutil.copy(p, tmp_path / p.name)
    else:
        # Self-contained fallback: minimal valid schema per required name.
        # Closed (additionalProperties:false) + empty required so any
        # payload validates.
        from src.schemas import REQUIRED_SCHEMAS
        import json
        for name in REQUIRED_SCHEMAS:
            (tmp_path / name).write_text(json.dumps({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": f"https://example.test/{name}",
                "title": name,
                "type": "object",
                "additionalProperties": True,
            }))
    return tmp_path


@pytest.fixture
def client(schema_dir_with_all_required, monkeypatch):
    """Build a TestClient pointing the app at our staged schema dir."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    # Re-import app so the lifespan picks up our env.
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app
    with TestClient(fresh_app) as c:
        yield c
