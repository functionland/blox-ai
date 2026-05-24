"""Shared pytest fixtures."""
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make `src` importable without packaging.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _locate_fula_ota_api_dir() -> Path | None:
    """Return the path to fula-ota's plugins/blox-ai/api directory, or None.

    Priority:
      1. BLOX_AI_FULA_OTA_SCHEMA_DIR env var (explicit override; CI + lab use this)
      2. Sibling checkout at ../fula-ota/docker/fxsupport/linux/plugins/blox-ai/api
      3. None (caller falls back to permissive stubs)
    """
    env = os.environ.get("BLOX_AI_FULA_OTA_SCHEMA_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    sibling = (
        _REPO_ROOT.parent / "fula-ota" / "docker" / "fxsupport" / "linux"
        / "plugins" / "blox-ai" / "api"
    )
    if sibling.is_dir():
        return sibling
    return None


@pytest.fixture
def schema_dir_with_all_required(tmp_path) -> Path:
    """Stage a directory with every schema the SchemaRegistry insists on.

    Sources from fula-ota when discoverable via env var or sibling
    checkout (preferred: keeps tests honest against the live contract).
    Falls back to permissive stubs when neither is available — note that
    some tests assert on schema STRICTNESS and will fail under the stub
    fallback, which is the right signal: "set BLOX_AI_FULA_OTA_SCHEMA_DIR
    to run the full suite."
    """
    fula_ota_api = _locate_fula_ota_api_dir()
    if fula_ota_api is not None:
        for p in fula_ota_api.glob("*.schema.json"):
            shutil.copy(p, tmp_path / p.name)
    else:
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


def _real_schemas_in_use() -> bool:
    """True iff the schema_dir fixture will use the REAL fula-ota schemas
    rather than the permissive stub fallback. Tests that assert on schema
    STRICTNESS use this to skip cleanly under stubs."""
    return _locate_fula_ota_api_dir() is not None


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
