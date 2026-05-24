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


def _staged_whitelist(tmp_path) -> Path | None:
    """Copy the real fula-ota action_whitelist.json into tmp so the
    executor has the canonical trust boundary. Falls back to None when
    the fula-ota sibling isn't available (tests that need the executor
    skip cleanly in that case)."""
    api = _locate_fula_ota_api_dir()
    if api is None:
        return None
    wl = api.parent / "action_whitelist.json"
    if not wl.is_file():
        return None
    dest = tmp_path / "action_whitelist.json"
    shutil.copy(wl, dest)
    return dest


@pytest.fixture
def client(schema_dir_with_all_required, tmp_path, monkeypatch):
    """Build a TestClient pointing the app at our staged schema dir +
    whitelist + writable audit log path.

    Overrides the production RealDiagExecutor with MockDiagExecutor so
    tests get deterministic per-tool responses (the real one shells out
    to docker / kubo / dmesg / journalctl, none of which work on the
    Windows dev box or in CI runners). Tests that specifically exercise
    the real path use the `client_with_real_diag` fixture instead.
    """
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))

    # C4: point the executor at a staged whitelist + writable audit/sec.
    wl = _staged_whitelist(tmp_path)
    if wl is not None:
        monkeypatch.setenv("BLOX_AI_WHITELIST_PATH", str(wl))
    audit_log = tmp_path / "ai-actions.jsonl"
    monkeypatch.setenv("BLOX_AI_AUDIT_LOG_PATH", str(audit_log))
    sec_code = tmp_path / "security-code"
    sec_code.write_text("1234")
    monkeypatch.setenv("BLOX_AI_SECURITY_CODE_PATH", str(sec_code))
    secret = tmp_path / "approval-secret"
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH", str(secret))
    flag_dir = tmp_path / "commands"
    flag_dir.mkdir()
    monkeypatch.setenv("BLOX_AI_COMMANDS_FLAG_DIR", str(flag_dir))

    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.runtime.mock_diag", "src.tools.diag_impls",
                "src.tools.executor", "src.tools.audit",
                "src.tools.approval_token",
                "src.session.tool_call_loop", "src.routes.troubleshoot",
                "src.routes.diag", "src.routes.execute"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app
    from src.runtime.mock_diag import MockDiagExecutor
    with TestClient(fresh_app) as c:
        c.app.state.tool_executor = MockDiagExecutor()
        # Expose paths for tests that want to read the audit log
        c.app.state.audit_log_path = str(audit_log)
        c.app.state.security_code_path = str(sec_code)
        yield c


@pytest.fixture
def client_with_real_diag(schema_dir_with_all_required, tmp_path, monkeypatch):
    """Variant of `client` that keeps RealDiagExecutor in place. Used by
    C3 tests that mock subprocess + state-file reads at the impl-module
    level rather than at the executor level."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("BLOX_AI_SCHEMA_DIR", str(schema_dir_with_all_required))
    wl = _staged_whitelist(tmp_path)
    if wl is not None:
        monkeypatch.setenv("BLOX_AI_WHITELIST_PATH", str(wl))
    audit_log = tmp_path / "ai-actions.jsonl"
    monkeypatch.setenv("BLOX_AI_AUDIT_LOG_PATH", str(audit_log))
    monkeypatch.setenv("BLOX_AI_APPROVAL_SECRET_PATH",
                       str(tmp_path / "approval-secret"))
    for mod in ("src.app", "src.schemas", "src.runtime.mock_backend",
                "src.tools.diag_impls", "src.tools.executor",
                "src.tools.audit", "src.tools.approval_token",
                "src.session.tool_call_loop", "src.routes.troubleshoot",
                "src.routes.diag", "src.routes.execute"):
        sys.modules.pop(mod, None)
    from src.app import app as fresh_app
    with TestClient(fresh_app) as c:
        yield c
