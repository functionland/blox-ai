"""C1 — /health + /status route tests."""
from __future__ import annotations


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_status_returns_model_loaded_shape(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    # MockBackend.status_snapshot() + schema list
    for field in (
        "model_loaded", "model_backend", "runbook_version",
        "active_sessions", "npu_health", "last_error",
        "schemas_loaded", "schema_dir",
    ):
        assert field in body, f"missing field: {field}"
    assert body["model_backend"] == "mock"
    assert body["model_loaded"] is True
    assert body["active_sessions"] == 0


def test_status_reports_loaded_schemas(client):
    r = client.get("/status")
    body = r.json()
    loaded = set(body["schemas_loaded"])
    # All required schemas (per REQUIRED_SCHEMAS) MUST appear here.
    from src.schemas import REQUIRED_SCHEMAS
    for name in REQUIRED_SCHEMAS:
        assert name in loaded, f"{name} missing from /status schemas_loaded"


def test_health_does_not_require_lifespan_dependencies(client):
    """Deliberate: /health must NOT depend on backend / schema / runbook.
    Docker HEALTHCHECK polls it; should turn green before model warm-up
    finishes."""
    r = client.get("/health")
    assert r.status_code == 200
    # The body is intentionally minimal — nothing leaked beyond ok flag.
    assert r.json() == {"ok": True}
