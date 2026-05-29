"""Tests for POST /diag/bundle — the one-shot "Raw diagnostics" snapshot
that backs the app's Raw Diagnostics card.

The bundle runs every read-only diag tool concurrently and returns
{generated_at, tools: {<name without diag/ prefix>: result}}. diag/summary
is excluded (it re-runs a subset internally, so bundling it would duplicate
work). These tests use the `client` fixture's MockDiagExecutor so every
tool resolves to a canned payload — the bundle's job under test is the
concurrency + shaping, not the tool internals (those are covered by the
per-tool diag tests + the real-impl tests).
"""
from __future__ import annotations

from datetime import datetime

from src.tools.diag_impls import known_tools


def _expected_keys() -> set[str]:
    """Every known tool minus diag/summary, with the diag/ prefix dropped —
    computed from the live dispatch table so this test can't drift from the
    tool set."""
    return {t.removeprefix("diag/") for t in known_tools() if t != "diag/summary"}


def test_bundle_returns_generated_at_and_tools(client):
    r = client.post("/diag/bundle")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"generated_at", "tools"}
    # generated_at must be a valid ISO-8601 timestamp.
    datetime.fromisoformat(body["generated_at"])
    assert isinstance(body["tools"], dict)


def test_bundle_covers_every_known_tool_except_summary(client):
    r = client.post("/diag/bundle")
    tools = r.json()["tools"]
    assert set(tools.keys()) == _expected_keys()
    assert "summary" not in tools, "diag/summary must be excluded from the bundle"


def test_bundle_keys_drop_the_diag_prefix(client):
    r = client.post("/diag/bundle")
    tools = r.json()["tools"]
    assert "internet" in tools
    assert "diag/internet" not in tools


def test_bundle_tool_payloads_are_real_not_errors(client):
    """With MockDiagExecutor every tool resolves, so no entry should carry
    an 'error' key. (A tool the executor can't run would surface as
    {'error': ...} for that entry rather than failing the whole snapshot —
    that isolation is the point, but here nothing should error.)"""
    r = client.post("/diag/bundle")
    tools = r.json()["tools"]
    for name, payload in tools.items():
        assert isinstance(payload, dict), f"{name} payload must be a dict"
        assert "error" not in payload, f"{name} unexpectedly errored: {payload}"
    # Spot-check one canned payload flowed through verbatim (proves the
    # executor was actually invoked, not just an empty shell returned).
    assert tools["internet"]["dns_ok"] is True


def test_get_bundle_does_not_hit_the_bundle_handler(client):
    """GET /diag/bundle must NOT reach the bundle handler. It collides with
    the GET /diag/{tool} Literal enum, where tool='bundle' is invalid → 422.
    POST is the contract (the core BLE proxy always POSTs json={}), chosen
    precisely to avoid this collision."""
    r = client.get("/diag/bundle")
    assert r.status_code == 422
