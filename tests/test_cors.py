"""Browser access (FxBlox Web): CORS allow-list + Origin guard — src/cors.py."""
from __future__ import annotations

import pytest

from src.cors import origin_allowed


BLOX_WEB = "https://blox.fx.land"
STAGING = "https://functionland.github.io"
EVIL = "https://evil.example"


def test_origin_allowed_defaults(monkeypatch):
    monkeypatch.delenv("BLOX_AI_CORS_ORIGINS", raising=False)
    assert origin_allowed(BLOX_WEB)
    assert origin_allowed(STAGING)
    assert origin_allowed("https://docs.fx.land")
    assert origin_allowed("http://localhost:5173")
    assert origin_allowed("http://127.0.0.1:4173")
    assert not origin_allowed(EVIL)
    assert not origin_allowed("http://blox.fx.land")  # scheme matters
    assert not origin_allowed(None)
    assert not origin_allowed("")


def test_origin_allowed_env_override(monkeypatch):
    monkeypatch.setenv("BLOX_AI_CORS_ORIGINS", "https://custom.example, https://other.example")
    assert origin_allowed("https://custom.example")
    assert origin_allowed("https://other.example")
    assert not origin_allowed(BLOX_WEB)
    assert origin_allowed("http://localhost:3000")  # dev origins stay allowed


def test_preflight_from_allowed_origin(client):
    r = client.options(
        "/health",
        headers={
            "Origin": BLOX_WEB,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type, x-fula-support",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == BLOX_WEB
    allowed_headers = r.headers["access-control-allow-headers"].lower()
    assert "content-type" in allowed_headers
    assert "x-fula-support" in allowed_headers
    assert "POST" in r.headers["access-control-allow-methods"]


def test_preflight_from_unknown_origin_gets_no_cors_headers(client):
    r = client.options(
        "/health",
        headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_get_from_allowed_origin_has_acao(client):
    r = client.get("/health", headers={"Origin": BLOX_WEB})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == BLOX_WEB


def test_get_from_unknown_origin_passes_without_acao(client):
    """Reads are not blocked server-side (the browser can't read them anyway)."""
    r = client.get("/health", headers={"Origin": EVIL})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cross_site_post_from_unknown_origin_is_403(client):
    r = client.post("/cancel", json={"session_id": "x"}, headers={"Origin": EVIL})
    assert r.status_code == 403
    assert r.text == "origin not allowed"


def test_post_without_origin_is_untouched(client):
    """Mobile app / curl / BLE proxy send no Origin header → normal handling."""
    r = client.post("/cancel", json={"session_id": "x"})
    assert r.status_code != 403


def test_post_from_allowed_origin_is_untouched(client):
    r = client.post("/cancel", json={"session_id": "x"}, headers={"Origin": BLOX_WEB})
    assert r.status_code != 403
    assert r.headers["access-control-allow-origin"] == BLOX_WEB


@pytest.mark.parametrize("path", ["/troubleshoot", "/execute-action", "/support/wireguard"])
def test_guard_covers_state_changing_routes(client, path):
    r = client.post(path, json={}, headers={"Origin": EVIL})
    assert r.status_code == 403
