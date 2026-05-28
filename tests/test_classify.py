"""Phase 1.d — POST /troubleshoot/classify tests.

Uses MockBackend whose `classify` is deterministic (keyword fallback);
production RKLLMBackend uses real LLM inference."""
from __future__ import annotations

import pytest


def test_classify_disconnected(client):
    r = client.post("/troubleshoot/classify",
                     json={"prompt": "my Blox is offline and disconnected"})
    assert r.status_code == 200
    assert r.json() == {"scenario_id": "disconnected"}


def test_classify_not_earning(client):
    r = client.post("/troubleshoot/classify",
                     json={"prompt": "Why am I not earning any rewards?"})
    assert r.status_code == 200
    assert r.json() == {"scenario_id": "not-earning"}


def test_classify_cannot_join_pool(client):
    r = client.post("/troubleshoot/classify",
                     json={"prompt": "I want to join a pool but it fails"})
    assert r.status_code == 200
    assert r.json() == {"scenario_id": "cannot-join-pool"}


def test_classify_other_for_unclear_prompt(client):
    r = client.post("/troubleshoot/classify",
                     json={"prompt": "the LED is purple now"})
    assert r.status_code == 200
    assert r.json() == {"scenario_id": "other"}


def test_classify_rejects_empty_prompt(client):
    r = client.post("/troubleshoot/classify", json={"prompt": ""})
    assert r.status_code == 422


def test_classify_rejects_extra_fields(client):
    """Extra: forbid means caller typos surface as 422, not silently ignored."""
    r = client.post("/troubleshoot/classify",
                     json={"prompt": "x", "scenarioId": "x"})
    assert r.status_code == 422


def test_classify_returns_other_when_backend_lacks_method(client):
    """A backend that doesn't expose classify still gets a useful
    fallback response — the app's 'other' path leads to the existing
    AI mode, so the user is never blocked."""
    class NoClassifyBackend:
        name = "no-classify"
    client.app.state.backend = NoClassifyBackend()
    try:
        r = client.post("/troubleshoot/classify",
                         json={"prompt": "anything"})
        assert r.status_code == 200
        body = r.json()
        assert body["scenario_id"] == "other"
        assert body.get("reason") == "classifier_unavailable"
    finally:
        # Restore: pytest fixtures provide a MockBackend by default.
        # Subsequent tests in the same module rely on it.
        from src.runtime.mock_backend import MockBackend
        client.app.state.backend = MockBackend()


def test_classify_normalises_unexpected_backend_value(client):
    """Backend's classify SHOULD return one of 4 known values; if it
    returns something else (rare LLM hallucination), the route
    normalises to 'other' rather than passing garbage to the app."""
    class WeirdBackend:
        name = "weird"
        async def classify(self, prompt: str) -> str:
            return "snowman"   # not allowed
    client.app.state.backend = WeirdBackend()
    try:
        r = client.post("/troubleshoot/classify", json={"prompt": "x"})
        assert r.status_code == 200
        assert r.json() == {"scenario_id": "other"}
    finally:
        from src.runtime.mock_backend import MockBackend
        client.app.state.backend = MockBackend()


def test_classify_handles_backend_raising(client):
    """If classify raises (e.g. LLM crash), default to 'other'."""
    class CrashingBackend:
        name = "crash"
        async def classify(self, prompt: str) -> str:
            raise RuntimeError("LLM exploded")
    client.app.state.backend = CrashingBackend()
    try:
        r = client.post("/troubleshoot/classify", json={"prompt": "x"})
        assert r.status_code == 200
        assert r.json() == {"scenario_id": "other"}
    finally:
        from src.runtime.mock_backend import MockBackend
        client.app.state.backend = MockBackend()
