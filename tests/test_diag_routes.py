"""C3 — GET /diag/{tool} route tests."""
from __future__ import annotations

import pytest


# All 16 tool names that should resolve as path params.
ALL_TOOLS = [
    "internet", "relay", "time", "power", "storage", "containers",
    "wireguard", "heartbeat", "events", "readiness", "summary",
    "discovery_state", "systemd_services", "network_interface",
    "uniondrive", "identity_health",
]


@pytest.mark.parametrize("tool", ALL_TOOLS)
def test_diag_route_returns_200_via_mock_executor(client, tool):
    """The default `client` fixture uses MockDiagExecutor → every diag
    name should return 200 + a JSON object."""
    r = client.get(f"/diag/{tool}")
    assert r.status_code == 200, f"{tool}: {r.text}"
    body = r.json()
    assert isinstance(body, dict)


def test_diag_route_unknown_tool_returns_422(client):
    """Closed Literal type on the path param → unknown names get 422
    (FastAPI's body-validation error) rather than 404."""
    r = client.get("/diag/nonexistent")
    assert r.status_code == 422


def test_diag_route_executor_unknown_tool_returns_404(client_with_real_diag, monkeypatch):
    """If the path enum somehow grew faster than the executor's dispatch
    table, the route handler maps KeyError → 404."""
    from src.tools.diag_impls import RealDiagExecutor, UnknownToolError

    class Partial(RealDiagExecutor):
        async def __call__(self, tool, args):
            raise UnknownToolError(tool)

    client_with_real_diag.app.state.tool_executor = Partial()
    r = client_with_real_diag.get("/diag/internet")
    assert r.status_code == 404


def test_diag_route_executor_exception_returns_500(client_with_real_diag, monkeypatch):
    """If an impl raises an unexpected exception, the route handler maps
    to 500 with a bounded detail string."""
    class Bad:
        name = "bad"
        async def __call__(self, tool, args):
            raise RuntimeError("simulated impl crash")

    client_with_real_diag.app.state.tool_executor = Bad()
    r = client_with_real_diag.get("/diag/internet")
    assert r.status_code == 500
