"""C1 routes: /health (liveness) + /status (model + runtime details).

Both intentionally trivial — no auth, no rate limit, no body validation.
/health is what Docker's HEALTHCHECK polls; /status is what the BLE
`ai/status` proxy returns and what an operator hits during triage.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness. 200 + {ok:true} as soon as uvicorn is up.

    Deliberately does NOT depend on backend / schema / runbook loading —
    Docker's HEALTHCHECK should turn green BEFORE the model finishes
    loading so the container doesn't bounce in the warm-up window.
    """
    return {"ok": True}


@router.get("/status")
def status(request: Request) -> dict:
    """Detailed status. Mirrors the fula-ota plugin's `ai/status` BLE
    command shape. Includes the loaded schema set so an operator can spot
    a schema drift via the BLE proxy.
    """
    backend = request.app.state.backend
    schemas = request.app.state.schemas
    return {
        **backend.status_snapshot(),
        "schemas_loaded": sorted(schemas.by_name.keys()),
        "schema_dir": schemas.schema_dir,
    }
