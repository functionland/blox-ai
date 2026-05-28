"""Phase 1.d — POST /troubleshoot/classify.

Single endpoint that maps a free-text user prompt to one of the
known scenario_ids OR 'other'. The app pipeline:

  user types free text
        │
        ▼
  POST /troubleshoot/classify {prompt}
        │
        ▼
  {scenario_id: 'disconnected' | 'not-earning' | 'cannot-join-pool' | 'other'}
        │
   ┌────┴────┐
   │ known   │ → POST /troubleshoot/tree {scenario_id} (deterministic)
   │ 'other' │ → POST /troubleshoot {prompt}            (LLM fallback)
   └─────────┘

Locked decision (2026-05-28): LLM-only classifier; predefined
quick-start buttons skip the classifier entirely. Falls through to
'other' on any error so the app's fallback path always has somewhere
to go.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


router = APIRouter()
logger = logging.getLogger("blox-ai.classify")


_ALLOWED_SCENARIOS = frozenset({
    "disconnected", "not-earning", "cannot-join-pool", "other",
})


class ClassifyRequest(BaseModel):
    model_config = {"extra": "forbid"}
    prompt: str = Field(min_length=1, max_length=10_000)


@router.post("/troubleshoot/classify")
async def classify(req: ClassifyRequest, request: Request) -> Response:
    backend = request.app.state.backend
    classify_fn = getattr(backend, "classify", None)
    if classify_fn is None or not callable(classify_fn):
        # Backend doesn't expose classify — graceful degradation.
        # The app's fallback for 'other' is the existing AI mode, so
        # this preserves user-facing behaviour.
        return JSONResponse(
            status_code=200,
            content={"scenario_id": "other", "reason": "classifier_unavailable"},
        )
    try:
        scenario_id = await classify_fn(req.prompt)
    except Exception:
        logger.exception("classify backend raised; defaulting to 'other'")
        scenario_id = "other"
    # Defensive guardrail — the backend MAY return something outside
    # the allowed set. Normalize to 'other' so the app never has to
    # validate.
    if scenario_id not in _ALLOWED_SCENARIOS:
        logger.warning(
            "classify backend returned non-allowed value %r; mapping to 'other'",
            scenario_id,
        )
        scenario_id = "other"
    return JSONResponse(status_code=200, content={"scenario_id": scenario_id})
