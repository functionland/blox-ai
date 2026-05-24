"""Blox AI FastAPI app — top-level wiring.

Lifespan:
  1. Load + validate all JSON Schema contracts from /etc/fula/blox-ai/api/
     (set via BLOX_AI_SCHEMA_DIR for dev). Refuse to start if any schema
     fails to load.
  2. Initialise the model backend (MockBackend in dev / when RKLLM .so is
     absent; real RKLLMBackend on arm64 with vendored libs — wired in C7).
  3. Register signal handlers (SIGHUP for runbook reload — wired in C6).

Routes registered in C1: /health, /status. Other routers wire in C2-C6.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.runtime.mock_backend import MockBackend
from src.schemas import SchemaRegistry
from src.routes import health


logger = logging.getLogger("blox-ai")
logging.basicConfig(
    level=os.environ.get("BLOX_AI_LOG_LEVEL", "INFO"),
    format="%(asctime)s blox-ai %(levelname)s %(message)s",
)


DEFAULT_SCHEMA_DIR = "/etc/fula/blox-ai/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown. Fail fast on any contract-load problem."""
    schema_dir = os.environ.get("BLOX_AI_SCHEMA_DIR", DEFAULT_SCHEMA_DIR)
    logger.info("loading schemas from %s", schema_dir)
    try:
        app.state.schemas = SchemaRegistry.load(schema_dir)
    except Exception as e:
        logger.error("schema load failed: %s", e)
        raise
    logger.info("loaded %d schemas", len(app.state.schemas))

    # C7 will swap MockBackend for the real RKLLMBackend on arm64.
    app.state.backend = MockBackend()
    logger.info("backend=%s", app.state.backend.name)

    yield

    logger.info("shutdown")


app = FastAPI(
    title="Blox AI",
    description=(
        "On-device AI troubleshooting backend for Fula Blox edge devices "
        "(RK3588 + RKLLM). Implements the contracts defined in "
        "https://github.com/functionland/fula-ota/tree/blox-ai/docker/"
        "fxsupport/linux/plugins/blox-ai/api"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
