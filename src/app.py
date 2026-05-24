"""Blox AI FastAPI app — top-level wiring.

Lifespan:
  1. Load + validate all JSON Schema contracts from /etc/fula/blox-ai/api/
     (set via BLOX_AI_SCHEMA_DIR for dev). Refuse to start if any schema
     fails to load.
  2. Initialise the model backend (MockBackend in dev / when RKLLM .so is
     absent; real RKLLMBackend on arm64 with vendored libs — wired in C7).
  3. Initialise SessionManager (C5).
  4. Initialise RunbookLoader + install SIGHUP handler (C6).
"""
from __future__ import annotations

import logging
import os
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.runtime.mock_backend import MockBackend
from src.runtime.runbook_loader import RunbookLoader
from src.session.manager import SessionManager
from src.tools.diag_impls import RealDiagExecutor
from src.schemas import SchemaRegistry
from src.routes import cancel, diag, feedback, health, pending, troubleshoot


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

    # C7: try the real RKLLM backend (arm64 + librkllmrt.so + model file
    # present). Falls back to MockBackend cleanly when any of those isn't
    # present — so dev/CI/amd64 builds + lab devices without a model
    # published yet still boot the container without raising.
    from src.runtime.rkllm_runtime import try_load as _try_rkllm
    real_backend = _try_rkllm()
    app.state.backend = real_backend if real_backend is not None else MockBackend()
    logger.info("backend=%s", app.state.backend.name)

    # C3: real per-tool implementations that read /run/fula-*.state,
    # call kubo API, shell out to docker, etc. Tests can override via
    # fixture (see conftest.py); production uses Real.
    app.state.tool_executor = RealDiagExecutor()
    logger.info("tool_executor=%s", app.state.tool_executor.name)

    # C5: in-memory session registry for /troubleshoot conversations.
    # 30-min sliding TTL, 50-session cap, LRU eviction. Lost on container
    # restart by design (matches HMAC approval-secret rotation).
    app.state.session_manager = SessionManager()
    logger.info("session_manager initialized (ttl=%ds cap=%d)",
                app.state.session_manager.ttl_sec,
                app.state.session_manager.max_sessions)

    # C6: runbook loader + SIGHUP handler for fast iteration.
    runbook_path = os.environ.get(
        "BLOX_AI_RUNBOOK_PATH",
        "/usr/bin/fula/ai/runbook.md",
    )
    events_log_path = os.environ.get(
        "BLOX_AI_EVENTS_LOG_PATH",
        "/var/log/fula/events.jsonl",
    )
    app.state.runbook_loader = RunbookLoader(
        path=runbook_path,
        events_log_path=events_log_path,
    )
    app.state.runbook_loader.load_initial()

    def _on_sighup(signum, frame):  # noqa: ARG001
        logger.info("SIGHUP received; reloading runbook")
        result = app.state.runbook_loader.reload()
        logger.info("runbook reload outcome=%s", result.get("outcome"))

    # SIGHUP only exists on POSIX. Skip on Windows dev + when running
    # inside TestClient's worker thread (signal.signal must be the main
    # thread).
    try:
        signal.signal(signal.SIGHUP, _on_sighup)
        logger.info("SIGHUP handler installed (runbook reload)")
    except (AttributeError, ValueError) as e:
        # AttributeError: SIGHUP not defined (Windows).
        # ValueError: not the main thread (TestClient lifespan).
        logger.info("SIGHUP handler not installed: %s", e)

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
app.include_router(troubleshoot.router)
app.include_router(diag.router)
app.include_router(feedback.router)
app.include_router(pending.router)
app.include_router(cancel.router)
