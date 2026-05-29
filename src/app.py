"""Blox AI FastAPI app — top-level wiring.

Lifespan:
  1. Load + validate all JSON Schema contracts from /etc/fula/blox-ai/api/
     (set via BLOX_AI_SCHEMA_DIR for dev). Refuse to start if any schema
     fails to load.
  2. Initialise the model backend (MockBackend in dev / when RKLLM .so is
     absent; real RKLLMBackend on arm64 with vendored libs — wired in C7).
  3. Initialise SessionManager (C5).
  4. Initialise RunbookLoader + install SIGHUP handler (C6).
  5. Initialise ApprovalTokenSigner + ActionExecutor (C4 trust boundary).
"""
from __future__ import annotations

import logging
import os
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.runtime.mock_backend import MockBackend
from src.runtime.runbook_loader import RunbookLoader
from src.runtime.tree_dsl import TreeValidationError, load_tree_registry
from src.runtime.tree_runner import TreeRunner
from src.session.manager import SessionManager
from src.tools.approval_token import ApprovalTokenSigner
from src.tools.diag_impls import RealDiagExecutor, known_tools
from src.tools.executor import ActionExecutor, WhitelistError, load_whitelist
from src.schemas import SchemaRegistry
from src.routes import cancel, classify, diag, execute, feedback, health, pending, support, troubleshoot


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

    # C4: HMAC approval-token signer + whitelist-enforced executor.
    # Soft-fail in dev: log + leave executor unset when whitelist is
    # missing (test fixtures stage their own). Production deploy
    # guarantees the bind mount + exits via Docker healthcheck if
    # /execute-action returns 500 executor_not_initialised.
    app.state.approval_signer = ApprovalTokenSigner()
    try:
        whitelist = load_whitelist()
        app.state.action_executor = ActionExecutor(
            signer=app.state.approval_signer,
            whitelist=whitelist,
        )
        logger.info("action_executor wired (whitelist_hash=%s)",
                    whitelist.sha256_hex[:12])
    except WhitelistError as e:
        logger.warning("whitelist load failed (%s); /execute-action will 500", e)
        app.state.action_executor = None

    # C7: try the real RKLLM backend (arm64 + librkllmrt.so + model file
    # present). Falls back to MockBackend cleanly when any of those isn't
    # present — so dev/CI/amd64 builds + lab devices without a model
    # published yet still boot the container without raising.
    from src.runtime.rkllm_runtime import try_load as _try_rkllm
    real_backend = _try_rkllm()
    if real_backend is None:
        # Wire the signer into MockBackend so recommended_action events
        # carry real HMAC tokens that /execute-action can verify.
        app.state.backend = MockBackend(
            action_signer=app.state.approval_signer.sign,
        )
    else:
        app.state.backend = real_backend
    logger.info("backend=%s", app.state.backend.name)

    # C3: real per-tool implementations that read /run/fula-*.state,
    # call kubo API, shell out to docker, etc. Tests can override via
    # fixture (see conftest.py); production uses Real.
    app.state.tool_executor = RealDiagExecutor()
    logger.info("tool_executor=%s", app.state.tool_executor.name)

    # C7-final: wire the executor + signer into RKLLMBackend so it can
    # run diag tools inline + mint real HMAC approval tokens.
    if real_backend is not None and hasattr(real_backend, "wire_runtime_deps"):
        real_backend.wire_runtime_deps(
            tool_executor=app.state.tool_executor,
            action_signer=app.state.approval_signer.sign,
            runbook_loader=None,  # rewired below once runbook_loader exists
        )

    # Phase 1.c: deterministic tree runner. Loads YAML trees from
    # BLOX_AI_TREES_DIR; cross-validates against the diag tool set +
    # the action whitelist loaded above. Soft-fail in dev — if the
    # tree dir is missing or trees fail to load, /troubleshoot/tree
    # returns 503 but other endpoints still work.
    trees_dir = os.environ.get(
        "BLOX_AI_TREES_DIR",
        "/etc/fula/blox-ai/trees",
    )
    app.state.tree_runner = None
    if os.path.isdir(trees_dir):
        try:
            known_action_names = set()
            if app.state.action_executor is not None:
                wl = app.state.action_executor.whitelist
                # LoadedWhitelist exposes tier_2_names / tier_3_names
                # as frozensets per src/tools/executor.py.
                known_action_names |= set(wl.tier_2_names)
                known_action_names |= set(wl.tier_3_names)
            diag_short = {t.removeprefix("diag/") for t in known_tools()}
            registry = load_tree_registry(
                trees_dir,
                known_diag_tools=diag_short,
                known_action_names=known_action_names,
            )
            app.state.tree_runner = TreeRunner(
                trees=registry,
                tool_executor=app.state.tool_executor,
            )
            logger.info(
                "tree_runner wired with %d trees: %s",
                len(registry), sorted(registry.keys()),
            )
        except TreeValidationError as e:
            logger.error("tree registry load failed: %s; /troubleshoot/tree will 503", e)
    else:
        logger.info(
            "trees_dir %s not present; /troubleshoot/tree will 503", trees_dir,
        )

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

    # C7-final: now the runbook loader exists, re-wire RKLLMBackend so
    # its system prompt uses the loaded runbook content (SIGHUP reloads
    # pick up automatically — the backend re-reads via the loader's
    # get_text() on every turn).
    if real_backend is not None and hasattr(real_backend, "wire_runtime_deps"):
        real_backend.wire_runtime_deps(
            tool_executor=app.state.tool_executor,
            action_signer=app.state.approval_signer.sign,
            runbook_loader=app.state.runbook_loader,
        )

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
app.include_router(execute.router)
app.include_router(classify.router)
app.include_router(support.router)
