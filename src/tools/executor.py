"""C4 — action executor with HMAC + whitelist + tier-3 + audit.

The trust boundary. The AI can propose any action_name it wants in a
`recommended_action` event; the executor here enforces:

  1. body validates against execute_action_request.schema.json
  2. approval_token verifies (HMAC + expiry + nonce + action_id binding)
  3. action_name is in the whitelist's tier_2/tier_3 lists
  4. args match argument_constraints (whitelisted values only)
  5. tier-3 actions: security_code from request matches the file at
     /etc/fula/blox-ai/security-code
  6. Serialized execution (asyncio.Lock — one action at a time)
  7. Dispatch:
     - maps_to_core=true → touch /home/pi/commands/.command_<name>
     - maps_to_core=false → subprocess (docker / nsenter)
  8. Audit log line per outcome (executed OR rejected)

The whitelist lives at /etc/fula/action_whitelist.json (NOT
/etc/fula/blox-ai/action_whitelist.json — per advisor catch / api/README
Phase 10 note). Hash stamped on every audit line so a forensic review
can correlate the line with the exact whitelist active at the time.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tools.approval_token import ApprovalTokenSigner, TokenError
from src.tools.audit import append as audit_append, now_iso


logger = logging.getLogger("blox-ai.executor")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXECUTOR_VERSION = "0.1.0"


def _whitelist_path() -> str:
    return os.environ.get(
        "BLOX_AI_WHITELIST_PATH",
        "/etc/fula/action_whitelist.json",
    )


def _security_code_path() -> str:
    return os.environ.get(
        "BLOX_AI_SECURITY_CODE_PATH",
        "/etc/fula/blox-ai/security-code",
    )


def _commands_flag_dir() -> str:
    return os.environ.get(
        "BLOX_AI_COMMANDS_FLAG_DIR",
        "/home/pi/commands",
    )


# Subprocess timeouts per action type
SUBPROCESS_TIMEOUT_DOCKER = 30
SUBPROCESS_TIMEOUT_SYSTEMCTL = 20
SUBPROCESS_TIMEOUT_WG = 10
SUBPROCESS_TIMEOUT_NTP = 10


# ---------------------------------------------------------------------------
# Whitelist loader
# ---------------------------------------------------------------------------

class WhitelistError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedWhitelist:
    raw: dict
    sha256_hex: str
    tier_1_names: frozenset[str]      # may include glob patterns
    tier_2_names: frozenset[str]
    tier_3_names: frozenset[str]
    tier_2_maps_to_core: frozenset[str]
    tier_3_maps_to_core: frozenset[str]
    arg_constraints: dict             # action_name → {arg → allowed list}

    def tier_of(self, action_name: str) -> int | None:
        if action_name in self.tier_3_names:
            return 3
        if action_name in self.tier_2_names:
            return 2
        # tier_1 supports glob ("diag/*"); not enforced at /execute-action
        # because diag/* lives in the GET routes, not this surface.
        return None

    def is_maps_to_core(self, action_name: str) -> bool:
        return action_name in self.tier_2_maps_to_core \
            or action_name in self.tier_3_maps_to_core


def load_whitelist(path: str | None = None) -> LoadedWhitelist:
    """Read + parse + hash the whitelist. Raises WhitelistError on any
    structural problem — the executor's __init__ propagates so the
    container refuses to start with a broken whitelist."""
    p = Path(path or _whitelist_path())
    try:
        raw_bytes = p.read_bytes()
    except OSError as e:
        raise WhitelistError(f"could not read whitelist at {p}: {e}") from e
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as e:
        raise WhitelistError(f"whitelist is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise WhitelistError("whitelist root must be a JSON object")

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

    t1_actions = (raw.get("tier_1_read") or {}).get("actions") or []
    t2 = (raw.get("tier_2_idempotent") or {}).get("actions") or {}
    t3 = (raw.get("tier_3_destructive") or {}).get("actions") or {}
    if not isinstance(t1_actions, list):
        raise WhitelistError("tier_1_read.actions must be a list")
    if not isinstance(t2, dict):
        raise WhitelistError("tier_2_idempotent.actions must be a dict")
    if not isinstance(t3, dict):
        raise WhitelistError("tier_3_destructive.actions must be a dict")

    tier_2_names = frozenset(t2.keys())
    tier_3_names = frozenset(t3.keys())
    tier_2_maps_to_core = frozenset(
        n for n, v in t2.items() if isinstance(v, dict) and v.get("maps_to_core")
    )
    tier_3_maps_to_core = frozenset(
        n for n, v in t3.items() if isinstance(v, dict) and v.get("maps_to_core")
    )

    arg_constraints_raw = raw.get("argument_constraints") or {}
    arg_constraints: dict[str, dict[str, list[str]]] = {}
    for name, constraints in arg_constraints_raw.items():
        if not isinstance(constraints, dict):
            continue
        per_arg: dict[str, list[str]] = {}
        for arg_name, allowed in constraints.items():
            if arg_name.startswith("_"):
                continue  # _notes etc.
            if isinstance(allowed, list) and all(isinstance(v, str) for v in allowed):
                per_arg[arg_name] = list(allowed)
        if per_arg:
            arg_constraints[name] = per_arg

    return LoadedWhitelist(
        raw=raw,
        sha256_hex=sha256_hex,
        tier_1_names=frozenset(t1_actions),
        tier_2_names=tier_2_names,
        tier_3_names=tier_3_names,
        tier_2_maps_to_core=tier_2_maps_to_core,
        tier_3_maps_to_core=tier_3_maps_to_core,
        arg_constraints=arg_constraints,
    )


# ---------------------------------------------------------------------------
# Security code reader (NO caching — per api/README contract)
# ---------------------------------------------------------------------------

def read_security_code(path: str | None = None) -> str | None:
    """Read on EVERY tier-3 request. Returns None if missing/empty."""
    p = path or _security_code_path()
    try:
        code = Path(p).read_text(encoding="utf-8").strip()
        return code if code else None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Wraps the trust boundary. One instance per app lifespan; holds
    the HMAC signer and an asyncio.Lock to serialize execution."""

    def __init__(
        self,
        signer: ApprovalTokenSigner,
        whitelist: LoadedWhitelist,
        audit_path: str | None = None,
    ):
        self.signer = signer
        self.whitelist = whitelist
        self.audit_path = audit_path
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------

    async def execute(
        self,
        action_id: str,
        approval_token: str,
        security_code: str | None,
        action_name: str,
        action_args: dict,
        approver_transport: str = "ble",
    ) -> dict:
        """Top-level entrypoint. Returns a dict shaped for the
        execution_result SSE event AND a parallel dict suitable for
        writing to the audit log.

        action_name + action_args MUST be supplied by the caller (the
        bridge resolves them from the original recommended_action event
        the executor's `approval_token` was issued for — the token is
        bound to action_id, not to action_name).
        """
        request_id = str(uuid.uuid4())
        started = time.monotonic()

        # Common audit-line scaffolding. We mutate `line` as we walk
        # through the validation gates; final line is written via
        # audit_append() exactly once before this method returns.
        line: dict = {
            "ts": now_iso(),
            "request_id": request_id,
            "action_id": action_id,
            "action": action_name,
            "args": action_args,
            "tier": 0,
            "approval_token_valid": False,
            "security_code_required": False,
            "executed": False,
            "rejected_reason": "",
            "approver_transport": approver_transport,
            "duration_ms": 0,
            "executor_version": EXECUTOR_VERSION,
            "whitelist_hash": self.whitelist.sha256_hex,
        }

        try:
            # ------------------------------------------------------------------
            # Whitelist check
            # ------------------------------------------------------------------
            tier = self.whitelist.tier_of(action_name)
            if tier is None:
                return self._reject(line, "action_not_in_whitelist", started)
            line["tier"] = tier
            line["security_code_required"] = (tier == 3)

            # ------------------------------------------------------------------
            # Args constraint check (only applies to non-maps_to_core actions)
            # ------------------------------------------------------------------
            if not self.whitelist.is_maps_to_core(action_name):
                constraint_violation = self._args_violation(action_name, action_args)
                if constraint_violation:
                    return self._reject(
                        line, "args_constraint_violation",
                        started, detail=constraint_violation,
                    )

            # ------------------------------------------------------------------
            # Approval token (HMAC + expiry + nonce + action_id binding)
            # ------------------------------------------------------------------
            try:
                self.signer.verify(approval_token, expected_action_id=action_id)
            except TokenError as te:
                return self._reject(line, te.reason, started)
            line["approval_token_valid"] = True

            # ------------------------------------------------------------------
            # Tier-3 security code (read on EVERY request, no caching)
            # ------------------------------------------------------------------
            if tier == 3:
                file_code = read_security_code()
                if file_code is None:
                    return self._reject(line, "security_code_file_missing", started)
                if not security_code:
                    return self._reject(
                        line, "security_code_required_but_missing", started,
                    )
                if security_code != file_code:
                    line["security_code_valid"] = False
                    return self._reject(line, "security_code_invalid", started)
                line["security_code_valid"] = True

            # ------------------------------------------------------------------
            # Serialize + execute
            # ------------------------------------------------------------------
            if self._lock.locked():
                return self._reject(line, "executor_busy", started)
            async with self._lock:
                result = await self._dispatch(action_name, action_args)

            line["executed"] = True
            # executed=true requires empty rejected_reason + result block
            line["rejected_reason"] = ""
            line["result"] = {
                "success": result["success"],
                "exit_code": result.get("exit_code", 0),
            }
            if result.get("stdout_excerpt"):
                line["stdout_excerpt"] = result["stdout_excerpt"]
            if result.get("stderr_excerpt"):
                line["stderr_excerpt"] = result["stderr_excerpt"]
            line["duration_ms"] = int((time.monotonic() - started) * 1000)

            audit_append(line, path=self.audit_path)
            return {
                "request_id": request_id,
                "audit_line": line,
                "http_status": 200,
                "sse_event": {
                    "type": "execution_result",
                    "action_id": action_id,
                    "success": result["success"],
                    "exit_code": result.get("exit_code", 0),
                    "duration_ms": line["duration_ms"],
                    **({"stdout_excerpt": result["stdout_excerpt"]}
                       if result.get("stdout_excerpt") else {}),
                    **({"stderr_excerpt": result["stderr_excerpt"]}
                       if result.get("stderr_excerpt") else {}),
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("executor internal error")
            line["error"] = str(e)[:1000]
            return self._reject(line, "internal_error", started)

    # ------------------------------------------------------------------
    # Rejection helper
    # ------------------------------------------------------------------

    def _reject(self, line: dict, reason: str, started: float,
                detail: str = "") -> dict:
        # executed=false → MUST have non-empty rejected_reason +
        # MUST NOT have `result` field (audit_log_line.schema.json
        # conditional invariant)
        line["executed"] = False
        line["rejected_reason"] = reason
        line.pop("result", None)
        if detail:
            line["error"] = detail[:1000]
        line["duration_ms"] = int((time.monotonic() - started) * 1000)

        audit_append(line, path=self.audit_path)

        # HTTP status mapping per api/README Phase 10:
        # 400=body_invalid (caller's job before reaching here)
        # 401=approval_token_* (HMAC mismatch / expired / replayed)
        # 403=whitelist / args / security_code
        # 429=executor_busy
        # 500=internal_error
        status_map = {
            "approval_token_invalid":         401,
            "approval_token_expired":         401,
            "approval_token_replayed":        401,
            "action_not_in_whitelist":        403,
            "args_constraint_violation":      403,
            "security_code_required_but_missing": 403,
            "security_code_invalid":          403,
            "security_code_file_missing":     403,
            "executor_busy":                  429,
            "recommendation_not_found":       409,
            "internal_error":                 500,
        }
        return {
            "request_id": line["request_id"],
            "audit_line": line,
            "http_status": status_map.get(reason, 500),
            "sse_event": {
                "type": "error",
                "code": reason.upper(),
                "message": detail or reason,
                "recoverable": reason in ("executor_busy",),
            },
        }

    # ------------------------------------------------------------------
    # Args constraint enforcement
    # ------------------------------------------------------------------

    def _args_violation(self, action_name: str, args: dict) -> str:
        """Return a string detail on violation, else empty string."""
        constraints = self.whitelist.arg_constraints.get(action_name)
        if constraints is None:
            # No constraints defined → accept (action's args spec is "open").
            # Note: for safety, an action with no defined constraints that's
            # nevertheless NOT maps_to_core is suspicious; whitelist authors
            # should always define constraints for subprocess-dispatched
            # actions. Caller already passed the whitelist tier check, so
            # this isn't fail-open against an unknown action.
            return ""
        if not isinstance(args, dict):
            return f"args must be a dict for {action_name}"
        for arg_name, allowed in constraints.items():
            value = args.get(arg_name)
            if value not in allowed:
                return f"{action_name}.{arg_name}={value!r} not in allowed {allowed}"
        return ""

    # ------------------------------------------------------------------
    # Dispatch — flag-file vs subprocess split (advisor catch #5)
    # ------------------------------------------------------------------

    async def _dispatch(self, action_name: str, args: dict) -> dict:
        """Returns {success, exit_code, stdout_excerpt?, stderr_excerpt?}."""
        if self.whitelist.is_maps_to_core(action_name):
            return await self._dispatch_flag_file(action_name)
        # Subprocess-dispatched, arg-bearing actions
        if action_name == "docker.restart":
            return await self._run(
                ["docker", "restart", args["container"]],
                timeout=SUBPROCESS_TIMEOUT_DOCKER,
            )
        if action_name == "systemctl.restart":
            return await self._run(
                ["nsenter", "--target", "1", "--mount", "--uts", "--ipc",
                 "--net", "--pid", "systemctl", "restart", args["unit"]],
                timeout=SUBPROCESS_TIMEOUT_SYSTEMCTL,
            )
        if action_name == "systemctl.reset-failed":
            return await self._run(
                ["nsenter", "--target", "1", "--mount", "--uts", "--ipc",
                 "--net", "--pid", "systemctl", "reset-failed", args["unit"]],
                timeout=SUBPROCESS_TIMEOUT_SYSTEMCTL,
            )
        if action_name == "wireguard.bounce":
            # Two sequential commands; collapse into one shell for atomicity
            return await self._run(
                ["nsenter", "--target", "1", "--mount", "--uts", "--ipc",
                 "--net", "--pid", "bash", "-c",
                 "wg-quick down support; wg-quick up support"],
                timeout=SUBPROCESS_TIMEOUT_WG,
            )
        if action_name == "ntp.resync":
            # Detect daemon then act
            return await self._run(
                ["nsenter", "--target", "1", "--mount", "--uts", "--ipc",
                 "--net", "--pid", "bash", "-c",
                 "systemctl is-active chronyd && chronyc -a makestep "
                 "|| systemctl restart systemd-timesyncd"],
                timeout=SUBPROCESS_TIMEOUT_NTP,
            )
        # Unknown subprocess action — shouldn't reach here because
        # tier_of() already passed
        return {"success": False, "exit_code": -1,
                "stderr_excerpt": f"no dispatcher for {action_name}"}

    async def _dispatch_flag_file(self, action_name: str) -> dict:
        """maps_to_core actions: touch /home/pi/commands/.command_<name>.
        commands.sh on the host (existing) sees the flag-file and runs
        the corresponding core action. No args, no subprocess from here."""
        flag = Path(_commands_flag_dir()) / f".command_{action_name}"
        try:
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.touch()
            return {"success": True, "exit_code": 0,
                    "stdout_excerpt": f"flag-file dispatched: {flag}"}
        except OSError as e:
            return {"success": False, "exit_code": -1,
                    "stderr_excerpt": f"flag-file dispatch failed: {e}"}

    async def _run(self, cmd: list[str], timeout: float) -> dict:
        """Run a subprocess off the event loop. Captures stdout/stderr;
        truncates each excerpt to 2 KB per the schema cap (Gemini HIGH:
        log-injection + disk-exhaustion mitigation)."""
        loop = asyncio.get_event_loop()
        def _exec():
            try:
                cp = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                return {
                    "success": cp.returncode == 0,
                    "exit_code": cp.returncode,
                    "stdout_excerpt": _truncate(cp.stdout, 2048),
                    "stderr_excerpt": _truncate(cp.stderr, 2048),
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "exit_code": -1,
                        "stderr_excerpt": f"timeout after {timeout}s"}
            except FileNotFoundError as e:
                return {"success": False, "exit_code": -1,
                        "stderr_excerpt": f"command not found: {e.filename or cmd[0]}"}
            except OSError as e:
                return {"success": False, "exit_code": -1,
                        "stderr_excerpt": f"OS error: {e}"}
        return await loop.run_in_executor(None, _exec)


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    # Strip CR/LF before truncation (log-injection mitigation)
    safe = s.replace("\r", " ").replace("\n", " ")
    return safe if len(safe) <= n else safe[: n - 1] + "…"
