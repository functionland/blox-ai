"""MockBackend — fake RKLLM-equivalent for dev + tests + amd64 builds.

Returns canned events. C7 replaces this with a real RKLLMBackend wired
to the vendored librkllmrt.so on arm64.

Two surfaces:
  - status_snapshot()  → dict, shape that /status returns (used in C1+).
  - run_troubleshoot(prompt, session_id) → async generator yielding
    canned SSE event dicts (used in C2+). The bridge in
    src.session.tool_call_loop wraps this generator and injects
    tool_result events for each tool_call.

The scripted sequence in C2:
    session_started → thought → tool_call(diag/summary)
                  ↳ (bridge injects tool_result)
    thought → verdict → recommended_action

C4 wires an action_signer so recommended_action emits carry a real
HMAC approval_token (otherwise /execute-action rejects them all at the
token-validation gate). The action chosen (`docker.restart` with
container=ipfs_cluster) is whitelisted in the canonical
action_whitelist.json. (kubo/ipfs_host is deliberately NOT individually
restartable — kubo repairs route through restart_fula — so the mock picks
ipfs_cluster, which reconnects to a still-running kubo.)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional


# Fallback token used when no signer is wired (e.g. C2/C3 tests that
# don't exercise /execute-action). 64 'a' chars satisfies the schema's
# minLength constraint; the executor will reject it on HMAC verify.
_PLACEHOLDER_TOKEN = "a" * 64


ActionSigner = Callable[[str], str]  # action_id → wire-format token


_CLASSIFY_KEYWORDS = {
    "disconnected": (
        "disconnect", "offline", "not reach", "can't reach", "cant reach",
        "cant see", "can't see", "lost", "unreachable",
    ),
    "not-earning": (
        "earn", "reward", "pin", "income",
    ),
    "cannot-join-pool": (
        "pool", "join", "membership", "cannot join",
    ),
}


@dataclass
class MockBackend:
    """Fake model backend used in dev and when no RKLLM .so is available."""

    name: str = "mock"
    loaded: bool = True
    runbook_version: int = 0  # populated in C6 once the loader wires in
    action_signer: Optional[ActionSigner] = field(default=None, repr=False)

    async def classify(self, prompt: str) -> str:
        """Phase 1.d — deterministic keyword fallback for tests/dev.
        Production uses RKLLMBackend.classify; the locked decision is
        LLM-only, no keyword tier in production. MockBackend uses
        keywords because tests must be deterministic — never wired in
        prod by lifespan when the real RKLLM .so is present."""
        p = (prompt or "").lower()
        for scenario, keywords in _CLASSIFY_KEYWORDS.items():
            if any(k in p for k in keywords):
                return scenario
        return "other"

    def _token_for(self, action_id: str) -> str:
        """Mint a real HMAC token via the wired signer, or fall back to
        the placeholder. Returning the placeholder still produces a
        schema-valid SSE event; /execute-action rejects it on HMAC
        verify (which is the right behavior for tests that don't wire
        the executor)."""
        if self.action_signer is None:
            return _PLACEHOLDER_TOKEN
        try:
            return self.action_signer(action_id)
        except Exception:
            return _PLACEHOLDER_TOKEN

    def status_snapshot(self) -> dict:
        """Shape that the /status route returns. Closed; only the documented
        fields. C5 will add active_sessions; C7 will add npu_health."""
        return {
            "model_loaded": self.loaded,
            "model_backend": self.name,
            "runbook_version": self.runbook_version,
            "active_sessions": 0,
            "npu_health": "n/a",
            "last_error": None,
        }

    async def run_troubleshoot(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Scripted SSE event sequence for the mock backend.

        Yields events WITHOUT tool_result — the bridge in
        tool_call_loop.py constructs the tool_result event by calling
        the diag executor with the tool name + args from each tool_call.

        Prompt routing:
          - default: diag/summary → verdict (C2 happy path).
          - "ask " in prompt: emits a user_question; bridge waits for
            /troubleshoot/user-reply (C5 conversational path).

        Real RKLLM-backed sequencing (C7) will replace this with token
        streaming + tool-call JSON parsing, but the event shapes stay
        identical.
        """
        sid = session_id or str(uuid.uuid4())
        yield {
            "type": "session_started",
            "session_id": sid,
            "protocol_version": 4,
            "ttl_seconds": 1800,
        }

        if "ask " in prompt.lower():
            # C5 conversational path: model asks then waits.
            yield {
                "type": "thought",
                "payload": "I need more information before I can diagnose.",
            }
            yield {
                "type": "user_question",
                "question_id": "mock-q-1",
                "payload": {
                    "question": "When did the device first start having trouble?",
                    "expected_response_type": "text",
                },
            }
            # Bridge will inject user_reply_received after /user-reply lands.
            # The mock backend doesn't consume the reply itself (real RKLLM
            # would inject into model context); here we just continue.
            yield {
                "type": "thought",
                "payload": "Thanks. Based on your answer, checking diag/summary.",
            }
            yield {
                "type": "tool_call",
                "call_id": "mock-call-1",
                "payload": {"tool": "diag/summary", "args": {}},
            }
            yield {
                "type": "verdict",
                "payload": {
                    "summary": "Mock backend: nothing actionable after Q&A.",
                    "severity": "green",
                },
            }
            return

        # Default C2 happy path
        yield {
            "type": "thought",
            "payload": f"Mock backend received prompt ({len(prompt)} chars). "
                       f"Will run diag/summary first per runbook.",
        }
        yield {
            "type": "tool_call",
            "call_id": "mock-call-1",
            "payload": {"tool": "diag/summary", "args": {}},
        }
        yield {
            "type": "thought",
            "payload": "diag/summary shows all subsystems green. "
                       "Nothing actionable.",
        }
        yield {
            "type": "verdict",
            "payload": {
                "summary": "Mock backend: device appears healthy.",
                "severity": "green",
                "root_cause": "no_issue_detected",
            },
        }
        action_id = "mock-act-1"
        yield {
            "type": "recommended_action",
            "action_id": action_id,
            "action_name": "docker.restart",
            "args": {"container": "ipfs_cluster"},
            "reasoning": "Mock backend recommendation (whitelisted "
                        "docker.restart with arg-constraint-passing "
                        "container=ipfs_cluster). On a real device + real "
                        "backend, the model would pick this only when "
                        "diag/containers shows ipfs_cluster wedged while "
                        "kubo itself is healthy.",
            "confidence": 0.8,
            "tier": 2,
            "approval_token": self._token_for(action_id),
        }
