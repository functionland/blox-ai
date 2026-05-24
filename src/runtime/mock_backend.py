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
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import AsyncIterator


# 64 'a' chars — meets the recommended_action.approval_token minLength:64
# constraint. C4 (executor) will replace with real HMAC tokens.
_MOCK_APPROVAL_TOKEN = "a" * 64


@dataclass
class MockBackend:
    """Fake model backend used in dev and when no RKLLM .so is available."""

    name: str = "mock"
    loaded: bool = True
    runbook_version: int = 0  # populated in C6 once the loader wires in

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

        Real RKLLM-backed sequencing (C7) will replace this with token
        streaming + tool-call JSON parsing, but the event shapes stay
        identical.
        """
        sid = session_id or str(uuid.uuid4())
        yield {
            "type": "session_started",
            "session_id": sid,
            "protocol_version": 3,
            "ttl_seconds": 1800,
        }
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
        # bridge injects tool_result here
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
        yield {
            "type": "recommended_action",
            "action_id": "mock-act-1",
            "action_name": "diag.noop",
            "args": {},
            "reasoning": "Mock backend has no real fix to recommend; "
                        "emitting a placeholder for SSE plumbing testing.",
            "confidence": 0.0,
            "tier": 2,
            "approval_token": _MOCK_APPROVAL_TOKEN,
        }
