"""MockBackend — fake RKLLM-equivalent for dev + tests + amd64 builds.

Returns canned events. C7 replaces this with a real RKLLMBackend wired
to the vendored librkllmrt.so on arm64.

The interface (`generate(...)`) returns an async iterator of event dicts
shaped to match `sse_events.schema.json`. Tests in C1 only exercise
`name` + `loaded`; C2 onwards extends this to a scripted tool-call
sequence.
"""
from __future__ import annotations

from dataclasses import dataclass


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
