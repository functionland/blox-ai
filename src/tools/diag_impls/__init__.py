"""RealDiagExecutor — dispatches a tool name to the right impl.

Drop-in replacement for src.runtime.mock_diag.MockDiagExecutor. Same
async-callable interface so the bridge in src.session.tool_call_loop
doesn't need any changes when swapped.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any


from src.tools.diag_impls.ble_state import diag_ble_state
from src.tools.diag_impls.containers import diag_containers
from src.tools.diag_impls.discovery_state import diag_discovery_state
from src.tools.diag_impls.events import diag_events
from src.tools.diag_impls.fula_go_health import diag_fula_go_health
from src.tools.diag_impls.heartbeat import diag_heartbeat
from src.tools.diag_impls.identity_health import diag_identity_health
from src.tools.diag_impls.image_versions import diag_image_versions
from src.tools.diag_impls.internet import diag_internet
from src.tools.diag_impls.kubo_health import diag_kubo_health
from src.tools.diag_impls.network_interface import diag_network_interface
from src.tools.diag_impls.plugins import diag_plugins
from src.tools.diag_impls.power import diag_power
from src.tools.diag_impls.readiness import diag_readiness
from src.tools.diag_impls.relay import diag_relay
from src.tools.diag_impls.storage import diag_storage
from src.tools.diag_impls.summary import diag_summary
from src.tools.diag_impls.systemd_services import diag_systemd_services
from src.tools.diag_impls.time_ import diag_time
from src.tools.diag_impls.uniondrive import diag_uniondrive
from src.tools.diag_impls.wireguard import diag_wireguard


logger = logging.getLogger("blox-ai.diag")


class UnknownToolError(KeyError):
    """Raised when an unrecognised tool name reaches the executor."""


_DISPATCH: dict[str, Callable[[], dict]] = {
    "diag/internet":          diag_internet,
    "diag/relay":              diag_relay,
    "diag/time":               diag_time,
    "diag/power":              diag_power,
    "diag/storage":            diag_storage,
    "diag/containers":         diag_containers,
    "diag/wireguard":          diag_wireguard,
    "diag/heartbeat":          diag_heartbeat,
    "diag/events":             diag_events,
    "diag/readiness":          diag_readiness,
    "diag/summary":            diag_summary,
    "diag/discovery_state":    diag_discovery_state,
    "diag/systemd_services":   diag_systemd_services,
    "diag/network_interface":  diag_network_interface,
    "diag/uniondrive":         diag_uniondrive,
    "diag/identity_health":    diag_identity_health,
    "diag/kubo_health":        diag_kubo_health,
    "diag/fula_go_health":     diag_fula_go_health,
    "diag/image_versions":     diag_image_versions,
    "diag/ble_state":          diag_ble_state,
    "diag/plugins":            diag_plugins,
}


def known_tools() -> list[str]:
    return list(_DISPATCH.keys())


class RealDiagExecutor:
    """Async-callable executor with the same shape as MockDiagExecutor."""

    name: str = "real"

    async def __call__(self, tool: str, args: dict) -> dict:
        impl = _DISPATCH.get(tool)
        if impl is None:
            raise UnknownToolError(tool)
        # Each impl is synchronous + bounded by its own subprocess timeouts.
        # Run in the default executor so we don't block the event loop;
        # bridge already awaits this coroutine.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, impl)
