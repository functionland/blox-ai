"""Browser access for FxBlox Web (https://blox.fx.land).

Two layers, both mirroring what go-fula's WAP server does on :3500:

1. Starlette's CORSMiddleware for the allow-listed origins (plus localhost /
   127.0.0.1 dev origins), so a page on blox.fx.land can read responses and
   preflight the JSON POSTs / the X-Fula-Support header.
2. OriginGuardMiddleware — a pure-ASGI guard (NOT BaseHTTPMiddleware, which
   interferes with StreamingResponse/SSE) that rejects state-changing
   requests carrying a NON-allow-listed Origin with 403. Browsers always send
   Origin on cross-origin POSTs, so this closes the cross-site form-POST hole
   for a visitor who is on the same LAN as the Blox. Requests without an
   Origin header (mobile app, curl, the on-device BLE proxy) are untouched.

Override the allow-list with BLOX_AI_CORS_ORIGINS="https://a.example,https://b.example".
"""
from __future__ import annotations

import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

DEFAULT_CORS_ORIGINS = [
    "https://blox.fx.land",
    "https://docs.fx.land",  # functionland.github.io project pages are served under this custom domain (staging)
    "https://functionland.github.io",
]
LOCAL_DEV_ORIGIN_RE = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"
_LOCAL_DEV_ORIGIN = re.compile(LOCAL_DEV_ORIGIN_RE)
ALLOW_HEADERS = ["content-type", "x-fula-support"]
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def allowed_origins() -> list[str]:
    raw = os.environ.get("BLOX_AI_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def origin_allowed(origin: str | None) -> bool:
    if not origin:
        return False
    lowered = origin.lower()
    if any(lowered == o.lower() for o in allowed_origins()):
        return True
    return bool(_LOCAL_DEV_ORIGIN.match(origin))


class OriginGuardMiddleware:
    """Reject state-changing requests whose Origin is present but not allow-listed."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method", "GET") not in SAFE_METHODS:
            origin = None
            for name, value in scope.get("headers", []):
                if name == b"origin":
                    origin = value.decode("latin-1")
                    break
            if origin and not origin_allowed(origin):
                response = PlainTextResponse("origin not allowed", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def install_cors(app: FastAPI) -> None:
    # add_middleware inserts outermost-first, so add the guard first and CORS
    # last: CORS (outer) answers preflights and decorates responses; the guard
    # (inner) only sees non-OPTIONS requests.
    app.add_middleware(OriginGuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_origin_regex=LOCAL_DEV_ORIGIN_RE,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=ALLOW_HEADERS,
        expose_headers=["content-type"],
        max_age=600,
    )
