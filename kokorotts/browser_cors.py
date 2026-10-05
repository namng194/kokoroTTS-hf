"""Cross-origin access for the browser Studio page.

The static HF Space (https, arbitrary origin) must be able to call this
backend (localhost Docker, PRO Space, ZeroGPU) straight from the visitor's
browser. Two browser gates apply:

1. CORS preflight: ``POST /tts/generate`` with ``Content-Type:
   application/json`` triggers an ``OPTIONS`` preflight. Without ACAO
   headers the browser blocks every call.
2. Private Network Access: an ``https`` page calling ``http://localhost``
   sends ``Access-Control-Request-Private-Network: true``; Chrome blocks
   the call unless the preflight answer carries
   ``Access-Control-Allow-Private-Network: true``.

Pure-ASGI implementation (no Starlette-version dependence): attach with
``app.add_middleware(BrowserStudioAccess)`` or wrap any ASGI app.
"""

from __future__ import annotations

from typing import Awaitable


class BrowserStudioAccess:
    """Allow browser Studio pages on any origin (incl. localhost HTTP)."""

    def __init__(self, app, allow_origins: str = "*") -> None:
        self.app = app
        self.allow_origins = allow_origins

    async def __call__(self, scope, receive, send) -> Awaitable[None]:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        origin = headers.get("origin", "")
        allow_origin = origin if self.allow_origins == "*" and origin else self.allow_origins
        wants_pna = headers.get("access-control-request-private-network", "").lower() == "true"

        if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
            # Drain request body, then answer the preflight ourselves.
            while True:
                event = await receive()
                if event["type"] != "http.request" or not event.get("more_body"):
                    break
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", allow_origin.encode("latin-1")),
                        (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                        (
                            b"access-control-allow-headers",
                            (headers.get("access-control-request-headers", "") or "*").encode("latin-1"),
                        ),
                        (b"access-control-max-age", b"600"),
                    ]
                    + ([(b"access-control-allow-private-network", b"true")] if wants_pna else []),
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def tagged_response(message) -> None:
            if message["type"] == "http.response.start":
                hdrs = dict(message.get("headers", []))
                lowered = {k.decode("latin-1").lower() for k in hdrs}
                if "access-control-allow-origin" not in lowered:
                    hdrs[b"access-control-allow-origin"] = allow_origin.encode("latin-1")
                if wants_pna and "access-control-allow-private-network" not in lowered:
                    hdrs[b"access-control-allow-private-network"] = b"true"
                message["headers"] = list(hdrs.items())
            await send(message)

        await self.app(scope, receive, tagged_response)
