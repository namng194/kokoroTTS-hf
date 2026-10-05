"""Browser cross-origin access: preflight + Private Network Access headers.

Pure-ASGI test (no fastapi/torch needed): drives BrowserStudioAccess with
a stub app and raw scope/receive/send.
"""

import asyncio
import unittest

from kokorotts.browser_cors import BrowserStudioAccess


async def stub_app(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


def run(scope, body=b""):
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(BrowserStudioAccess(stub_app)(scope, receive, send))
    return sent


def headers_of(sent):
    out = {}
    for message in sent:
        if message["type"] == "http.response.start":
            for k, v in message["headers"]:
                out[k.decode("latin-1").lower()] = v.decode("latin-1")
    return out


def scope_for(method, extra=None):
    headers = [(b"origin", b"https://nam194-kokorotts-hf.static.hf.space")]
    for k, v in (extra or {}).items():
        headers.append((k.encode("latin-1"), v.encode("latin-1")))
    return {
        "type": "http",
        "method": method,
        "path": "/tts/generate",
        "headers": headers,
    }


class TestBrowserStudioAccess(unittest.TestCase):
    def test_preflight_answers_with_cors_headers(self):
        sent = run(
            scope_for(
                "OPTIONS",
                {
                    "access-control-request-method": "POST",
                    "access-control-request-headers": "content-type",
                },
            )
        )
        status = [m for m in sent if m["type"] == "http.response.start"][0]["status"]
        self.assertEqual(status, 204)
        headers = headers_of(sent)
        self.assertEqual(
            headers["access-control-allow-origin"],
            "https://nam194-kokorotts-hf.static.hf.space",
        )
        self.assertIn("content-type", headers["access-control-allow-headers"])

    def test_preflight_grants_private_network_when_requested(self):
        sent = run(
            scope_for(
                "OPTIONS",
                {
                    "access-control-request-method": "POST",
                    "access-control-request-headers": "content-type",
                    "access-control-request-private-network": "true",
                },
            )
        )
        headers = headers_of(sent)
        self.assertEqual(headers["access-control-allow-private-network"], "true")

    def test_real_response_gets_cors_and_pna_headers(self):
        sent = run(scope_for("POST", {"access-control-request-private-network": "true"}))
        headers = headers_of(sent)
        self.assertEqual(
            headers["access-control-allow-origin"],
            "https://nam194-kokorotts-hf.static.hf.space",
        )
        self.assertEqual(headers["access-control-allow-private-network"], "true")

    def test_non_http_scope_passes_through(self):
        sent = run({"type": "lifespan", "headers": []})
        self.assertTrue(any(m["type"] == "http.response.start" for m in sent))


if __name__ == "__main__":
    unittest.main()
