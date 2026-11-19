"""Expose only prefix-cache reset on the loopback-only experiment server.

vLLM 0.34 puts reset under its broad development API. This ASGI middleware
uses the same EngineClient call without enabling unrelated development RPCs.
The server launcher binds 127.0.0.1. Optional VLLM_API_KEY is still required.
"""
from __future__ import annotations

import hmac
import json
import os


class CacheResetMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/reset_prefix_cache":
            return await self.app(scope, receive, send)
        status, result = 200, {"success": False}
        client = scope.get("client") or ("", 0)
        token = os.environ.get("VLLM_API_KEY", "")
        authorization = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin1")
        if client[0] not in {"127.0.0.1", "::1"}:
            status = 403
        elif token and not hmac.compare_digest(authorization, "Bearer " + token):
            status = 401
        elif scope.get("method") != "POST":
            status = 405
        elif scope.get("query_string", b""):
            # Do not allow aborting running requests or external cache controls.
            status = 400
        else:
            success = await scope["app"].state.engine_client.reset_prefix_cache(False, False)
            result["success"] = bool(success)
        body = json.dumps(result).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
