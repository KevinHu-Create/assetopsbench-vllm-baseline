"""Verify cache control cannot reset active requests or accept remote callers."""
from types import SimpleNamespace
import json
import unittest
from unittest.mock import AsyncMock, patch

from task2.cache_control import CacheResetMiddleware


class TestCacheControl(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, **changes):
        engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=True))
        downstream = AsyncMock()
        middleware = CacheResetMiddleware(downstream)
        scope = {"type": "http", "path": "/reset_prefix_cache", "method": "POST",
                 "client": ("127.0.0.1", 3210), "headers": [], "query_string": b"",
                 "app": SimpleNamespace(state=SimpleNamespace(engine_client=engine))}
        scope.update(changes)
        sent = []
        async def send(event):
            sent.append(event)
        await middleware(scope, AsyncMock(), send)
        return engine, downstream, sent

    async def test_only_idle_local_reset_reaches_engine(self):
        with patch.dict("os.environ", {"VLLM_API_KEY": ""}):
            engine, _, sent = await self.invoke()
        engine.reset_prefix_cache.assert_awaited_once_with(False, False)
        self.assertEqual(sent[0]["status"], 200)
        self.assertTrue(json.loads(sent[1]["body"])["success"])

    async def test_remote_wrong_method_query_and_missing_key_rejected(self):
        for changes, code in (({"client": ("203.0.113.1", 1)}, 403), ({"method": "GET"}, 405),
                              ({"query_string": b"reset_running_requests=true"}, 400)):
            with self.subTest(changes=changes), patch.dict("os.environ", {"VLLM_API_KEY": ""}):
                engine, _, sent = await self.invoke(**changes)
                self.assertEqual(sent[0]["status"], code)
                engine.reset_prefix_cache.assert_not_awaited()
        with patch.dict("os.environ", {"VLLM_API_KEY": "fixture-secret"}):
            engine, _, sent = await self.invoke()
            self.assertEqual(sent[0]["status"], 401)
            engine.reset_prefix_cache.assert_not_awaited()
            engine, _, sent = await self.invoke(headers=[(b"authorization", b"Bearer fixture-secret")])
            self.assertEqual(sent[0]["status"], 200)

    async def test_other_routes_are_untouched(self):
        engine, downstream, sent = await self.invoke(path="/v1/chat/completions")
        downstream.assert_awaited_once()
        engine.reset_prefix_cache.assert_not_awaited()
        self.assertEqual(sent, [])
