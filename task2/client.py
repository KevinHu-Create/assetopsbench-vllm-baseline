"""Streaming HTTP client with explicit transport failures and no automatic retries.

Timing starts before the HTTP request, ends at [DONE], and includes networking.
Role-only and usage-only SSE messages are not first tokens. SSE chunks can contain
multiple tokens, so chunk counts are never reported as generated token counts.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit


class StreamClient:
    def __init__(self, base_url: str, timeout: float = 120, api_key: str = ""):
        u = urlsplit(base_url)
        if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError("Use an http(s) base URL without credentials, query or fragment")
        self.scheme, self.host = u.scheme, u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.path = u.path.rstrip("/") + "/chat/completions"
        self.timeout, self.api_key = timeout, api_key
        self.local, self.lock, self.connections = threading.local(), threading.Lock(), []

    def connection(self):
        if getattr(self.local, "conn", None) is None:
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            self.local.conn = cls(self.host, self.port, timeout=self.timeout)
            with self.lock:
                self.connections.append(self.local.conn)
        return self.local.conn

    def reset(self):
        if getattr(self.local, "conn", None):
            self.local.conn.close()
            self.local.conn = None

    def close(self):
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()

    def generate(self, payload: dict, request_id: str) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "X-Request-Id": request_id}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = json.dumps({**payload, "stream": True, "stream_options": {"include_usage": True}}, ensure_ascii=False).encode()
        started_unix_ms = int(time.time() * 1000)
        t0 = time.perf_counter_ns()
        ttft = first_content = last_token = None
        usage = {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None}
        chunks, content, reasoning = [], [], []
        status, detail, code, finish = "ok", None, None, None
        done = False
        try:
            conn = self.connection()
            conn.request("POST", self.path, body, headers)
            resp = conn.getresponse()
            code = resp.status
            if code != 200:
                # Do not echo server error bodies: they may contain credentials or internals.
                resp.read(1024)
                status = "api_timeout" if code in {408, 504} else "api_error"
                detail = f"HTTP {code}"
            else:
                while True:
                    # Bound the entire request as well as individual socket reads.
                    remaining = self.timeout - (time.perf_counter_ns() - t0) / 1e9
                    if remaining <= 0:
                        raise TimeoutError("Request deadline exceeded")
                    if conn.sock:
                        conn.sock.settimeout(remaining)
                    line = resp.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    elapsed = (time.perf_counter_ns() - t0) / 1e6
                    if event.get("error"):
                        raise ValueError("Server reported a streaming error")
                    if event.get("usage"):
                        raw_usage = event["usage"]
                        for key in ("prompt_tokens", "completion_tokens"):
                            value = raw_usage.get(key)
                            if value is not None and (type(value) is not int or value < 0):
                                raise ValueError("Invalid usage counts")
                            usage[key] = value
                        value = (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        if value is not None and (type(value) is not int or value < 0):
                            raise ValueError("Invalid cached token count")
                        usage["cached_tokens"] = value
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if text or thought:
                            if ttft is None:
                                ttft = elapsed
                            last_token = elapsed
                            chunks.append({"elapsed_ms": round(elapsed, 3), "content": text, "reasoning_content": thought})
                            content.append(text)
                            reasoning.append(thought)
                        if text and first_content is None:
                            first_content = elapsed
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                if not done or finish is None:
                    status, detail = "incomplete_stream", "Missing [DONE] or finish_reason"
                elif ttft is None:
                    status, detail = "empty_response", "No generated text"
                # Close after the final event rather than blocking on EOF on a persistent connection.
                # New connections are explicit and consistently included in every request's timing.
        except (TimeoutError, socket.timeout):
            status, detail = "api_timeout", "Request deadline exceeded"
        except (ValueError, UnicodeError, TypeError, AttributeError):
            status, detail = "stream_error", "Invalid SSE payload"
        except (OSError, http.client.HTTPException):
            status, detail = "api_error", "HTTP connection failed"
        finally:
            self.reset()
        latency = (time.perf_counter_ns() - t0) / 1e6
        tpot = None
        if status == "ok" and usage["completion_tokens"] and usage["completion_tokens"] > 1 and last_token is not None:
            tpot = (last_token - ttft) / (usage["completion_tokens"] - 1)
        return {"raw_response": "".join(content), "reasoning_response": "".join(reasoning), "stream_chunks": chunks,
                "finish_reason": finish, "timing_ms": {"request_start_unix_ms": started_unix_ms,
                    "ttft_ms": round(ttft, 3) if ttft is not None else None,
                    "first_content_ms": round(first_content, 3) if first_content is not None else None,
                    "latency_ms": round(latency, 3), "tpot_ms": round(tpot, 3) if tpot is not None else None},
                "usage": usage, "error": {"status": status, "http_code": code, "message": detail}}
