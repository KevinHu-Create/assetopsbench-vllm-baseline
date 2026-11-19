"""FAKE local SSE service for transport verification. This performs no inference."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


@contextmanager
def fixture_server():
    captured = []
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == "/metrics":
                body = b"# Fake fixture, not GPU metrics\nfixture_requests 1\n"
                self.send_response(200)
            else:
                body = b'{}'
                self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                captured.append({"payload": payload, "request_id": self.headers.get("X-Request-Id")})
            text = payload["messages"][-1]["content"]
            if text == "http_error":
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            def event(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()
            event({"choices": [{"delta": {"role": "assistant"}}]})
            if text == "malformed":
                self.wfile.write(b'data: {broken\n\n')
                return
            # This fixture delay verifies that role-only events do not become TTFT.
            time.sleep(.015)
            if text == "reasoning":
                event({"choices": [{"delta": {"reasoning_content": "fixture reasoning"}}]})
                time.sleep(.01)
            event({"choices": [{"delta": {"content": "2"}}]})
            if text == "incomplete":
                return
            event({"choices": [{"delta": {}, "finish_reason": "length" if text == "length" else "stop"}]})
            event({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 3 if text == "reasoning" else 1,
                   **({"prompt_tokens_details": {"cached_tokens": 0}} if text != "no_cache_usage" else {})}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
