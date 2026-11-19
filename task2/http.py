"""Small endpoint controls; credentials stay in memory and out of output files."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


def endpoint(url: str, method: str = "GET", timeout: float = 30) -> bytes:
    headers = {}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    request = Request(url, method=method, headers=headers, data=b"" if method == "POST" else None)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def json_endpoint(url: str) -> dict:
    return json.loads(endpoint(url))


def reset_prefix_cache(base_url: str) -> dict:
    """Require positive vLLM 0.34 confirmation; HTTP 200 alone is insufficient."""
    raw = endpoint(base_url.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache", "POST")
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError("Prefix cache reset was not confirmed; do not start measurement")
    return result
