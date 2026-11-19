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
