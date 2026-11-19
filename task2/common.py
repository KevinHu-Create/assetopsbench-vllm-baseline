"""Shared serialization and experiment configuration, without third-party dependencies."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    config = read_json(path)
    for key in ("concurrency", "measured_sessions", "max_tokens", "max_model_len", "max_num_seqs", "tensor_parallel_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["warmup_sessions"]) is not int or config["warmup_sessions"] < 0:
        raise ValueError("warmup_sessions must be a nonnegative integer")
    if not 0 < config["gpu_memory_utilization"] < 1 or config["timeout_seconds"] <= 0:
        raise ValueError("Invalid GPU utilization or timeout")
    if config["cache_mode"] not in {"off", "on"}:
        raise ValueError("cache_mode must be off or on")
    if config["dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("Use an explicit float16 or bfloat16 dtype")
    if config.get("quantization") != "awq" or config["dtype"] != "float16":
        raise ValueError("This shared T4 candidate requires AWQ weights with float16 compute")
    if config.get("attention_backend") != "TRITON_ATTN" or type(config.get("enforce_eager")) is not bool:
        raise ValueError("Pin the attention backend and eager execution mode")
    if len(config["revision"]) != 40:
        raise ValueError("Pin the Hugging Face model revision to a commit SHA")
    return config


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    values = sorted(values)
    n = len(values)
    median = (values[(n - 1) // 2] + values[n // 2]) / 2
    return {"p50": round(median, 3), "p95": round(values[math.ceil(n * .95) - 1], 3),
            "mean": round(sum(values) / n, 3), "n": n}
