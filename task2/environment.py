"""Capture reproducibility metadata, without reading credentials or cloud account data."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import write_json


def capture() -> dict:
    packages = {}
    for name in ("vllm", "torch", "transformers", "tokenizers", "flashinfer-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu, gpu_error = [], None
    if shutil.which("nvidia-smi"):
        run = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if run.returncode == 0:
            gpu = [line.strip() for line in run.stdout.splitlines() if line.strip()]
        else:
            gpu_error = "nvidia-smi failed"
    else:
        gpu_error = "nvidia-smi unavailable; NVIDIA GPU not established"
    freeze = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
                            capture_output=True, text=True, timeout=30)
    # pip list omits installation URLs and embedded credentials that pip freeze can expose.
    import json
    try:
        inventory = json.loads(freeze.stdout) if freeze.returncode == 0 else []
    except ValueError:
        inventory = []
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "machine": platform.machine(), "python": platform.python_version(), "packages": packages,
            "nvidia_gpus": gpu, "gpu_error": gpu_error, "package_inventory": inventory,
            "formal_gpu_ready": bool(gpu and packages["vllm"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/environment.json"))
    p.add_argument("--require-gpu", action="store_true")
    args = p.parse_args()
    info = capture()
    write_json(args.out, info)
    print(f"GPU ready: {info['formal_gpu_ready']}; saved {args.out}")
    if args.require_gpu and not info["formal_gpu_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
