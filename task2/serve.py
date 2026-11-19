"""Launch a pinned vLLM service and record the actual process configuration.

Run on the NVIDIA GPU host, in a dedicated environment. This does not provision
cloud resources. The process remains in the foreground; Ctrl-C stops the child.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import load_config, write_json
from .environment import capture
from .http import json_endpoint


def command(config: dict, port: int, profile_dir: Path | None) -> list[str]:
    args = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", config["model"],
            "--revision", config["revision"], "--tokenizer-revision", config["revision"],
            "--dtype", config["dtype"], "--tensor-parallel-size", str(config["tensor_parallel_size"]),
            "--max-model-len", str(config["max_model_len"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--max-num-seqs", str(config["max_num_seqs"]), "--seed", str(config["seed"]),
            "--generation-config", "vllm", "--enable-request-id-headers",
            "--host", "127.0.0.1", "--port", str(port),
            "--enable-prefix-caching" if config["cache_mode"] == "on" else "--no-enable-prefix-caching"]
    if profile_dir:
        args += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(profile_dir.resolve()),
                    "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False})]
    return args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/server-off"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()
    config = load_config(args.config)
    cmd = command(config, args.port, args.profile_dir)
    if args.print_only:
        import shlex
        print(shlex.join(cmd))
        return
    info = capture()
    if not info["formal_gpu_ready"]:
        raise SystemExit("An NVIDIA GPU and installed vLLM are required; see task2.environment")
    if info["packages"]["vllm"] != config["vllm_version"]:
        raise SystemExit("Installed vLLM version does not match the pinned configuration")
    # Refuse an occupied endpoint; never record another process as our service.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit("Port is already in use. Stop the old service before changing cache mode.")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"state": "starting", "config": config, "command": cmd, "environment": info,
                "profile_dir": str(args.profile_dir.resolve()) if args.profile_dir else None,
                "base_url": f"http://127.0.0.1:{args.port}/v1"}
    write_json(args.out / "manifest.json", manifest)
    env = dict(os.environ, VLLM_NO_USAGE_STATS="1")
    with (args.out / "server.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        manifest["pid"] = child.pid
        def stop_child():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            deadline = time.monotonic() + 1800
            while child.poll() is None and time.monotonic() < deadline:
                try:
                    models = json_endpoint(manifest["base_url"] + "/models")
                    version = json_endpoint(manifest["base_url"].removesuffix("/v1") + "/version")
                    if config["model"] in {m["id"] for m in models["data"]}:
                        manifest.update(state="ready", served_models=models, server_version=version)
                        write_json(args.out / "manifest.json", manifest)
                        print(f"Ready: {manifest['base_url']}; manifest: {args.out / 'manifest.json'}", flush=True)
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("vLLM failed to become ready; inspect server.log")
            child.wait()
        except KeyboardInterrupt:
            pass
        finally:
            stop_child()
            manifest.update(state="stopped", exit_code=child.returncode)
            write_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
