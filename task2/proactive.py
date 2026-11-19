"""Task 3/4 entry: validate the common server before invoking the task-one runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .benchmark import validate_server
from .common import load_config, read_json, write_json
from .client import StreamClient
from .http import reset_prefix_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    p.add_argument("--server-manifest", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--scenarios", type=Path, nargs="+", required=True)
    p.add_argument("--max-steps-per-tick", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.server_manifest)
    validate_server(cfg, manifest, args.base_url.rstrip("/"))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    client = StreamClient(args.base_url, cfg["timeout_seconds"], os.environ.get("VLLM_API_KEY", ""))
    warmup = []
    try:
        for i in range(cfg["warmup_sessions"]):
            warmup.append(client.generate({"model": cfg["model"], "messages": [{"role": "user", "content": 'Return only JSON: {"ready":true}'}],
                "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"], "seed": cfg["seed"]}, f"proactive-warmup-{i}"))
    finally:
        client.close()
    write_json(args.out / "proactive_warmup.json", warmup)
    if any(row["error"]["status"] != "ok" for row in warmup):
        raise RuntimeError("Proactive warmup failed; formal replay was not started")
    reset = reset_prefix_cache(args.base_url)
    write_json(args.out / "cache_reset.json", {"after_warmup": True, "before_timing": True, **reset})
    from framework.run import run_replay
    log, summary_path = run_replay(args.scenarios, args.out, backend="vllm", cache_mode=cfg["cache_mode"],
        base_url=args.base_url, model=cfg["model"], api_key=os.environ.get("VLLM_API_KEY", ""),
        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], seed=cfg["seed"],
        timeout=cfg["timeout_seconds"], max_steps_per_tick=args.max_steps_per_tick, concurrency=cfg["concurrency"],
        gpu_tag=manifest["team_environment"]["gpu"]["name"], dtype=cfg["dtype"])
    identity = {"team_environment_id": manifest["team_environment_id"], "team_environment": manifest["team_environment"]}
    summary = read_json(summary_path)
    summary.update(identity)
    write_json(summary_path, summary)
    write_json(args.out / "shared_environment.json", {**identity, "config": cfg, "server_manifest": manifest,
        "timing_protocol": "framework live Keep-Alive timings; use the common task2.benchmark replay entry for cross-workload system comparisons"})
    temp = log.with_suffix(".annotated.tmp")
    with log.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            row["team_environment_id"] = identity["team_environment_id"]
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(log)
    print(f"Shared environment: {identity['team_environment_id']}; logs: {args.out}")


if __name__ == "__main__":
    main()
