"""Measure original/fixed workloads with separately labelled warmup and profile runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

from .client import StreamClient
from .common import digest, load_config, quantiles, read_json, write_json
from .http import endpoint, json_endpoint

SERVER_KEYS = ("model", "revision", "vllm_version", "dtype", "tensor_parallel_size", "max_model_len",
               "gpu_memory_utilization", "max_num_seqs", "seed", "cache_mode")


def summarize(rows: list[dict], wall_ms: float) -> dict:
    success = [r for r in rows if r["error"]["status"] == "ok"]
    seconds = wall_ms / 1000
    def tokens(key):
        values = [r["usage"].get(key) for r in success]
        return sum(values) if values and all(v is not None for v in values) else None
    prompt, output, cached = (tokens(k) for k in ("prompt_tokens", "completion_tokens", "cached_tokens"))
    latencies = {key: quantiles([r["timing_ms"][key] for r in success if r["timing_ms"].get(key) is not None])
                 for key in ("ttft_ms", "latency_ms", "tpot_ms")}
    scored = [r for r in success if r.get("task_correct") is not None]
    return {"total_requests": len(rows), "ok_requests": len(success), "failed_requests": len(rows) - len(success),
            "failure_rate": (len(rows) - len(success)) / len(rows) if rows else None,
            "failure_counts": dict(Counter(r["error"]["status"] for r in rows if r["error"]["status"] != "ok")),
            "wall_ms": round(wall_ms, 3), **latencies,
            "successful_rps": len(success) / seconds if seconds else None,
            "output_tokens_per_second": output / seconds if output is not None and seconds else None,
            "total_tokens_per_second": (prompt + output) / seconds if prompt is not None and output is not None and seconds else None,
            "usage_totals": {"prompt_tokens": prompt, "completion_tokens": output, "cached_tokens": cached},
            "usage_coverage": {key: sum(r["usage"].get(key) is not None for r in success) for key in ("prompt_tokens", "completion_tokens", "cached_tokens")},
            "cache_hit_rate": cached / prompt if cached is not None and prompt else None,
            "format_failures": sum(r.get("parse_ok") is False for r in success),
            "truncated_responses": sum(r.get("finish_reason") == "length" for r in rows),
            "task_accuracy": sum(r["task_correct"] for r in scored) / len(scored) if scored else None,
            "task_scored_requests": len(scored)}


def load_sessions(path: Path) -> list[list[dict]]:
    grouped = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise ValueError("Each request needs nonempty messages")
            grouped[row["session_id"]].append(row)
    if not grouped:
        raise ValueError("Empty workload")
    sessions = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r.get("seq_idx", r.get("replay_seq", 0)))
        seqs = [r.get("seq_idx", r.get("replay_seq", 0)) for r in rows]
        if len(seqs) != len(set(seqs)):
            raise ValueError("Duplicate seq_idx in a session")
        sessions.append(rows)
    return sessions


def validate_server(config: dict, manifest: dict, base_url: str) -> None:
    if manifest.get("state") != "ready" or not manifest.get("environment", {}).get("formal_gpu_ready"):
        raise ValueError("A ready GPU server manifest is required for a formal run")
    for key in SERVER_KEYS:
        if config[key] != manifest["config"][key]:
            raise ValueError(f"Server/config mismatch: {key}")
    version = json_endpoint(base_url.removesuffix("/v1") + "/version")
    if version.get("version") != config["vllm_version"]:
        raise ValueError("Live vLLM version differs from pinned version")
    models = json_endpoint(base_url + "/models")
    if config["model"] not in {row["id"] for row in models["data"]}:
        raise ValueError("Model is not served by the live endpoint")


def run(requests: Path, config: dict, out: Path, base_url: str, server_manifest: Path | None,
        gt_path: Path | None = None, profile: bool = False, smoke: bool = False) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting evidence")
    sessions = load_sessions(requests)
    source_manifest_path = requests.parent / "manifest.json"
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else None
    if source_manifest and source_manifest.get("requests_sha256") != digest(requests.read_bytes()):
        raise ValueError("Prepared workload hash mismatch")
    manifest = read_json(server_manifest) if server_manifest else None
    if not smoke:
        if manifest is None:
            raise ValueError("--server-manifest is mandatory for GPU measurements")
        validate_server(config, manifest, base_url)
    if profile and (smoke or not manifest or not manifest.get("profile_dir")):
        raise ValueError("Profiling requires a real GPU server launched with --profile-dir")
    for session in sessions:
        for row in session:
            original = row.get("inference_config") or {}
            for key in ("model", "temperature", "max_tokens", "seed"):
                if key in original and original[key] != config[key]:
                    raise ValueError(f"Fixed request configuration differs: {key}; update the run config")
    gt = read_json(gt_path) if gt_path else {}
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    meta = {"schema_version": "3.0", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "http_smoke" if smoke else "vllm", "measurement_kind": "smoke" if smoke else "profile" if profile else "gpu_baseline",
            "verified_gpu_run": not smoke, "config": config, "server_manifest": manifest,
            "workload_sha256": digest(requests.read_bytes()), "source_manifest": source_manifest,
            "timing_definition": "Client monotonic clock, before HTTP send to final SSE event; no retries; one connection per request.",
            "task1_source_commit": "98ee1d0a83f2ddab3449aac204da7bd4b4f64371"}
    write_json(out / "run_config.json", meta)
    (out / "requests.jsonl").write_bytes(requests.read_bytes())
    client = StreamClient(base_url, config["timeout_seconds"], os.environ.get("VLLM_API_KEY", ""))
    profiling_started = False
    records = []
    def session_job(index: int, phase: str) -> list[dict]:
        session = sessions[index % len(sessions)]
        result_rows = []
        for seq, row in enumerate(session):
            request_id = f"{run_id}-{phase}-{index}-{seq}"
            payload = {"model": config["model"], "messages": row["messages"],
                       "temperature": config["temperature"], "max_tokens": config["max_tokens"], "seed": config["seed"]}
            result = client.generate(payload, request_id)
            text = result["raw_response"].strip()
            fmt = row.get("response_format")
            parse_ok = None
            action = None
            if fmt == "integer":
                parse_ok = re.fullmatch(r"-?\d+", text) is not None
            elif fmt == "action_json" or row.get("inference_config"):
                from framework.client import parse_action_json
                action, parse_error = parse_action_json(text)
                parse_ok = parse_error is None
            scenario = row.get("scenario_id", "unknown")
            result.update(schema_version="3.0", run_id=run_id, scenario_id=scenario, phase=phase,
                          session_id=f"{row['session_id']}-copy-{index}", seq_idx=seq, turn_idx=row.get("turn_idx", 0),
                          step_idx=row.get("step_idx", 0), request_id=request_id,
                          prev_request_id=f"{run_id}-{phase}-{index}-{seq-1}" if seq else None,
                          llm_request_messages=row["messages"], llm_request_payload={**payload, "stream": True, "stream_options": {"include_usage": True}},
                          cache_mode=config["cache_mode"], parse_ok=parse_ok, action=action,
                          task_correct=(text == str(gt[scenario])) if scenario in gt and result["error"]["status"] == "ok" else None)
            result_rows.append(result)
        return result_rows
    try:
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            warmup = [row for rows in pool.map(lambda i: session_job(i, "warmup"), range(config["warmup_sessions"])) for row in rows]
            with (out / "warmup_log.jsonl").open("w", encoding="utf-8") as f:
                for row in warmup:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if any(r["error"]["status"] != "ok" for r in warmup):
                raise RuntimeError("Warmup failed; inspect warmup_log.jsonl before measuring")
            if profile:
                endpoint(base_url.removesuffix("/v1") + "/start_profile", "POST")
                profiling_started = True
            # Optional metrics are collected outside the measured wall-clock interval.
            for suffix in ("before",):
                try:
                    (out / f"metrics_{suffix}.prom").write_bytes(endpoint(base_url.removesuffix("/v1") + "/metrics"))
                except OSError:
                    pass
            t0 = time.perf_counter_ns()
            with (out / "run_log.jsonl").open("w", encoding="utf-8") as log:
                for rows in pool.map(lambda i: session_job(i, "profile" if profile else "measured"), range(config["measured_sessions"])):
                    for row in rows:
                        records.append(row)
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            try:
                (out / "metrics_after.prom").write_bytes(endpoint(base_url.removesuffix("/v1") + "/metrics"))
            except OSError:
                pass
    finally:
        client.close()
        if profiling_started:
            try:
                endpoint(base_url.removesuffix("/v1") + "/stop_profile", "POST", timeout=1800)
                write_json(out / "profile_control.json", {"stopped": True, "server_trace_dir": manifest["profile_dir"]})
            except Exception:
                write_json(out / "profile_control.json", {"stopped": False, "error": "stop_profile failed; check server logs"})
                raise
    report = {**meta, "aggregate": summarize(records, wall_ms)}
    write_json(out / "run_summary.json", report)
    aggregate = report["aggregate"]
    import csv
    fields = {"kind": meta["measurement_kind"], "model": config["model"], "dtype": config["dtype"], "cache": config["cache_mode"],
              "concurrency": config["concurrency"], "requests": aggregate["total_requests"], "failures": aggregate["failed_requests"],
              "ttft_p50_ms": aggregate["ttft_ms"]["p50"], "ttft_p95_ms": aggregate["ttft_ms"]["p95"],
              "rps": aggregate["successful_rps"], "output_tokens_s": aggregate["output_tokens_per_second"],
              "format_failures": aggregate["format_failures"], "task_accuracy": aggregate["task_accuracy"]}
    with (out / "baseline.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader(); writer.writerow(fields)
    (out / "baseline.md").write_text("| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |\n| " + " | ".join("N/A" if v is None else str(v) for v in fields.values()) + " |\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--server-manifest", type=Path)
    p.add_argument("--gt", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Label HTTP fixture tests as non-GPU smoke evidence")
    args = p.parse_args()
    result = run(args.requests, load_config(args.config), args.out, args.base_url.rstrip("/"), args.server_manifest, args.gt, args.profile, args.smoke)
    print(json.dumps(result["aggregate"], indent=2))
    if result["aggregate"]["failed_requests"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
