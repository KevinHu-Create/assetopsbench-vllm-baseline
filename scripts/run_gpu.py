"""Run unprofiled original baselines and a short, separate GPU profiling capture.

Invoke with the GPU virtual environment's Python. A foreground vLLM server child
is stopped in finally. Re-running requires a new empty output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

# Make the repository package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2.common import load_config, read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/gpu-baseline"))
    p.add_argument("--team-profile", type=Path, default=Path("configs/team_environment.json"))
    p.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    p.add_argument("--trial", action="store_true", help="Candidate environment; results cannot pass formal acceptance")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if read_json(args.team_profile).get("status") != "locked":
        raise SystemExit("Lock one common tasks 2/3/4 environment profile before formal runs")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Choose a fresh output directory")
    if not (root / "workloads/original/requests.jsonl").exists():
        raise SystemExit("Run scripts/prepare_original.sh first")
    args.out.mkdir(parents=True, exist_ok=True)
    server_dir = args.out / "server"
    server_dir.mkdir()
    manifest = server_dir / "manifest.json"
    command = [sys.executable, "-m", "task2.serve", "--config", str(args.config), "--out", str(server_dir),
               "--profile-dir", str(args.out / "traces"), "--team-profile", str(args.team_profile)]
    if args.trial:
        command.append("--trial")
    trial_flags = ["--trial"] if args.trial else []
    with (server_dir / "launcher.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 1900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited; inspect server/launcher.log and server/server.log")
                try:
                    if read_json(manifest)["state"] == "ready":
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("vLLM startup timed out")
            for concurrency in (1, 4, 8):
                cfg = {**config, "concurrency": concurrency}
                cfg_path = args.out / f"config-c{concurrency}.json"
                write_json(cfg_path, cfg)
                subprocess.run([sys.executable, "-m", "task2.benchmark", "--config", str(cfg_path),
                    "--requests", "workloads/original/requests.jsonl", "--gt", "workloads/original/gt.json",
                    "--server-manifest", str(manifest), "--out", str(args.out / f"baseline-c{concurrency}"), *trial_flags], cwd=root, check=True)
            profile_cfg = args.out / "config-profile.json"
            write_json(profile_cfg, {**config, "concurrency": 1, "warmup_sessions": 1, "measured_sessions": 2})
            subprocess.run([sys.executable, "-m", "task2.benchmark", "--config", str(profile_cfg),
                "--requests", "workloads/original/requests.jsonl", "--gt", "workloads/original/gt.json",
                "--server-manifest", str(manifest), "--profile", "--out", str(args.out / "profile"), *trial_flags], cwd=root, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    traces = sorted((args.out / "traces").rglob("*.trace.json*"))
    if not traces:
        raise RuntimeError("No GPU profiling trace was produced; inspect vLLM logs")
    for i, path in enumerate(traces):
        subprocess.run([sys.executable, "-m", "task2.trace_summary", str(path), "--out", str(args.out / f"trace-summary-{i}.json")], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "task2.check", str(args.out), *trial_flags], cwd=root, check=True)
    if args.trial:
        print("Real GPU trial complete; team configuration is unconfirmed, so this is not formal acceptance.")


if __name__ == "__main__":
    main()
