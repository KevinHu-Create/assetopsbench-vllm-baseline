"""Validate the one shared hardware/software profile for tasks 2, 3 and 4.

A draft is intentionally unusable for formal runs. Hardware is checked against
captured nvidia-smi data, rather than a hand-written gpu_tag.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import digest, load_config, read_json, write_json

INFERENCE_KEYS = ("model", "revision", "vllm_version", "dtype", "tensor_parallel_size", "max_model_len",
                  "gpu_memory_utilization", "max_num_seqs", "seed", "temperature", "max_tokens",
                  "quantization", "attention_backend", "enforce_eager")


def profile_id(profile: dict) -> str:
    return digest(json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def validate(profile: dict, config: dict, environment: dict, *, allow_trial: bool = False) -> str:
    if profile.get("status") != "locked":
        raise ValueError("Tasks 2/3/4 shared environment is still a draft; select and lock it before formal GPU runs")
    if profile.get("scope") == "candidate" and not allow_trial:
        raise ValueError("Candidate environment is for explicit trials only; team configuration is not confirmed")
    expected = profile["gpu"]
    if not expected.get("name") or not isinstance(expected.get("memory_mib"), int) or not expected.get("count"):
        raise ValueError("Lock the GPU's exact name, memory_mib and count")
    observed = []
    for row in csv.reader(environment.get("nvidia_gpus", [])):
        if len(row) != 4:
            raise ValueError("GPU inventory must contain name, memory.total, driver_version, compute_cap")
        observed.append({"name": row[0].strip(), "memory_mib": int(row[1].strip()),
                         "driver_version": row[2].strip(), "compute_cap": row[3].strip()})
    if len(observed) != expected["count"]:
        raise ValueError("Shared GPU count mismatch")
    software = profile["software"]
    for gpu in observed:
        if gpu["name"] != expected["name"] or gpu["memory_mib"] != expected["memory_mib"]:
            raise ValueError("Shared GPU name/VRAM mismatch; switching GPU requires one new profile for all three tasks")
        if not software.get("driver_version") or gpu["driver_version"] != software["driver_version"]:
            raise ValueError("Shared GPU driver mismatch")
    for package in ("vllm", "torch"):
        if not software.get(package) or environment.get("packages", {}).get(package) != software[package]:
            raise ValueError(f"Shared {package} version mismatch")
    if not software.get("cuda_runtime") or environment.get("torch_cuda_runtime") != software["cuda_runtime"]:
        raise ValueError("Shared PyTorch CUDA runtime mismatch")
    for key in INFERENCE_KEYS:
        if key not in profile["inference"] or config[key] != profile["inference"][key]:
            raise ValueError(f"Shared inference configuration mismatch: {key}")
    if config["vllm_version"] != software["vllm"]:
        raise ValueError("Profile vLLM software/inference mismatch")
    return profile_id(profile)


def same_environment(summaries: list[dict]) -> str:
    ids = [s.get("team_environment_id") for s in summaries]
    if not ids or any(not value for value in ids) or len(set(ids)) != 1:
        raise ValueError("Missing or different shared environment IDs; do not combine these runs as controlled comparisons")
    return ids[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--profile", type=Path, default=Path("configs/team_environment.json"))
    verify.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    verify.add_argument("--environment", type=Path, required=True, help="Environment capture or server manifest")
    snapshot = sub.add_parser("snapshot", help="Capture a candidate runtime; does not confirm team agreement")
    snapshot.add_argument("--profile", type=Path, default=Path("configs/team_environment.json"))
    snapshot.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    snapshot.add_argument("--environment", type=Path, required=True)
    snapshot.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("summaries", type=Path, nargs="+")
    args = p.parse_args()
    if args.command == "snapshot":
        info = read_json(args.environment)
        profile = read_json(args.profile)
        # Preserve the requested GPU/model. Never silently select another GPU.
        rows = list(csv.reader(info.get("nvidia_gpus", [])))
        if not rows:
            raise ValueError("No NVIDIA GPU was captured")
        profile["software"].update(driver_version=rows[0][2].strip(),
            cuda_runtime=info["torch_cuda_runtime"], torch=info["packages"]["torch"])
        profile.update(status="locked", scope="candidate")
        validate(profile, load_config(args.config), info, allow_trial=True)
        write_json(args.out, profile)
        print(f"Candidate runtime snapshot: {args.out}; team agreement is still required")
    elif args.command == "verify":
        info = read_json(args.environment)
        print(validate(read_json(args.profile), load_config(args.config), info.get("environment", info)))
    else:
        print(same_environment([read_json(path) for path in args.summaries]))


if __name__ == "__main__":
    main()
