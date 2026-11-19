"""Check whether real GPU evidence, performance tables and a CUDA trace exist."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .common import read_json, write_json


def check(root: Path, *, allow_trial: bool = False) -> dict:
    missing = []
    environment_ids = []
    for concurrency in (1, 4, 8):
        folder = root / f"baseline-c{concurrency}"
        try:
            summary = read_json(folder / "run_summary.json")
            allowed_kinds = {"gpu_baseline", "gpu_compatibility_trial"} if allow_trial else {"gpu_baseline"}
            if not allow_trial and summary.get("measurement_kind") == "gpu_compatibility_trial":
                missing.append(f"baseline-c{concurrency}: candidate trial; team configuration unconfirmed")
            elif not summary.get("verified_gpu_run") or summary.get("measurement_kind") not in allowed_kinds:
                missing.append(f"baseline-c{concurrency}: requires an unprofiled GPU run")
            environment_ids.append(summary.get("team_environment_id"))
            if not summary.get("team_environment_id"):
                missing.append(f"baseline-c{concurrency}: missing shared environment ID")
            reset = read_json(folder / "cache_reset.json")
            if reset.get("success") is not True or reset.get("after_warmup") is not True or reset.get("before_timing") is not True:
                missing.append(f"baseline-c{concurrency}: prefix cache reset not confirmed")
            aggregate = summary["aggregate"]
            if aggregate["ok_requests"] < 30 or aggregate["output_tokens_per_second"] is None:
                missing.append(f"baseline-c{concurrency}: need >=30 successful requests with token usage")
            if not (folder / "run_log.jsonl").is_file() or not (folder / "baseline.csv").is_file():
                missing.append(f"baseline-c{concurrency}: missing logs or table")
        except (OSError, ValueError, KeyError):
            missing.append(f"baseline-c{concurrency}: no usable summary")
    if environment_ids and len(set(environment_ids)) != 1:
        missing.append("baseline groups used different shared environments")
    try:
        profile = read_json(root / "profile/profile_control.json")
        if not profile["stopped"]:
            missing.append("profiler did not stop cleanly")
    except (OSError, ValueError, KeyError):
        missing.append("no successful profiling capture")
    cuda = False
    traces = list((root / "traces").rglob("*.trace.json*"))
    for path in traces:
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            events = data.get("traceEvents", []) if isinstance(data, dict) else data
            cuda = cuda or any(e.get("cat") == "kernel" for e in events)
        except (OSError, ValueError, TypeError):
            continue
    if not cuda:
        missing.append("no trace containing CUDA kernel events")
    return {"complete": not missing, "scope": "candidate_trial" if allow_trial else "formal_team_baseline",
            "missing": missing, "trace_files": len(traces)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--trial", action="store_true", help="Validate real candidate evidence without confirming team agreement")
    args = p.parse_args()
    result = check(args.results, allow_trial=args.trial)
    write_json(args.results / "acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
