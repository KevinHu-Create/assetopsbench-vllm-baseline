"""Read-only GCP billing/GPU quota check; creates no VM or other paid resource."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--out", type=Path, default=Path("outputs/gcp-preflight.json"))
    args = p.parse_args()
    commands = {
        "billing": ["gcloud", "billing", "projects", "describe", args.project, "--format=json"],
        "global": ["gcloud", "compute", "project-info", "describe", "--project=" + args.project, "--format=json"],
        "regional": ["gcloud", "compute", "regions", "describe", args.region, "--project=" + args.project, "--format=json"],
    }
    def query(item):
        key, command = item
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            # Avoid dumping potentially sensitive CLI diagnostics into a shared report.
            raise RuntimeError(f"{key} query failed (exit {result.returncode}); inspect gcloud locally")
        return key, json.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=3) as pool:
        data = dict(pool.map(query, commands.items()))
    global_gpu = next((q for q in data["global"].get("quotas", []) if q["metric"] == "GPUS_ALL_REGIONS"), None)
    regional_gpu = [q for q in data["regional"].get("quotas", []) if q["metric"] == "NVIDIA_L4_GPUS"]
    available = lambda q: q is not None and q.get("limit", 0) - q.get("usage", 0) >= 1
    ready = bool(data["billing"].get("billingEnabled") and available(global_gpu) and any(available(q) for q in regional_gpu))
    report = {"project": args.project, "region": args.region,
              "billing_enabled": bool(data["billing"].get("billingEnabled")),
              "global_gpu_quota": global_gpu, "regional_l4_quota": regional_gpu,
              "quota_ready_for_one_l4": ready,
              "scope": "Quota readiness only; not a guarantee of GPU stock, compatible drivers or model fit. No resources created."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
