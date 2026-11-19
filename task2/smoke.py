"""Run a labelled local HTTP fixture test. Numbers are synthetic, never GPU results."""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run
from .common import load_config
from .fixture import fixture_server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/http-smoke"))
    p.add_argument("--requests", type=Path, default=Path("workloads/original/requests.jsonl"))
    args = p.parse_args()
    cfg = {**load_config(Path("configs/baseline.json")), "warmup_sessions": 1, "measured_sessions": 4, "concurrency": 2}
    with fixture_server() as (url, captured):
        result = run(args.requests, cfg, args.out, url, None, smoke=True)
        assert len(captured) == 5 and result["aggregate"]["ok_requests"] == 4
    print(f"HTTP fixture passed. No model or GPU was used. Evidence: {args.out}")


if __name__ == "__main__":
    main()
