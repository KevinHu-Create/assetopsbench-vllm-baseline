"""Summarize a real Chrome/PyTorch trace; overlapping event sums are not wall time."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from .common import digest, write_json


def summarize_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    groups = defaultdict(lambda: {"events": 0, "duration_us": 0.0})
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        key = (str(event.get("cat", "uncategorized")), str(event.get("name", "unnamed")))
        groups[key]["events"] += 1
        groups[key]["duration_us"] += event["dur"]
    entries = [{"category": cat, "name": name, **values} for (cat, name), values in groups.items()]
    entries.sort(key=lambda v: v["duration_us"], reverse=True)
    return {"trace_name": path.name, "trace_sha256": digest(path.read_bytes()), "total_trace_events": len(events),
            "duration_definition": "Sum of X-event durations in microseconds; inclusive and potentially overlapping. Not critical-path wall time.",
            "prefill_decode_attribution": "Not inferred from kernel names. Inspect request IDs, iteration ranges and the original trace in Perfetto.",
            "top_events": entries[:50]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = summarize_trace(args.trace)
    write_json(args.out, result)
    print(f"Summarized {result['total_trace_events']} events")


if __name__ == "__main__":
    main()
