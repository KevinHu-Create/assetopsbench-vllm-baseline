"""Prepare an unchanged IBM task and the exact official direct-LLM prompt.

Only literal strings in a pinned source AST are read; source code is not executed.
Ground truth is written separately and is never included in model requests.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

from .common import digest, write_json

IBM_COMMIT = "0d044061ab3ffd4b213ddf6a0868c360d7e9667c"
RUNNER_PATH = "src/agent/direct_llm_agent/runner.py"
QUESTION_PATH = "src/couchdb/scenarios_data/scenario_1/question.txt"


def extract_messages(source: str, question: str) -> list[dict]:
    tree = ast.parse(source)
    system = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                     t.id == "_DIRECT_LLM_SYSTEM_PROMPT" for t in node.targets))
    prompt = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets))
    if not isinstance(prompt, ast.JoinedStr):
        raise ValueError("Upstream prompt structure changed; inspect before adapting")
    parts = []
    for item in prompt.values:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            parts.append(item.value)
        elif isinstance(item, ast.FormattedValue) and isinstance(item.value, ast.Name) and item.conversion == -1 and item.format_spec is None:
            parts.append({"_DIRECT_LLM_SYSTEM_PROMPT": system, "question": question}[item.value.id])
        else:
            raise ValueError("Unsupported expression in original prompt")
    return [{"role": "user", "content": "".join(parts)}]


def prepare(upstream: Path, out: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != IBM_COMMIT:
        raise ValueError(f"Expected IBM commit {IBM_COMMIT}, got {commit}")
    selected = [RUNNER_PATH, QUESTION_PATH, "src/couchdb/scenarios_data/scenario_1/groundtruth.txt"]
    contents = {}
    for path in selected:
        raw = (upstream / path).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(upstream), "show", f"HEAD:{path}"])
        if raw != committed:
            raise ValueError(f"Upstream file has local edits: {path}")
        contents[path] = raw
    messages = extract_messages(contents[RUNNER_PATH].decode(), contents[QUESTION_PATH].decode())
    out.mkdir(parents=True, exist_ok=True)
    requests = out / "requests.jsonl"
    import json
    row = {"session_id": "ibm-scenario-1", "scenario_id": "scenario_1", "seq_idx": 0,
           "messages": messages, "response_format": "integer"}
    requests.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"repository": "https://github.com/IBM/AssetOpsBench", "commit": commit,
                "baseline_kind": "official_direct_llm_prompt_streaming_transport",
                "scenario_id": "scenario_1", "question_path": QUESTION_PATH,
                "runner_path": RUNNER_PATH,
                "source_sha256": {p: digest(b) for p, b in contents.items()},
                "requests_sha256": digest(requests.read_bytes()),
                "scope": "Model-only baseline, matching upstream direct_llm; no MCP tools or data retrieval. Not a full agent baseline."}
    write_json(out / "manifest.json", manifest)
    write_json(out / "gt.json", {"scenario_1": contents[selected[2]].decode().strip()})
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("workloads/original"))
    args = p.parse_args()
    print(prepare(args.upstream, args.out)["requests_sha256"])


if __name__ == "__main__":
    main()
