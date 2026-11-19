"""Contract tests for measurement failures, original prompts and honest evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from task2.benchmark import run, summarize, load_sessions
from task2.check import check
from task2.client import StreamClient
from task2.common import load_config, write_json
from task2.fixture import fixture_server
from task2.prepare import extract_messages
from task2.serve import command
from task2.trace_summary import summarize_trace

ROOT = Path(__file__).resolve().parents[1]


class TestTask2(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/baseline.json")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def generate(self, url, text):
        client = StreamClient(url, timeout=1)
        self.addCleanup(client.close)
        return client.generate({"model": "fixture", "messages": [{"role": "user", "content": text}], "max_tokens": 8}, "test-id")

    def test_role_events_not_ttft_and_reasoning_is_separate(self):
        with fixture_server() as (url, _):
            result = self.generate(url, "reasoning")
        self.assertEqual(result["error"]["status"], "ok")
        self.assertGreaterEqual(result["timing_ms"]["ttft_ms"], 10)
        self.assertGreater(result["timing_ms"]["first_content_ms"], result["timing_ms"]["ttft_ms"])
        self.assertEqual(result["raw_response"], "2")
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertEqual(len(result["stream_chunks"]), 2)  # chunks are not token counts

    def test_stream_and_http_failure_classification(self):
        with fixture_server() as (url, _):
            for text, status in (("http_error", "api_error"), ("malformed", "stream_error"), ("incomplete", "incomplete_stream")):
                with self.subTest(text=text):
                    self.assertEqual(self.generate(url, text)["error"]["status"], status)

    def test_missing_cache_is_null_and_length_counted(self):
        with fixture_server() as (url, _):
            a = self.generate(url, "no_cache_usage")
            b = self.generate(url, "length")
        self.assertIsNone(a["usage"]["cached_tokens"])
        summary = summarize([a, b], 1000)
        self.assertIsNone(summary["usage_totals"]["cached_tokens"])
        self.assertEqual(summary["truncated_responses"], 1)
        self.assertEqual(summary["output_tokens_per_second"], 2)

    def test_failed_requests_do_not_pollute_success_latency(self):
        with fixture_server() as (url, _):
            good, bad = self.generate(url, "normal"), self.generate(url, "http_error")
        summary = summarize([good, bad], 1000)
        self.assertEqual(summary["ttft_ms"]["n"], 1)
        self.assertEqual(summary["successful_rps"], 1)
        self.assertEqual(summary["failure_rate"], .5)

    def test_exact_official_prompt_ast_and_no_execution(self):
        source = '_DIRECT_LLM_SYSTEM_PROMPT = "SYS\\n"\ndef run(question):\n    prompt = f"{_DIRECT_LLM_SYSTEM_PROMPT}\\n\\nUser task:\\n{question}\\n\\nFinal answer:"\n'
        self.assertEqual(extract_messages(source, "QUESTION"), [{"role": "user", "content": "SYS\n\n\nUser task:\nQUESTION\n\nFinal answer:"}])
        unsafe = source.replace('{question}', '{dangerous()}')
        with self.assertRaises(ValueError):
            extract_messages(unsafe, "QUESTION")

    def make_workload(self):
        requests = self.root / "requests.jsonl"
        rows = [{"session_id": "original", "scenario_id": "scenario_1", "seq_idx": i,
                 "messages": [{"role": "user", "content": "normal"}], "response_format": "integer"} for i in (0, 1)]
        requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return requests

    def test_warmup_exclusion_session_order_gt_isolation_and_smoke_label(self):
        requests = self.make_workload()
        gt = self.root / "gt.json"
        write_json(gt, {"scenario_1": "DO_NOT_SEND_PRIVATE_GT"})
        cfg = {**self.config, "warmup_sessions": 1, "measured_sessions": 4, "concurrency": 2}
        with fixture_server() as (url, captured):
            report = run(requests, cfg, self.root / "out", url, None, gt, smoke=True)
        self.assertEqual(report["aggregate"]["total_requests"], 8)
        self.assertEqual(len(captured), 10)
        self.assertEqual(report["measurement_kind"], "smoke")
        self.assertFalse(report["verified_gpu_run"])
        self.assertEqual(report["aggregate"]["task_accuracy"], 0)
        self.assertNotIn("DO_NOT_SEND_PRIVATE_GT", json.dumps(captured))
        for index in range(4):
            session_ids = [r["request_id"] for r in captured if f"-measured-{index}-" in r["request_id"]]
            self.assertTrue(session_ids[0].endswith("-0"))
            self.assertTrue(session_ids[1].endswith("-1"))
        with self.assertRaises(ValueError):
            run(requests, cfg, self.root / "out", "http://unused/v1", None, smoke=True)

    def test_formal_run_cannot_use_fixture_without_gpu_manifest(self):
        with self.assertRaises(ValueError):
            run(self.make_workload(), self.config, self.root / "formal", "http://127.0.0.1:1/v1", None)
        self.assertFalse(check(self.root)["complete"])

    def test_real_cache_switch_and_pinned_model(self):
        off = command(self.config, 8000, None)
        on = command({**self.config, "cache_mode": "on"}, 8000, self.root)
        self.assertIn("--no-enable-prefix-caching", off)
        self.assertIn("--enable-prefix-caching", on)
        self.assertIn(self.config["revision"], off)
        self.assertIn("--profiler-config", on)
        self.assertNotIn("--profiler-config", off)

    def test_reject_duplicate_session_steps_and_invalid_config(self):
        path = self.make_workload()
        line = path.read_text().splitlines()[0]
        path.write_text(line + "\n" + line + "\n")
        with self.assertRaises(ValueError):
            load_sessions(path)
        config = self.root / "bad.json"
        write_json(config, {**self.config, "concurrency": 0})
        with self.assertRaises(ValueError):
            load_config(config)

    def test_trace_summary_preserves_categories_without_walltime_claim(self):
        path = self.root / "trace.json"
        write_json(path, {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "gemm", "dur": 30},
            {"ph": "X", "cat": "cpu_op", "name": "gemm", "dur": 50}, {"ph": "M", "name": "ignored"}]})
        report = summarize_trace(path)
        self.assertEqual(report["total_trace_events"], 3)
        self.assertEqual(len(report["top_events"]), 2)
        self.assertIn("overlapping", report["duration_definition"])


if __name__ == "__main__":
    unittest.main()
