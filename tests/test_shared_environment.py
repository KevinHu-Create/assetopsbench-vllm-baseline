"""Prevent unconfirmed profiles and mixed-GPU comparisons from becoming formal data."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from task2.common import load_config, read_json
from task2.http import reset_prefix_cache
from task2.team import INFERENCE_KEYS, same_environment, validate

ROOT = Path(__file__).resolve().parents[1]


class TestSharedEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/baseline.json")
        self.profile = {"status": "locked", "gpu": {"name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40536, "count": 1},
            "software": {"driver_version": "580.95.05", "cuda_runtime": "12.9", "vllm": self.config["vllm_version"], "torch": "fixture-torch"},
            "inference": {k: self.config[k] for k in INFERENCE_KEYS}}
        self.info = {"nvidia_gpus": ["NVIDIA A100-SXM4-40GB, 40536, 580.95.05, 8.0"],
                     "packages": {"vllm": self.config["vllm_version"], "torch": "fixture-torch"}, "torch_cuda_runtime": "12.9"}

    def test_unselected_repository_profile_cannot_start_formal_measurement(self):
        with self.assertRaisesRegex(ValueError, "draft"):
            validate(read_json(ROOT / "configs/team_environment.json"), self.config, self.info)

    def test_same_profile_accepts_only_planned_cache_concurrency_changes(self):
        identity = validate(self.profile, self.config, self.info)
        changed = {**self.config, "cache_mode": "on", "concurrency": 4}
        self.assertEqual(identity, validate(self.profile, changed, self.info))
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate(self.profile, {**self.config, "dtype": "bfloat16"}, self.info)

    def test_l4_a10080_and_extra_gpu_cannot_reuse_a10040_profile(self):
        for rows in (["NVIDIA L4, 23034, 580.95.05, 8.9"],
                     ["NVIDIA A100-SXM4-80GB, 81920, 580.95.05, 8.0"], self.info["nvidia_gpus"] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate(self.profile, self.config, {**self.info, "nvidia_gpus": rows})

    def test_software_drift_rejected_and_missing_profile_id_not_comparable(self):
        changed = copy.deepcopy(self.info)
        changed["packages"]["torch"] = "different-torch"
        with self.assertRaisesRegex(ValueError, "torch"):
            validate(self.profile, self.config, changed)
        for summaries in ([{}, {}], [{"team_environment_id": "a"}, {"team_environment_id": "b"}]):
            with self.assertRaises(ValueError):
                same_environment(summaries)
        self.assertEqual(same_environment([{"team_environment_id": "a"}] * 3), "a")

    def test_candidate_profile_requires_explicit_trial(self):
        candidate = {**self.profile, "scope": "candidate"}
        with self.assertRaisesRegex(ValueError, "Candidate"):
            validate(candidate, self.config, self.info)
        self.assertTrue(validate(candidate, self.config, self.info, allow_trial=True))

    def test_http200_false_cache_reset_is_not_success(self):
        with patch("task2.http.endpoint", return_value=b'{"success": false}'):
            with self.assertRaises(RuntimeError):
                reset_prefix_cache("http://fixture/v1")
        with patch("task2.http.endpoint", return_value=b'{"success": true}') as endpoint:
            self.assertTrue(reset_prefix_cache("http://fixture/v1")["success"])
            self.assertEqual(endpoint.call_args.args, ("http://fixture/reset_prefix_cache", "POST"))


if __name__ == "__main__":
    unittest.main()
