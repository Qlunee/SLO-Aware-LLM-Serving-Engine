"""Runtime identity checks for a reused V3 predictor."""
import unittest
from unittest.mock import patch

from benchmarks.v3_slack import run as slack_run
from benchmarks.v3_slack.run import runtime_mismatches, profiled_execution_drift


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.frozen = {"nccl_version": [2, 28, 9], "kv_blocks": 256,
            "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090",
                      "compute_capability": [8, 6], "total_memory_bytes": 25297616896}]}

    def test_json_round_trip_tuple_is_same_identity(self):
        current = {**self.frozen, "nccl_version": (2, 28, 9)}
        self.assertEqual(runtime_mismatches(self.frozen, current, False), [])

    def test_cross_gpu_requires_explicit_diagnostic_opt_in(self):
        gpu = {**self.frozen["gpus"][0], "total_memory_bytes": 25298141184}
        current = {**self.frozen, "gpus": [gpu]}
        self.assertEqual(runtime_mismatches(self.frozen, current, False), ["gpus"])
        self.assertEqual(runtime_mismatches(self.frozen, current, True), [])

    def test_cross_gpu_diagnostic_still_rejects_material_differences(self):
        gpu = {**self.frozen["gpus"][0], "name": "different GPU"}
        current = {**self.frozen, "gpus": [gpu]}
        self.assertEqual(runtime_mismatches(self.frozen, current, True), ["gpus"])
        current = {**self.frozen, "kv_blocks": 226}
        self.assertEqual(runtime_mismatches(self.frozen, current, True), ["kv_blocks"])

    def test_warmup_primes_singleton_decode_for_both_policies(self):
        trace = {"requests": [{"request_id": "first"}, {"request_id": "second"}]}
        engine, config = object(), object()
        with patch.object(slack_run, "warm_engine", side_effect=[
                {"requests": 1}, {"requests": 2}]) as warm:
            result = slack_run.prepare_warmup(engine, config, trace)
        self.assertEqual(warm.call_count, 2)
        self.assertEqual(warm.call_args_list[0].args,
                         (engine, config, {"requests": [trace["requests"][0]]}))
        self.assertEqual(warm.call_args_list[1].args, (engine, config, trace))
        self.assertEqual(result["singleton_decode_primed"], True)

    def test_policy_source_can_change_without_invalidating_profiled_v1_path(self):
        frozen = {"nanovllm/engine/mixed_scheduler.py": "old-v1",
                  "nanovllm/engine/slack_scheduler.py": "old-policy",
                  "nanovllm/engine/step_predictor.py": "old-predictor"}
        current = {**frozen, "nanovllm/engine/slack_scheduler.py": "new-policy",
                   "nanovllm/engine/step_predictor.py": "new-predictor"}
        self.assertEqual(profiled_execution_drift(frozen, current), [])
        current["nanovllm/engine/mixed_scheduler.py"] = "changed-v1"
        self.assertEqual(profiled_execution_drift(frozen, current),
                         ["nanovllm/engine/mixed_scheduler.py"])


if __name__ == "__main__":
    unittest.main()
