import unittest
import importlib.util
import tempfile
from pathlib import Path

from diagnostics.online_token_replay.core import validate_plan, valid_prefix_slots


def example():
    return {
        "rows": [{"request_id": "r", "seq_id": 7, "phase": "decode",
                  "start": 3, "end": 4, "a": 0, "b": 1,
                  "table": [2, 0], "history": [4, 5, 6, 7],
                  "output_count": 2}],
        "input_ids": [7], "positions": [3], "slots": [1],
        "sample_seq_ids": [7], "sample_indices": [0],
        "block_size": 2, "total_blocks": 3,
    }


class PlanTests(unittest.TestCase):
    def test_valid_history_and_physical_slots(self):
        self.assertEqual(validate_plan(example()), [0, 2])

    def test_snapshot_excludes_current_write_and_uninitialized_tail(self):
        self.assertEqual(valid_prefix_slots(example()), [0, 4, 5])

    def test_wrong_slot_or_history_is_rejected(self):
        for field, value in (("slots", [0]), ("input_ids", [99]), ("positions", [2])):
            plan = example()
            plan[field] = value
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_missing_sample_row_is_rejected(self):
        plan = example()
        plan["sample_seq_ids"] = []
        with self.assertRaises(ValueError):
            validate_plan(plan)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch CPU runtime")
class SnapshotTests(unittest.TestCase):
    def test_snapshot_roundtrip_sanitizes_invalid_bytes_and_restores_every_time(self):
        import torch
        from diagnostics.online_token_replay.tensors import snapshot_kv, restore_kv
        pool = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2, 1, 1).repeat(2, 1, 1, 1, 1, 1)
        image = snapshot_kv(pool, example(), max_bytes=1000)
        self.assertEqual(image["block_ids"], [0, 2])
        self.assertEqual(image["kv"].flatten().tolist(), [0., 0., 4., 5.] * 2)
        pool.fill_(-99)
        restore_kv(pool, image)
        self.assertEqual(pool[0, 0, :, :, 0, 0].tolist(), [[0., 0.], [-99., -99.], [4., 5.]])
        pool.fill_(88)
        restore_kv(pool, image)
        self.assertEqual(pool[0, 0, 2, :, 0, 0].tolist(), [4., 5.])

    def test_fixed_forward_restores_kv_before_each_execution(self):
        import torch
        from types import SimpleNamespace
        from diagnostics.online_token_replay.replay import forward_snapshot
        from diagnostics.online_token_replay.tensors import snapshot_kv, pack_context
        from nanovllm.utils.context import Context
        plan = example()
        plan["selection"] = "native"
        pool = torch.ones((2, 1, 3, 2, 1, 1))
        class HardwareBoundaryModel(torch.nn.Module):
            # CPU-only tensor fixture, not an Attention/model correctness oracle.
            def forward(self, ids, positions):
                value = pool[0, 0, 2, 0, 0, 0].clone()
                pool.add_(100)
                return ids.float().unsqueeze(1)+value
            def compute_logits(self, hidden):
                return hidden * torch.tensor([[1., 2., 3., 4.]])
        image = snapshot_kv(pool, plan, 1000)
        context = Context(False, slot_mapping=torch.tensor([1], dtype=torch.int32),
                          context_lens=torch.tensor([4], dtype=torch.int32),
                          block_tables=torch.tensor([[2, 0]], dtype=torch.int32))
        snapshot = {"plan": plan, "image": image, "context": pack_context(context),
                    "input_ids": torch.tensor([7]), "positions": torch.tensor([3])}
        runner = SimpleNamespace(kv_cache=pool, model=HardwareBoundaryModel())
        first = forward_snapshot(runner, snapshot, "r")
        second = forward_snapshot(runner, snapshot, "r")
        self.assertEqual(first.tolist(), [8., 16., 24., 32.])
        self.assertTrue(torch.equal(first, second))
        isolated = forward_snapshot(runner, snapshot, "r", isolate=True)
        self.assertTrue(torch.equal(first, isolated))

    def test_logit_comparison_reports_top1_change_without_acceptance_override(self):
        import torch
        from diagnostics.online_token_replay.tensors import compare_logits
        result = compare_logits(torch.tensor([0., 3., 2.]), torch.tensor([0., 2., 3.]))
        self.assertFalse(result["same_top1"])
        self.assertEqual(result["max_abs"], 1.)
        self.assertEqual(result["actual"]["top2_margin"], 1.)
        self.assertFalse(result["acceptance_override"])

    def test_cross_capture_comparison_refuses_different_histories(self):
        import torch
        from diagnostics.online_token_replay.report import compare_captured_vectors
        a = {"request_id": "r", "history": [1, 2], "logits": torch.tensor([0., 1.])}
        b = {"request_id": "r", "history": [1, 3], "logits": torch.tensor([0., 1.])}
        result = compare_captured_vectors(a, b)
        self.assertEqual(result["status"], "HISTORY_DIFFERS_NOT_A_COMMON_HISTORY_COMPARISON")
        self.assertIsNone(result["logits"])
        b["history"] = [1, 2]
        self.assertTrue(compare_captured_vectors(a, b)["logits"]["exact"])

    def test_snapshot_cap_and_nonfinite_logits_fail(self):
        import torch
        from diagnostics.online_token_replay.tensors import snapshot_kv, logit_summary
        with self.assertRaises(ValueError):
            snapshot_kv(torch.ones((2, 1, 3, 2, 1, 1)), example(), max_bytes=1)
        with self.assertRaises(ValueError):
            logit_summary(torch.tensor([float("nan"), 1.]))

    def test_online_capture_records_actual_sampled_row_and_restores_hooks(self):
        import torch
        from types import SimpleNamespace
        from diagnostics.online_token_replay.capture import OnlineCapture
        from nanovllm.utils.context import set_context, reset_context
        class Seq:
            seq_id, num_completion_tokens = 7, 2
            num_cached_tokens, num_scheduled_tokens = 3, 1
            block_table, token_ids = [2, 0], [4, 5, 6, 7]
            def __len__(self):
                return 4
        class HardwareBoundaryModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(1, 4, bias=False)
                with torch.no_grad():
                    self.lm_head.weight.copy_(torch.tensor([[1.], [2.], [3.], [4.]]))
            def forward(self, ids, positions):
                return ids.float().unsqueeze(1)
        class HardwareBoundaryRunner:
            def __init__(self):
                self.model = HardwareBoundaryModel()
                self.kv_cache = torch.ones((2, 1, 3, 2, 1, 1))
                self.block_size = 2
                self.config = SimpleNamespace(num_kvcache_blocks=3)
            def run(self, seqs, is_prefill):
                set_context(False, slot_mapping=torch.tensor([1]),
                            context_lens=torch.tensor([4]), block_tables=torch.tensor([[2, 0]]))
                try:
                    hidden = self.model(torch.tensor([7]), torch.tensor([3]))
                    return self.model.lm_head(hidden).argmax(-1).tolist()
                finally:
                    reset_context()
        runner = HardwareBoundaryRunner()
        engine = SimpleNamespace(model_runner=runner, step_count=1,
                                 by_seq={7: SimpleNamespace(request_id="r")})
        original = runner.run
        with tempfile.TemporaryDirectory() as directory:
            recorder = OnlineCapture(engine, "baseline", ["r"], 3, Path(directory), 1000)
            recorder.install()
            try:
                self.assertEqual(runner.run([Seq()], False), [3])
                self.assertEqual(recorder.entries[0]["targets"][0]["sampled_token"], 3)
                image = torch.load(Path(directory)/"snapshot-000.pt", weights_only=True)
                self.assertEqual(image["target_logits"]["7"].tolist(), [7., 14., 21., 28.])
            finally:
                recorder.close()
            self.assertEqual(runner.run, original)
            self.assertEqual(runner.run([Seq()], False), [3])
            self.assertEqual(len(recorder.entries), 1)

    def test_indexed_replay_excludes_intermediate_prefill_rows(self):
        import torch
        from types import SimpleNamespace
        from diagnostics.online_token_replay.replay import forward_snapshot
        from diagnostics.online_token_replay.tensors import snapshot_kv, pack_context
        from nanovllm.utils.context import Context
        from nanovllm.layers.mixed_attention import MixedAttentionContext
        rows = [
            {"request_id": "intermediate", "seq_id": 7, "phase": "prefill", "start": 0,
             "end": 2, "a": 0, "b": 2, "table": [0, 1], "history": [4, 5, 6], "output_count": 0},
            {"request_id": "target", "seq_id": 8, "phase": "prefill", "start": 0,
             "end": 2, "a": 2, "b": 4, "table": [2], "history": [8, 9], "output_count": 0},
        ]
        plan = {"rows": rows, "input_ids": [4, 5, 8, 9], "positions": [0, 1, 0, 1],
                "slots": [0, 1, 4, 5], "sample_seq_ids": [8], "sample_indices": [3],
                "block_size": 2, "total_blocks": 3, "selection": "indexed"}
        pool = torch.ones((2, 1, 3, 2, 1, 1))
        mixed = MixedAttentionContext(0, 4, None, None, torch.tensor([0, 2, 4]),
            torch.tensor([0, 2, 4]), torch.tensor([[0, 1], [2, -1]]), 2, 2, False)
        context = Context(False, slot_mapping=torch.tensor([0, 1, 4, 5]), mixed=mixed)
        class HardwareBoundaryModel(torch.nn.Module):
            def forward(self, ids, positions):
                return ids.float().unsqueeze(1)
            def compute_logits(self, hidden):
                from nanovllm.utils.context import get_context
                if get_context().is_prefill:
                    hidden = hidden[get_context().cu_seqlens_q[1:]-1]
                return hidden*torch.tensor([[1., 2., 3., 4.]])
        snapshot = {"plan": plan, "image": snapshot_kv(pool, plan, 1000),
                    "context": pack_context(context), "input_ids": torch.tensor(plan["input_ids"]),
                    "positions": torch.tensor(plan["positions"])}
        runner = SimpleNamespace(kv_cache=pool, model=HardwareBoundaryModel())
        self.assertEqual(forward_snapshot(runner, snapshot, "target").tolist(), [9., 18., 27., 36.])
        self.assertEqual(forward_snapshot(runner, snapshot, "target", isolate=True).tolist(), [9., 18., 27., 36.])

    def test_attention_metadata_mismatch_is_rejected(self):
        import torch
        from diagnostics.online_token_replay.tensors import validate_context, pack_context
        from nanovllm.utils.context import Context
        encoded = pack_context(Context(False, slot_mapping=torch.tensor([1]),
            context_lens=torch.tensor([3]), block_tables=torch.tensor([[2, 0]])))
        with self.assertRaises(ValueError):
            validate_context(example(), encoded)


if __name__ == "__main__":
    unittest.main()
