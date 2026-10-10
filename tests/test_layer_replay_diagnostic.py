"""CPU regression tests for the forensic probe, not production acceptance."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

from benchmarks.v1_diagnostics.layer_replay import (
    LogicalKV, Recorder, expected_slots, gather_pages, tensor_error,
)


class LayerReplayProbeTests(unittest.TestCase):
    def test_reordered_pages_and_tail(self):
        cache = torch.arange(12.).reshape(3, 2, 1, 2)
        actual = gather_pages(cache, [2, 0], 3)
        torch.testing.assert_close(actual, torch.tensor([[[8., 9.]], [[10., 11.]], [[0., 1.]]]))
        self.assertEqual(expected_slots([2, 0], 1, 3, 2), [5, 0])
        with self.assertRaises(ValueError):
            gather_pages(cache, [-1], 1)
        with self.assertRaises(ValueError):
            gather_pages(cache, [0], 3)

    def test_shadow_independent_of_page_mapping_and_detects_corruption(self):
        shadow = LogicalKV(4)
        k = torch.tensor([[[1., 2.]], [[3., 4.]]])
        v = k + 10
        shadow.write(0, k, v)
        k.fill_(99)  # The recorder owns a copy, not an alias.
        shadow.write(2, torch.tensor([[[5., 6.]]]), torch.tensor([[[15., 16.]]]))
        sk, sv = shadow.read(3)
        self.assertEqual(sk[0, 0, 0].item(), 1.)
        self.assertTrue(tensor_error(sv, sv.clone())["exact"])
        corrupt = sv.clone()
        corrupt[0, 0, 0] += 7
        self.assertFalse(tensor_error(corrupt, sv)["exact"])
        with self.assertRaises(ValueError):
            shadow.read(4)

    def test_finite_shape_and_zero_reference(self):
        self.assertEqual(tensor_error(torch.ones(2), torch.zeros(2))["max_abs"], 1.)
        with self.assertRaises(ValueError):
            tensor_error(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError):
            tensor_error(torch.tensor([float("nan")]), torch.ones(1))

    def test_shadow_rewrite_and_bounds(self):
        shadow = LogicalKV(2)
        x = torch.ones(2, 1, 2)
        shadow.write(0, x, x)
        shadow.write(1, 3*x[:1], 4*x[:1])
        k, v = shadow.read(2)
        self.assertEqual(k[1, 0, 0].item(), 3.)
        self.assertEqual(v[1, 0, 0].item(), 4.)
        with self.assertRaises(ValueError):
            shadow.write(2, x[:1], x[:1])

    def test_observational_hooks_preserve_output_and_detect_old_kv_corruption(self):
        from nanovllm.utils.context import get_context, reset_context, set_context
        from benchmarks.v1_validation.reference import dense_suffix_attention

        class ToyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 1
                self.head_dim = 2
                self.scale = 2**-.5
                self.k_cache = torch.zeros(2, 2, 1, 2)
                self.v_cache = torch.zeros_like(self.k_cache)

            def forward(self, q, k, v):
                c = get_context()
                for j, slot in enumerate(c.slot_mapping.tolist()):
                    page, offset = divmod(slot, 2)
                    self.k_cache[page, offset].copy_(k[j])
                    self.v_cache[page, offset].copy_(v[j])
                length = int(c.cu_seqlens_k[-1]) if c.is_prefill else int(c.context_lens[0])
                return dense_suffix_attention(q, gather_pages(self.k_cache, [1, 0], length),
                                              gather_pages(self.v_cache, [1, 0], length), self.scale)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.attn = ToyAttention()
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(2, 4, bias=False)

            def forward(self, ids, positions):
                x = torch.stack([ids.float(), positions.float()+1], -1).unsqueeze(1)
                return self.model.layers[0].self_attn.attn(torch.ones_like(x), x, x+3)

        class Seq:
            def __init__(self, tokens, computed, outputs):
                self.token_ids, self.num_cached_tokens = tokens, computed
                self.num_completion_tokens, self.block_table = outputs, [1, 0]

            def __len__(self):
                return len(self.token_ids)

        try:
            for corrupt in (False, True):
                model = ToyModel()
                engine = SimpleNamespace(model_runner=SimpleNamespace(model=model))
                with tempfile.TemporaryDirectory() as folder:
                    rec = Recorder(engine, [7, 8], 1, Path(folder))
                    rec.start("toy")
                    rec.active = rec.spec(Seq([7, 8], 0, 0), 0, 2, 0, 2, "prefill", 0, 0)
                    set_context(True, cu_seqlens_q=torch.tensor([0, 2]),
                        cu_seqlens_k=torch.tensor([0, 2]), slot_mapping=torch.tensor([2, 3]))
                    model(torch.tensor([7, 8]), torch.tensor([0, 1]))
                    if corrupt:
                        model.model.layers[0].self_attn.attn.v_cache[1, 0].add_(7)
                    rec.active = rec.spec(Seq([7, 8, 9], 2, 1), 0, 1, 2, 3, "decode", 0, 0)
                    set_context(False, slot_mapping=torch.tensor([0]),
                                context_lens=torch.tensor([3]), block_tables=torch.tensor([[1, 0]]))
                    ids, pos = torch.tensor([9]), torch.tensor([2])
                    observed = model(ids, pos)
                    model.lm_head(observed.flatten(1))
                    rec.close()
                    plain = model(ids, pos)
                    torch.testing.assert_close(observed, plain, atol=0, rtol=0)
                    path = rec.paths["toy"]
                    self.assertEqual(len(path["layers"]), 1)
                    self.assertEqual(path["kv_checks"][-1]["v"]["exact"], not corrupt)
                    self.assertEqual(path["kv_checks"][-1]["k"]["exact"], True)
                    self.assertTrue(all(value for check in path["state_checks"]
                                        for value in check.values() if isinstance(value, bool)))
                    self.assertTrue((Path(folder)/"toy/layer_00.pt").exists())
        finally:
            reset_context()


if __name__ == "__main__":
    unittest.main()
