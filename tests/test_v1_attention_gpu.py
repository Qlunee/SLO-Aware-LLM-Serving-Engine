"""Direct paged mixed attention vs dense per-request FA, on the same Q/K/V."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flash_attn import flash_attn_func

from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.mixed_attention import MixedAttentionContext, forward_mixed


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA / FlashAttention")
class AttentionTests(unittest.TestCase):
    def test_single_rank_has_no_nccl_collectives(self):
        from nanovllm.engine.mixed_model_runner import MixedModelRunner

        class TinyModel(torch.nn.Module):
            def forward(self, ids, positions):
                return torch.ones(len(ids), 2, device="cuda")

            def compute_logits(self, hidden):
                return hidden

        runner = MixedModelRunner.__new__(MixedModelRunner)
        runner.world_size, runner.rank = 1, 0
        runner.block_size, runner.kv_block_cap = 256, 2
        runner.config = SimpleNamespace(gpu_memory_utilization=0.5, hf_config=SimpleNamespace(
            num_key_value_heads=1, hidden_size=128, num_attention_heads=1,
            head_dim=128, num_hidden_layers=1, dtype=torch.bfloat16))
        runner.model = TinyModel()
        runner.prepare_mixed = lambda batch: (torch.tensor([1], device="cuda"),
                                               torch.tensor([0], device="cuda"))
        batch = SimpleNamespace(validate=lambda: None, pure_decode=False,
                                sample_indices=(0,), sample_seq_ids=(123,))
        with patch("torch.distributed.all_reduce", side_effect=AssertionError("single-rank all_reduce")), \
             patch("torch.distributed.all_gather_object", side_effect=AssertionError("single-rank gather")):
            runner.allocate_kv_cache()
            result = runner.run_mixed(batch, verify_tp_plan=True)
        self.assertEqual(result["tokens"], {123: 0})

    def test_gqa_paged_mixed_and_page_reuse(self):
        torch.manual_seed(41)
        device, dtype = "cuda", torch.bfloat16
        block_size, q_heads, kv_heads, dim = 256, 8, 2, 128

        def integer(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        # Decode: history 256 + 1. Prefill: cached 255 + a 3-token chunk.
        # Reverse page order to expose accidental logical==physical assumptions.
        tables = [[3, 1], [2, 0]]
        q = torch.randn(4, q_heads, dim, device=device, dtype=dtype)
        k = torch.randn(4, kv_heads, dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        cache_k = torch.empty(4, block_size, kv_heads, dim, device=device, dtype=dtype)
        cache_v = torch.empty_like(cache_k)
        for round_id in range(2):
            # Change old bytes AND slot mapping; stale pages must never be read.
            if round_id:
                tables = [[0, 2], [1, 3]]
            old_k = [torch.randn(n, kv_heads, dim, device=device, dtype=dtype) for n in [256, 255]]
            old_v = [torch.randn_like(x) for x in old_k]
            cache_k.fill_(99)
            cache_v.fill_(-99)
            for row, count in enumerate([256, 255]):
                cache_k[tables[row][0], :count] = old_k[row]
                cache_v[tables[row][0], :count] = old_v[row]
            slots = [tables[0][1] * block_size]
            slots.extend(tables[1][p // block_size] * block_size + p % block_size for p in [255, 256, 257])
            meta = MixedAttentionContext(1, 3, integer([257]), integer([tables[0]]),
                                         integer([0, 3]), integer([0, 258]), integer([tables[1]]), 3, 258)
            context = SimpleNamespace(mixed=meta, slot_mapping=integer(slots))
            actual = forward_mixed(q, k, v, cache_k, cache_v, context, dim**-0.5, store_kvcache)
            expected = []
            for row, query_slice in enumerate([slice(0, 1), slice(1, 4)]):
                dense_k = torch.cat([old_k[row], k[query_slice]])
                dense_v = torch.cat([old_v[row], v[query_slice]])
                expected.append(flash_attn_func(q[query_slice].unsqueeze(0), dense_k.unsqueeze(0),
                                               dense_v.unsqueeze(0), causal=True).squeeze(0))
            torch.testing.assert_close(actual, torch.cat(expected), atol=0.02, rtol=0.02)

    def test_cold_prefill_uses_contiguous_kv(self):
        from flash_attn import flash_attn_varlen_func
        torch.manual_seed(42)
        q = torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(256, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache_k = torch.empty(2, 256, 2, 128, device="cuda", dtype=k.dtype)
        cache_v = torch.empty_like(cache_k)
        cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
        meta = MixedAttentionContext(0, 256, None, None, cu, cu,
            torch.tensor([[1]], device="cuda", dtype=torch.int32), 256, 256, False)
        context = SimpleNamespace(mixed=meta,
            slot_mapping=torch.arange(256, 512, device="cuda", dtype=torch.int32))
        with patch("nanovllm.layers.mixed_attention.flash_attn_varlen_func",
                   wraps=flash_attn_varlen_func) as call:
            actual = forward_mixed(q, k, v, cache_k, cache_v, context, 128**-0.5, store_kvcache)
        self.assertIsNone(call.call_args.kwargs["block_table"])
        self.assertEqual(call.call_args.args[1].data_ptr(), k.data_ptr())
        expected = flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True).squeeze(0)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(cache_k[1], k, atol=0, rtol=0)
        torch.testing.assert_close(cache_v[1], v, atol=0, rtol=0)

    def test_multiple_prefill_rows_with_different_cached_lengths(self):
        torch.manual_seed(43)
        lengths, histories = [1, 255, 257, 513], [256, 0, 256, 513]
        physical = iter([13, 2, 11, 4, 14, 0, 9, 1, 12, 7, 5])
        tables = [[next(physical) for _ in range((h + n + 255) // 256)]
                  for h, n in zip(histories, lengths)]
        cache_k = torch.full((16, 256, 2, 128), 99.0, device="cuda", dtype=torch.bfloat16)
        cache_v = torch.full_like(cache_k, -99)
        q = torch.randn(sum(lengths), 8, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(lengths), 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        old_k = [torch.randn(h, 2, 128, device="cuda", dtype=k.dtype) for h in histories]
        old_v = [torch.randn_like(x) for x in old_k]
        slots, expected, offset = [], [], 0
        for table, h, n, pk, pv in zip(tables, histories, lengths, old_k, old_v):
            for pos in range(h):
                cache_k[table[pos // 256], pos % 256] = pk[pos]
                cache_v[table[pos // 256], pos % 256] = pv[pos]
            slots.extend(table[p // 256] * 256 + p % 256 for p in range(h, h + n))
            part = slice(offset, offset + n)
            expected.append(flash_attn_func(q[part].unsqueeze(0),
                torch.cat([pk, k[part]]).unsqueeze(0), torch.cat([pv, v[part]]).unsqueeze(0),
                causal=True).squeeze(0))
            offset += n
        def integer(values):
            return torch.tensor(values, device="cuda", dtype=torch.int32)
        cu_q, cu_k = [0], [0]
        for h, n in zip(histories[1:], lengths[1:]):
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + h + n)
        width = max(map(len, tables[1:]))
        prefill_tables = [t + [-1] * (width - len(t)) for t in tables[1:]]
        meta = MixedAttentionContext(1, sum(lengths[1:]), integer([257]), integer([tables[0]]),
            integer(cu_q), integer(cu_k), integer(prefill_tables), max(lengths[1:]), 1026, True)
        actual = forward_mixed(q, k, v, cache_k, cache_v,
            SimpleNamespace(mixed=meta, slot_mapping=integer(slots)), 128**-0.5, store_kvcache)
        torch.testing.assert_close(actual, torch.cat(expected), atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
