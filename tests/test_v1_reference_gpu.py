"""Real BF16 mixed FA vs independent FP32 dense attention, with negative control."""
import unittest
from types import SimpleNamespace
import torch
from benchmarks.v1_validation.reference import dense_suffix_attention


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/FlashAttention")
class IndependentAttentionTests(unittest.TestCase):
    def test_real_gqa_mixed_rows_and_page_reuse(self):
        from nanovllm.layers.attention import store_kvcache
        from nanovllm.layers.mixed_attention import MixedAttentionContext, forward_mixed
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(101)
        # Real Qwen3-4B head geometry; Decode plus 249+8 boundary and cold Prefill.
        histories, counts = [256, 249, 0], [1, 8, 17]
        tables = [[5, 1], [3, 0], [2]]
        q = torch.randn(sum(counts), 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(counts), 8, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ck = torch.empty(6, 256, 8, 128, device="cuda", dtype=k.dtype)
        cv = torch.empty_like(ck)
        integer = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
        for round_id in range(2):
            if round_id:
                tables = [[0, 4], [1, 5], [3]]
            ck.fill_(19); cv.fill_(-23)
            slots, expected, offset = [], [], 0
            for table, h, n in zip(tables, histories, counts):
                old_k = torch.randn(h, 8, 128, device="cuda", dtype=k.dtype)
                old_v = torch.randn_like(old_k)
                for pos in range(h):
                    ck[table[pos//256], pos%256] = old_k[pos]
                    cv[table[pos//256], pos%256] = old_v[pos]
                sl = slice(offset, offset+n)
                expected.append(dense_suffix_attention(q[sl], torch.cat([old_k, k[sl]]),
                                                       torch.cat([old_v, v[sl]])))
                slots.extend(table[p//256]*256+p%256 for p in range(h, h+n))
                offset += n
            meta = MixedAttentionContext(1, 25, integer([257]), integer([tables[0]]),
                integer([0, 8, 25]), integer([0, 257, 274]),
                integer([tables[1], tables[2]+[-1]]), 17, 257, True)
            actual = forward_mixed(q, k, v, ck, cv,
                SimpleNamespace(mixed=meta, slot_mapping=integer(slots)), 128**-.5, store_kvcache)
            torch.testing.assert_close(actual.float(), torch.cat(expected), atol=.02, rtol=.02)

    def test_corrupt_old_values_are_detected(self):
        from nanovllm.layers.attention import store_kvcache
        from nanovllm.layers.mixed_attention import MixedAttentionContext, forward_mixed
        q = torch.zeros(8, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(8, 8, 128, device="cuda", dtype=q.dtype)
        v = torch.full_like(k, 2.)
        ck = torch.zeros(2, 256, 8, 128, device="cuda", dtype=q.dtype)
        cv = torch.ones_like(ck)
        integer = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
        meta = MixedAttentionContext(0, 8, None, None, integer([0, 8]), integer([0, 257]),
                                     integer([[0, 1]]), 8, 257, True)
        ctx = SimpleNamespace(mixed=meta, slot_mapping=torch.arange(249,257,device="cuda",dtype=torch.int32))
        expected = dense_suffix_attention(q, torch.zeros(257,8,128,device="cuda"),
            torch.cat([torch.ones(249,8,128,device="cuda"), v.float()]))
        good = forward_mixed(q,k,v,ck,cv,ctx,128**-.5,store_kvcache)
        torch.testing.assert_close(good.float(),expected,atol=.02,rtol=.02)
        cv[0,:249].fill_(7.)
        bad = forward_mixed(q,k,v,ck,cv,ctx,128**-.5,store_kvcache)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(bad.float(),expected,atol=.02,rtol=.02)


if __name__ == "__main__":
    unittest.main()
