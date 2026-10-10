"""One QKV projection, one KV write, two phase-specific attention calls."""
from dataclasses import dataclass

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

"""
它的核心目标是：
同时处理两类请求：
decode（正在生成下一 token）
prefill（还在处理前缀/输入）
把它们合并到一次 attention 计算里
让模型在一个 batch 中既能继续生成，又能处理新来的 token 片段
"""
@dataclass
class MixedAttentionContext:
    num_decode: int
    num_prefill_tokens: int
    decode_lens: torch.Tensor | None
    decode_tables: torch.Tensor | None
    prefill_cu_q: torch.Tensor | None
    prefill_cu_k: torch.Tensor | None
    prefill_tables: torch.Tensor | None
    max_q: int
    max_k: int
    prefill_has_history: bool = True


def forward_mixed(q, k, v, k_cache, v_cache, context, scale, store_fn):
    meta = context.mixed
    if not k_cache.numel() or not v_cache.numel():
        raise RuntimeError("mixed forward requires an allocated BF16 KV pool")
    store_fn(k, v, k_cache, v_cache, context.slot_mapping)
    parts = []
    d = meta.num_decode
    if d:
        # decode用带 kv cache的 flash attention
        out = flash_attn_with_kvcache(
            q[:d].unsqueeze(1), k_cache, v_cache,
            cache_seqlens=meta.decode_lens, block_table=meta.decode_tables,
            softmax_scale=scale, causal=True,
        )
        parts.append(out.squeeze(1))
    if meta.num_prefill_tokens:
        out = flash_attn_varlen_func(
            # q[d:] 是 prefill 的 query，因为前面 d 个是 decode
            q[d:], k_cache if meta.prefill_has_history else k[d:],
            v_cache if meta.prefill_has_history else v[d:],
            cu_seqlens_q=meta.prefill_cu_q,
            cu_seqlens_k=meta.prefill_cu_k,
            max_seqlen_q=meta.max_q, max_seqlen_k=meta.max_k,
            block_table=meta.prefill_tables if meta.prefill_has_history else None,
            softmax_scale=scale, causal=True,
        )
        parts.append(out)
    # The flat layout is decode-first, then prefill. No arbitrary scatter needed.
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
