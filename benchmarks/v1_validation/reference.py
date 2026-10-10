"""Explicit dense causal oracle. Never calls FlashAttention or the mixed helper."""
import math
import types
import torch


def dense_suffix_attention(q, k, v, scale=None):
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("expected [tokens, heads, dim] Q/K/V")
    nq, heads, dim = q.shape
    nk, kv_heads, kd = k.shape
    if not 0 < nq <= nk or dim != kd or heads % kv_heads:
        raise ValueError("invalid causal suffix / GQA shape")
    if not all(torch.isfinite(x).all().item() for x in (q, k, v)):
        raise ValueError("nonfinite Q/K/V")
    q, k, v = (x.float() for x in (q, k, v))
    k, v = (x.repeat_interleave(heads // kv_heads, dim=1) for x in (k, v))
    scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(1, 2)
    scores = scores * (dim ** -.5 if scale is None else scale)
    query_positions = torch.arange(nk-nq, nk, device=q.device)
    key_positions = torch.arange(nk, device=q.device)
    scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], -math.inf)
    return (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)


def rms_reference(x, weight, eps, residual=None):
    # Avoid .float()/to(dtype) aliases followed by in-place updates in FP32 mode.
    total = x.float() if residual is None else x.float() + residual.float()
    normalized = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps)
    normalized = normalized.to(x.dtype) * weight
    return normalized if residual is None else (normalized, total.to(x.dtype))


def install_fp32_diagnostic(engine):
    """One-shot, Eager/TP1 only. Engine must be closed after the diagnostic.

    Production metadata/KV writes remain in use; attention arithmetic is dense
    FP32. This tests scheduling/cache semantics, NOT BF16 FA kernel accuracy.
    The independent BF16 QKV tests cover that separate boundary.
    """
    if engine.config.tensor_parallel_size != 1 or not engine.config.enforce_eager:
        raise ValueError("FP32 diagnostic requires Eager TP=1")
    from nanovllm.layers.layernorm import RMSNorm
    import nanovllm.layers.attention as original
    import nanovllm.layers.mixed_attention as mixed
    torch.backends.cuda.matmul.allow_tf32 = False
    engine.model_runner.model.float()
    engine.model_runner.kv_cache = engine.model_runner.kv_cache.float()
    for i, layer in enumerate(engine.model_runner.model.model.layers):
        layer.self_attn.attn.k_cache = engine.model_runner.kv_cache[0, i]
        layer.self_attn.attn.v_cache = engine.model_runner.kv_cache[1, i]
    norm_methods = []
    def norm(self, x, residual=None):
        return rms_reference(x, self.weight, self.eps, residual)
    for module in engine.model_runner.model.modules():
        if isinstance(module, RMSNorm):
            norm_methods.append((module, module.forward))
            module.forward = types.MethodType(norm, module)

    def gather(cache, table, length):
        pages = (length + cache.shape[1] - 1) // cache.shape[1]
        return cache[table[:pages].long()].flatten(0, 1)[:length]

    def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
               softmax_scale, causal, block_table=None):
        if not causal:
            raise ValueError("oracle only supports causal attention")
        qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        outputs = []
        for i, (a, b) in enumerate(zip(qs, qs[1:])):
            length = ks[i+1]-ks[i]
            pk = k[ks[i]:ks[i+1]] if block_table is None else gather(k, block_table[i], length)
            pv = v[ks[i]:ks[i+1]] if block_table is None else gather(v, block_table[i], length)
            outputs.append(dense_suffix_attention(q[a:b], pk, pv, softmax_scale))
        return torch.cat(outputs)

    def decode(q, k, v, cache_seqlens, block_table, softmax_scale, causal):
        if not causal:
            raise ValueError("oracle only supports causal attention")
        return torch.stack([dense_suffix_attention(q[i], gather(k, block_table[i], n),
                gather(v, block_table[i], n), softmax_scale)
                for i, n in enumerate(cache_seqlens.tolist())])

    bindings = []
    for module in (original, mixed):
        for name, replacement in (("flash_attn_varlen_func", varlen), ("flash_attn_with_kvcache", decode)):
            bindings.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)

    def restore_bindings():
        # The model intentionally remains FP32; it is owned by this diagnostic.
        for module, name, value in bindings:
            setattr(module, name, value)
        for module, method in norm_methods:
            module.forward = method
    return restore_bindings
