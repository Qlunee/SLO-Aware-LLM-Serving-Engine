"""Tensor snapshot/restore contracts, usable with CPU tensors in tests."""
from dataclasses import fields

import torch

from .core import validate_plan, valid_prefix_slots


def snapshot_kv(pool, plan, max_bytes):
    pages = validate_plan(plan)
    if pool.ndim != 6 or pool.shape[0] != 2 or list(pool.shape[2:4]) != [plan["total_blocks"], plan["block_size"]]:
        raise ValueError("pool geometry differs from plan")
    shape = list(pool.shape)
    shape[2] = len(pages)
    size = pool.element_size()
    for n in shape:
        size *= n
    if size > max_bytes:
        raise ValueError(f"snapshot requires {size} bytes, exceeding cap {max_bytes}")
    saved = torch.zeros(shape, dtype=pool.dtype, device="cpu")
    valid = set(valid_prefix_slots(plan))
    block = plan["block_size"]
    for j, page in enumerate(pages):
        offsets = [i for i in range(block) if page*block+i in valid]
        if offsets:
            if offsets != list(range(offsets[-1]+1)):
                raise ValueError("noncontiguous valid prefix in physical page")
            # Do not transfer uninitialized tails to disk, even transiently.
            saved[:, :, j, :len(offsets)].copy_(pool[:, :, page, :len(offsets)].detach().cpu())
    return {"block_ids": pages, "kv": saved, "pool_shape": list(pool.shape),
            "valid_slots": sorted(valid), "sanitized_uninitialized_tails": True}


def restore_kv(pool, image):
    if list(pool.shape) != image["pool_shape"] or pool.dtype != image["kv"].dtype:
        raise ValueError("replay pool shape/dtype differs from capture")
    pages, saved = image["block_ids"], image["kv"]
    if len(set(pages)) != len(pages) or saved.shape[2] != len(pages):
        raise ValueError("corrupt compact KV snapshot")
    for j, page in enumerate(pages):
        if not 0 <= page < pool.shape[2]:
            raise ValueError("snapshot page outside pool")
        pool[:, :, page].copy_(saved[:, :, j])


def pack_context(context):
    def pack(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().clone()
        if hasattr(x, "__dataclass_fields__"):
            return {f.name: pack(getattr(x, f.name)) for f in fields(x)}
        if x is None or type(x) in (int, float, bool):
            return x
        raise TypeError(f"unsupported context value: {type(x)}")
    return pack(context)


def unpack_context(encoded, device):
    from nanovllm.layers.mixed_attention import MixedAttentionContext
    from nanovllm.utils.context import Context
    def move(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x
    values = {k: move(v) for k, v in encoded.items() if k != "mixed"}
    mixed = encoded["mixed"]
    values["mixed"] = None if mixed is None else MixedAttentionContext(**{k: move(v) for k, v in mixed.items()})
    return Context(**values)


def validate_context(plan, encoded):
    """Check the actual Attention metadata against the CPU logical plan."""
    validate_plan(plan)
    def as_list(x):
        return x.tolist() if isinstance(x, torch.Tensor) else x
    def tables(rows, table):
        data = as_list(table)
        if data is None or len(data) != len(rows):
            raise ValueError("Attention table row count differs")
        for row, actual in zip(rows, data):
            if actual[:len(row["table"])] != row["table"] or any(p != -1 for p in actual[len(row["table"]):]):
                raise ValueError("Attention table differs from logical block table")
    def cumulative(rows, field):
        values = [0]
        for r in rows:
            values.append(values[-1]+(r["b"]-r["a"] if field == "q" else r["end"]))
        return values
    if as_list(encoded["slot_mapping"]) != plan["slots"]:
        raise ValueError("Attention slot mapping differs")
    rows, mixed = plan["rows"], encoded["mixed"]
    if mixed is None:
        if encoded["is_prefill"]:
            if (as_list(encoded["cu_seqlens_q"]) != cumulative(rows, "q")
                    or as_list(encoded["cu_seqlens_k"]) != cumulative(rows, "k")):
                raise ValueError("legacy Prefill context lengths differ")
            if encoded["block_tables"] is not None:
                tables(rows, encoded["block_tables"])
            elif any(r["start"] for r in rows):
                raise ValueError("cached Prefill omitted its page table")
        else:
            if any(r["phase"] != "decode" for r in rows) or as_list(encoded["context_lens"]) != [r["end"] for r in rows]:
                raise ValueError("Decode context lengths/phases differ")
            tables(rows, encoded["block_tables"])
    else:
        d = mixed["num_decode"]
        decode, prefill = rows[:d], rows[d:]
        if (encoded["is_prefill"] or len(decode) != d or not prefill
                or any(r["phase"] != "decode" for r in decode)
                or any(r["phase"] == "decode" for r in prefill)
                or mixed["num_prefill_tokens"] != sum(r["b"]-r["a"] for r in prefill)
                or as_list(mixed["prefill_cu_q"]) != cumulative(prefill, "q")
                or as_list(mixed["prefill_cu_k"]) != cumulative(prefill, "k")
                or mixed["prefill_has_history"] != any(r["start"] > 0 for r in prefill)):
            raise ValueError("Mixed Attention phase/prefix geometry differs")
        if d:
            if as_list(mixed["decode_lens"]) != [r["end"] for r in decode]:
                raise ValueError("Mixed Decode context lengths differ")
            tables(decode, mixed["decode_tables"])
        tables(prefill, mixed["prefill_tables"])


def logit_summary(logits):
    x = logits.detach().float().cpu()
    if x.ndim != 1 or len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError("expected a finite vocabulary vector")
    best = x.topk(2)
    return {"argmax": int(x.argmax()), "top2_ids": best.indices.tolist(),
            "top2_values": best.values.tolist(), "top2_margin": float(best.values[0]-best.values[1]),
            "token_13_minus_382": float(x[13]-x[382]) if len(x) > 382 else None}


def compare_logits(actual, reference):
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    sa, sb = logit_summary(a), logit_summary(b)
    if a.shape != b.shape:
        raise ValueError("vocabulary shapes differ")
    return {"exact": torch.equal(a, b), "same_top1": sa["argmax"] == sb["argmax"],
            "max_abs": float((a-b).abs().max()),
            "relative_l2": float((a-b).norm()/b.norm().clamp_min(1e-12)),
            "actual": sa, "reference": sb,
            "acceptance_override": False}
