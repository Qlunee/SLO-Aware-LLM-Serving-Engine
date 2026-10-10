import hashlib
import pickle

import torch
import torch.distributed as dist

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.mixed_attention import MixedAttentionContext
from nanovllm.utils.context import reset_context, set_context


class MixedModelRunner(ModelRunner):
    def __init__(self, config, rank, event, kv_block_cap=None):
        self.kv_block_cap = kv_block_cap
        super().__init__(config, rank, event)

    def allocate_kv_cache(self):
        # Use the minimum feasible capacity across ranks, not rank 0's alone.
        config, hf = self.config, self.config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]
        heads = hf.num_key_value_heads // self.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * hf.num_hidden_layers * self.block_size * heads * dim * hf.dtype.itemsize
        count = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if self.kv_block_cap is not None:
            count = min(count, self.kv_block_cap)
        if self.world_size > 1:
            capacity = torch.tensor(count, dtype=torch.int64, device="cuda")
            dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
            count = int(capacity.item())
        config.num_kvcache_blocks = count
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError("no KV blocks available on at least one TP rank")
        self.kv_cache = torch.empty(
            2, hf.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, heads, dim,
        )
        layer = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer]
                module.v_cache = self.kv_cache[1, layer]
                layer += 1

    def write_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args], protocol=pickle.HIGHEST_PROTOCOL)
        if len(data) + 4 > len(self.shm.buf):
            raise ValueError("TP metadata exceeds frozen 1MiB shared-memory buffer")
        self.shm.buf[:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4:4 + len(data)] = data
        for event in self.event:
            event.set()

    @staticmethod
    def device_tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).cuda(non_blocking=True)

    def tables(self, rows):
        width = max(map(len, rows))
        return self.device_tensor([list(r) + [-1] * (width - len(r)) for r in rows])

    def prepare_mixed(self, batch):
        ids = self.device_tensor(batch.input_ids, torch.int64)
        positions = self.device_tensor(batch.positions, torch.int64)
        slots = self.device_tensor(batch.slot_mapping)
        d = batch.num_decode_tokens
        if batch.pure_decode:
            set_context(False, slot_mapping=slots,
                        context_lens=self.device_tensor([r.end_pos for r in batch.requests]),
                        block_tables=self.tables(batch.block_tables))
            return ids, positions
        prefill = batch.requests[d:]
        cu_q, cu_k = [0], [0]
        for row in prefill:
            cu_q.append(cu_q[-1] + row.num_scheduled_tokens)
            cu_k.append(cu_k[-1] + row.end_pos)
        meta = MixedAttentionContext(
            d, batch.num_prefill_tokens,
            self.device_tensor([r.end_pos for r in batch.requests[:d]]) if d else None,
            self.tables(batch.block_tables[:d]) if d else None,
            self.device_tensor(cu_q), self.device_tensor(cu_k),
            self.tables(batch.block_tables[d:]),
            max(r.num_scheduled_tokens for r in prefill), max(r.end_pos for r in prefill),
            any(r.start_pos > 0 for r in prefill),
        )
        # False prevents the legacy LMHead from selecting prefill rows again.
        set_context(False, slot_mapping=slots, mixed=meta)
        return ids, positions

    @torch.inference_mode()
    def run_mixed(self, batch, return_logits=False, verify_tp_plan=False, greedy=True):
        batch.validate()
        if verify_tp_plan and self.world_size > 1:
            digest = hashlib.sha256(pickle.dumps(batch, protocol=4)).hexdigest()
            all_digests = [None] * self.world_size
            dist.all_gather_object(all_digests, digest)
            if len(set(all_digests)) != 1:
                raise RuntimeError("TP ranks received different batch plans")
        try:
            ids, positions = self.prepare_mixed(batch)
            if batch.pure_decode:
                # 纯 decode 场景因为请求很规律，所以尝试 cuda_graph
                hit = (not self.enforce_eager and len(ids) <= 512
                       and any(b >= len(ids) for b in self.graph_bs))
                # is_prefill=True forces eager if there is no fitting graph bucket.
                logits = self.run_model(ids, positions, not hit)
                graph_mode = "cuda_graph" if hit else "eager"
            else:
                # 混合 batch 里有 prefill 和 decode 混在一起，这类 shape 往往不稳定，因此走eager模式，避免 cuda graph capture 失败
                hidden = self.model(ids, positions)  # exactly ONE full-model forward
                if batch.sample_indices:
                    indices = self.device_tensor(batch.sample_indices, torch.int64)
                    logits = self.model.compute_logits(hidden.index_select(0, indices))
                else:
                    # All ranks skip LMHead together. No pointless all-vocabulary projection.
                    logits = None
                graph_mode = "eager"
            if self.rank != 0:
                return None
            if batch.sample_indices:
                if greedy:
                    tokens = logits.argmax(dim=-1).tolist()
                else:
                    temperatures = self.device_tensor(batch.temperatures, torch.float32)
                    tokens = self.sampler(logits, temperatures).tolist()
            else:
                tokens = []
                # Intermediate chunks still need a CPU-visible completion boundary.
                # Otherwise block hashes could be published before GPU KV writes finish.
                torch.cuda.synchronize()
            return {
                "tokens": dict(zip(batch.sample_seq_ids, tokens)),
                "logits": logits.float().cpu() if return_logits and logits is not None else None,
                "graph_mode": graph_mode,
            }
        finally:
            reset_context()
