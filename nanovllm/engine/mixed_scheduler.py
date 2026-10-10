"""Decode-first fixed-chunk policy. No SLO controller, aging or watermark yet."""
from collections import deque

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.engine.batch_metadata import ScheduledRequest, build_batch

"""
self.waiting：等待队列，刚加入但还没开始处理的请求
self.running：正在执行的请求
self.pending：当前已经调度出但还没提交完成的 batch，已经选中并准备发给模型的一批请求
"""
class MixedScheduler(Scheduler):
    def __init__(self, config, chunk_size=512):
        super().__init__(config)
        if chunk_size <= 0 or self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("chunk/budget/max_num_seqs must be positive")
        self.chunk_size = chunk_size
        self.max_model_len = config.max_model_len
        self.sequences = {}
        self.pending = None
        self.preemption_count = 0

    def add(self, seq):
        # Conservative per-request admission check, not aggregate reservation.
        total = len(seq) + seq.max_tokens - 1
        if not len(seq) or seq.max_tokens <= 0:
            raise ValueError("empty prompt or invalid output length")
        if total > self.max_model_len or total > len(self.block_manager.blocks) * self.block_size:
            raise ValueError("request cannot fit model length or physical KV capacity")
        if seq.seq_id in self.sequences:
            raise ValueError("duplicate seq_id")
        self.sequences[seq.seq_id] = seq
        self.waiting.append(seq)

    def preempt(self, seq):
        if self.pending is not None:
            raise RuntimeError("cannot preempt an in-flight batch")
        self.running.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.preemption_count += 1

    def schedule(self):
        if self.pending is not None:
            raise RuntimeError("previous batch has not been committed")
        rows, selected = [], set()
        remaining = self.max_num_batched_tokens
        # Round-robin among decode requests when max_num_seqs/budget truncates.
        # 先处理 running 队列中的 decode 请求
        for seq in list(self.running):
            if remaining == 0 or len(rows) == self.max_num_seqs:
                break
            if seq not in self.running:
                continue  # already preempted as a victim
            if seq.num_cached_tokens != len(seq) - 1:
                raise RuntimeError("RUNNING request is not decode-ready")
            while not self.block_manager.can_append(seq):
                victim = next((s for s in reversed(self.running)
                               if s.seq_id not in selected and s is not seq), None)
                if victim is None:
                    self.preempt(seq)
                    break
                self.preempt(victim)
            else:
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                rows.append(ScheduledRequest(seq.seq_id, "decode", seq.num_cached_tokens, 1, len(seq)))
                selected.add(seq.seq_id)
                remaining -= 1

        # Bounded scan: a blocked head must not hide an already allocated chunk.
        # This is a correctness/liveness rule, not the later fairness policy.
        # 再处理 waiting 队列中的 prefill 请求
        for seq in list(self.waiting):
            if remaining == 0 or len(rows) == self.max_num_seqs:
                break
            if not seq.block_table:
                cached = self.block_manager.can_allocate(seq)
                if cached == -1:
                    continue
                self.block_manager.allocate(seq, cached)
            start = seq.num_cached_tokens
            count = min(len(seq) - start, self.chunk_size, remaining)
            if count <= 0:
                raise RuntimeError("prefill must compute at least one token")
            seq.num_scheduled_tokens = count
            seq.is_prefill = True
            phase = "recompute" if seq.num_completion_tokens else "prefill"
            rows.append(ScheduledRequest(seq.seq_id, phase, start, count, len(seq)))
            selected.add(seq.seq_id)
            remaining -= count

        if not rows:
            raise RuntimeError("nonempty scheduler made no progress; check KV allocation")
        # 最后构造 batch
        batch = build_batch(rows, self.sequences, self.max_num_batched_tokens, self.block_size)
        self.pending = batch
        return batch
    
    # 模型返回后更新状态
    def postprocess(self, batch, sampled_tokens):
        if batch is not self.pending:
            raise RuntimeError("wrong batch or repeated commit")
        if set(sampled_tokens) != set(batch.sample_seq_ids):
            raise ValueError("missing or extra sampled request")
        # Validate all rows before changing any logical state.
        for row in batch.requests:
            seq = self.sequences[row.seq_id]
            if seq.num_cached_tokens != row.start_pos or len(seq) != row.logical_tokens:
                raise RuntimeError("stale batch")
        for row in batch.requests:
            seq = self.sequences[row.seq_id]
            # Forward has returned: publish only full blocks whose KV is now valid.
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens = row.end_pos
            seq.computed_high_water = max(seq.computed_high_water, row.end_pos)
            seq.num_scheduled_tokens = 0
            if not row.should_sample:
                continue
            token = int(sampled_tokens[seq.seq_id])
            seq.append_token(token)
            if seq.status == SequenceStatus.WAITING:
                self.waiting.remove(seq)
                self.running.append(seq)
                seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            if ((not seq.ignore_eos and token == self.eos)
                    or seq.num_completion_tokens >= seq.max_tokens):
                self.running.remove(seq)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
        # Selected surviving decode requests move to the tail.
        for row in batch.requests:
            seq = self.sequences[row.seq_id]
            if row.phase == "decode" and seq.status == SequenceStatus.RUNNING:
                self.running.remove(seq)
                self.running.append(seq)
        self.pending = None

    def assert_consistent(self):
        queues = list(self.waiting) + list(self.running)
        if len({s.seq_id for s in queues}) != len(queues):
            raise AssertionError("request appears in multiple queue positions")
        if {s.seq_id for s in queues} != {i for i, s in self.sequences.items() if not s.is_finished}:
            raise AssertionError("request lost from queues")
        refs = [0] * len(self.block_manager.blocks)
        for seq in self.sequences.values():
            for block in seq.block_table:
                refs[block] += 1
        for block, expected in zip(self.block_manager.blocks, refs):
            if block.ref_count != expected:
                raise AssertionError("block reference leak")
        used = {i for i, count in enumerate(refs) if count}
        if used != self.block_manager.used_block_ids:
            raise AssertionError("used block set mismatch")
        free = list(self.block_manager.free_block_ids)
        if len(free) != len(set(free)) or set(free) != set(range(len(refs))) - used:
            raise AssertionError("free block set mismatch")
