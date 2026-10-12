"""Opt-in V2 fixed policies; the frozen V1 scheduler is not changed."""
import math
from time import perf_counter_ns

from nanovllm.engine.mixed_scheduler import MixedScheduler
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.engine.batch_metadata import ScheduledRequest, build_batch


class FixedScheduler(MixedScheduler):
    def __init__(self, config, chunk_size=256, policy="chunk",
                 max_prefill_wait_ms=1000, clock=perf_counter_ns):
        super().__init__(config, chunk_size)
        if policy not in ("chunk", "decode_first", "decode_first_wait"):
            raise ValueError("unknown fixed policy")
        if not math.isfinite(max_prefill_wait_ms) or max_prefill_wait_ms <= 0:
            raise ValueError("max_prefill_wait_ms must be positive and finite")
        self.policy = policy
        self.clock = clock
        self.max_prefill_wait_ms = max_prefill_wait_ms
        self.wait_since = {}
        self.last_decision = {}

    def add(self, seq):
        super().add(seq)
        self.wait_since[seq.seq_id] = self.clock()

    def preempt(self, seq):
        super().preempt(seq)
        self.wait_since[seq.seq_id] = self.clock()

    def schedule(self):
        if self.pending is not None:
            raise RuntimeError("previous batch has not been committed")
        now = self.clock()
        oldest = max(((now-self.wait_since[s.seq_id])/1e6 for s in self.waiting), default=0)
        self.last_decision = {"policy": self.policy, "oldest_prefill_wait_ms": oldest,
                              "forced_prefill_seq_id": None, "wait_protection_conflict": None}
        if self.policy == "chunk" or not self.running:
            return super().schedule()
        rows, selected = [], set()
        remaining = self.max_num_batched_tokens

        def prefill(seq, budget):
            if not seq.block_table:
                cached = self.block_manager.can_allocate(seq)
                if cached == -1:
                    return 0
                self.block_manager.allocate(seq, cached)
            start = seq.num_cached_tokens
            count = min(len(seq)-start, self.chunk_size, budget)
            if count <= 0:
                raise RuntimeError("prefill must make progress")
            seq.num_scheduled_tokens, seq.is_prefill = count, True
            phase = "recompute" if seq.num_completion_tokens else "prefill"
            rows.append(ScheduledRequest(seq.seq_id, phase, start, count, len(seq)))
            selected.add(seq.seq_id)
            return count

        if self.policy == "decode_first_wait":
            eligible = sorted((s for s in self.waiting
                if now-self.wait_since[s.seq_id] >= self.max_prefill_wait_ms*1e6),
                key=lambda s: (self.wait_since[s.seq_id], s.seq_id))
            if eligible:
                # One forced chunk per Step; reserve at least one Decode token
                # if both the token and sequence limits permit a mixed batch.
                reserve = 1 if self.max_num_seqs > 1 and remaining > 1 else 0
                for seq in eligible:
                    count = prefill(seq, remaining-reserve)
                    if count:
                        remaining -= count
                        self.last_decision["forced_prefill_seq_id"] = seq.seq_id
                        break
                else:
                    self.last_decision["wait_protection_conflict"] = "KV_capacity"

        for seq in list(self.running):
            if not remaining or len(rows) == self.max_num_seqs:
                break
            if seq not in self.running:
                continue
            if seq.num_cached_tokens != len(seq)-1:
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
                seq.num_scheduled_tokens, seq.is_prefill = 1, False
                rows.append(ScheduledRequest(seq.seq_id, "decode", seq.num_cached_tokens, 1, len(seq)))
                selected.add(seq.seq_id)
                remaining -= 1

        if not rows:
            # All Decode candidates may have been preempted. Recompute rather
            # than raising solely because strict priority hid the waiting queue.
            return super().schedule()
        # Allocation/selection may prioritize an overdue chunk, but the frozen
        # attention metadata contract requires Decode rows before Prefill rows.
        rows.sort(key=lambda row: row.phase != "decode")
        batch = build_batch(rows, self.sequences, self.max_num_batched_tokens, self.block_size)
        self.pending = batch
        return batch

    def postprocess(self, batch, sampled_tokens):
        super().postprocess(batch, sampled_tokens)
        now = self.clock()
        for row in batch.requests:
            seq = self.sequences[row.seq_id]
            if seq.status == SequenceStatus.WAITING:
                self.wait_since[seq.seq_id] = now
            else:
                self.wait_since.pop(seq.seq_id, None)
