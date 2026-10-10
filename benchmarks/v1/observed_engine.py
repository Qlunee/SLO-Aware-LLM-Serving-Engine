"""V1 adapter for the unchanged V0 open-loop driver and RequestRecord contract."""
from time import perf_counter_ns

from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
from nanovllm.sampling_params import SamplingParams

from benchmarks.v0.accounting import merge_interval


class ObservedMixedEngine(MixedLLMEngine):
    def __init__(self, model, observation="minimal", sampler="greedy", **kwargs):
        if observation not in {"native", "minimal", "full"}:
            raise ValueError("unknown observation mode")
        if sampler not in {"greedy", "original"}:
            raise ValueError("unknown sampler")
        super().__init__(model, greedy=sampler == "greedy", **kwargs)
        self.observation = observation
        self.origin_ns = 0
        self.by_seq = {}
        self.history = {}
        self.steps = []
        self.step_count = 0
        self.total_preemptions = 0
        original_preempt = self.scheduler.preempt

        def observed_preempt(seq):
            timestamp = self.now()
            original_preempt(seq)
            if observation == "full":
                record = self.by_seq[seq.seq_id]
                record.preemptions += 1
                self.total_preemptions += 1
                record.waiting_spans.append({"start_ns": timestamp, "end_ns": None,
                                             "reason": "preempted"})

        if observation == "full":
            self.scheduler.preempt = observed_preempt

    def start_clock(self):
        self.origin_ns = perf_counter_ns()

    def now(self):
        return perf_counter_ns() - self.origin_ns

    def submit(self, spec, record):
        seq_id = self.add_request(spec.prompt_token_ids, SamplingParams(
            temperature=1.0, max_tokens=spec.max_output_tokens, ignore_eos=True))
        record.seq_id = seq_id
        record.enqueue_ns = self.now()
        record.status = "queued"
        self.by_seq[seq_id] = record
        if self.observation == "full":
            record.preemptions = record.recomputed_tokens = 0
            record.waiting_spans.append({"start_ns": record.enqueue_ns, "end_ns": None,
                                         "reason": "initial_prefill"})
            self.history[seq_id] = []

    def step(self):
        self.step_count += 1
        if self.observation == "native":
            outputs, counts = super().step()
            timestamp = self.now()
            for seq_id, token_ids in outputs:
                record = self.by_seq[seq_id]
                record.output_token_ids = list(token_ids)
                record.finish_ns, record.status = timestamp, "completed"
            return outputs, counts
        start = self.now()
        previous_preemptions = self.total_preemptions
        batch = self.scheduler.schedule()
        scheduled = self.now()
        for row in batch.requests:
            record = self.by_seq[row.seq_id]
            record.status = "active"
            if record.first_scheduled_ns is None:
                record.first_scheduled_ns = scheduled
            if self.observation == "full":
                for span in reversed(record.waiting_spans):
                    if span["end_ns"] is None:
                        span["end_ns"] = scheduled
                        break
        # No logits copy, TP digest collective or per-step CPU invariant scan.
        result = self.model_runner.call("run_mixed", batch, False, False, self.greedy)
        run_end = self.now()
        self.scheduler.postprocess(batch, result["tokens"])
        visible = self.now()
        outputs = []
        for row in batch.requests:
            seq = self.scheduler.sequences[row.seq_id]
            record = self.by_seq[row.seq_id]
            if row.should_sample:
                record.token_times_ns.append(visible)
                record.output_token_ids.append(seq.last_token)
            if seq.is_finished:
                record.finish_ns, record.status = visible, "completed"
                outputs.append((seq.seq_id, list(seq.completion_token_ids)))
        if self.observation == "full":
            recomputed = 0
            for row in batch.requests:
                overlap, merged = merge_interval(self.history[row.seq_id], row.start_pos, row.end_pos)
                self.history[row.seq_id] = merged
                self.by_seq[row.seq_id].recomputed_tokens += overlap
                recomputed += overlap
                seq = self.scheduler.sequences[row.seq_id]
                if not seq.is_finished and seq.status.name == "WAITING":
                    self.by_seq[row.seq_id].waiting_spans.append({
                        "start_ns": visible, "end_ns": None, "reason": "between_prefill_chunks"})
            manager, runner = self.scheduler.block_manager, self.model_runner
            contexts = [r.end_pos for r in batch.requests]
            end = self.now()
            phase = ("mixed" if batch.num_decode_tokens and batch.num_prefill_tokens
                     else "decode" if batch.pure_decode else "prefill")
            self.steps.append({
                "step_id": self.step_count, "start_ns": start, "end_ns": end,
                "duration_ms": (end - start) / 1e6, "schedule_ms": (scheduled - start) / 1e6,
                "run_ms": (run_end - scheduled) / 1e6, "postprocess_ms": (visible - run_end) / 1e6,
                "phase": phase, "num_requests": len(batch.requests),
                "prefill_tokens": batch.num_prefill_tokens, "num_decode": batch.num_decode_tokens,
                "sum_context": sum(contexts), "max_context": max(contexts),
                "used_blocks": len(manager.used_block_ids), "total_blocks": len(manager.blocks),
                "waiting": len(self.scheduler.waiting), "running": len(self.scheduler.running),
                "recomputed_tokens": recomputed,
                "preemptions": self.total_preemptions - previous_preemptions,
                "token_budget": batch.token_budget, "chunk_size": self.scheduler.chunk_size,
                "kv_dtype": str(runner.kv_cache.dtype),
                "kv_payload_bytes": runner.kv_cache.numel() * runner.kv_cache.element_size(),
                "scale_bytes": 0, "graph_mode": result["graph_mode"],
                "min_slack": None, "predicted_ms": None, "workspace_bytes": None,
            })
        return outputs, {"prefill_tokens": batch.num_prefill_tokens,
                         "decode_tokens": batch.num_decode_tokens}

    def snapshot_outputs(self):
        for seq_id, record in self.by_seq.items():
            record.output_token_ids = list(self.scheduler.sequences[seq_id].completion_token_ids)

    def close_waiting_spans(self, timestamp):
        for record in self.by_seq.values():
            for span in record.waiting_spans:
                if span["end_ns"] is None:
                    span["end_ns"] = timestamp
