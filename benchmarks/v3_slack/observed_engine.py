"""Expose per-request TTFT deadlines and ITL clocks to the opt-in Slack scheduler."""
from benchmarks.v1.observed_engine import ObservedMixedEngine
from nanovllm.engine.slack_scheduler import SlackScheduler


class ObservedSlackEngine(ObservedMixedEngine):
    def __init__(self, model, predictor, *, safety_margin_ms=15.0,
                 fallback_chunk=256, hysteresis=False, rise_steps=2,
                 rise_margin_ms=10.0, aging=False, aging_warning_ms=300.0,
                 aging_hard_ms=800.0, kv_guard=False,
                 kv_low_free_blocks=8, kv_high_free_blocks=16, **kwargs):
        super().__init__(model, observation="minimal", sampler="greedy",
                         chunk_size=fallback_chunk, **kwargs)
        self.scheduler = SlackScheduler(self.config, predictor,
            safety_margin_ms=safety_margin_ms, fallback_chunk=fallback_chunk,
            hysteresis=hysteresis, rise_steps=rise_steps,
            rise_margin_ms=rise_margin_ms,
            aging=aging, aging_warning_ms=aging_warning_ms,
            aging_hard_ms=aging_hard_ms,
            kv_guard=kv_guard, kv_low_free_blocks=kv_low_free_blocks,
            kv_high_free_blocks=kv_high_free_blocks,
            clock=self.now)
        self.observation = "full"
        preempt = self.scheduler.preempt

        def observed_preempt(seq):
            timestamp = self.now()
            preempt(seq)
            record = self.by_seq[seq.seq_id]
            record.preemptions += 1
            self.total_preemptions += 1
            record.waiting_spans.append({"start_ns": timestamp, "end_ns": None,
                                         "reason": "preempted"})

        self.scheduler.preempt = observed_preempt

    def submit(self, spec, record):
        super().submit(spec, record)
        self.scheduler.register_request(record.seq_id, record.itl_slo_ms,
                                        record.ttft_slo_ms, record.arrival_ns,
                                        record.enqueue_ns)

    def step(self):
        result = super().step()
        row = self.steps[-1]
        # The inherited observer records each interval from integer clocks.
        # Reconstruct the postprocess endpoint before its diagnostic logging.
        visible = row["start_ns"] + round((row["schedule_ms"] + row["run_ms"] +
                                           row["postprocess_ms"]) * 1e6)
        self.scheduler.observe_visible(self.scheduler.last_batch, visible)
        row.update(self.scheduler.last_decision)
        row["min_slack"] = self.scheduler.last_decision["min_slack_ms"]
        return result
