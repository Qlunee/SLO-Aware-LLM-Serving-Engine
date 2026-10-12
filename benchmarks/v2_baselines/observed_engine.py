"""Reuse V1 forward/observation; opt in to a new scheduler before submission."""
from benchmarks.v1.observed_engine import ObservedMixedEngine
from nanovllm.engine.fixed_scheduler import FixedScheduler


class ObservedFixedEngine(ObservedMixedEngine):
    def __init__(self, model, policy="chunk", chunk_size=256,
                 max_prefill_wait_ms=1000, **kwargs):
        # Avoid binding V1's preemption observer to the scheduler being replaced.
        super().__init__(model, observation="minimal", sampler="greedy",
                         chunk_size=chunk_size, **kwargs)
        self.scheduler = FixedScheduler(self.config, chunk_size, policy,
                                        max_prefill_wait_ms, clock=self.now)
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

    def step(self):
        result = super().step()
        if self.steps:
            self.steps[-1].update(self.scheduler.last_decision)
        return result
