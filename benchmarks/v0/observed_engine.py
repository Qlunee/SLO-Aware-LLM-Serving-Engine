## 给nano-vLLM加的观测适配层：让原引擎继续处理请求，同时记录“请求等了多久、Token何时产生、GPU每一步处理了什么”。
"""
它主要做四件事：
- 记录请求时序：实际入队、首次调度、每个输出Token和完成时间，供metrics.py计算TTFT、ITL。
- 记录Step信息：当前是Prefill还是Decode、处理多少Token、等待队列长度和KV Block占用。
- 记录抢占与重算：请求何时被抢占，以及恢复后实际重复计算了多少Token。
- 提供Greedy采样适配：使用argmax，方便比较观测前后的输出是否一致。
"""
from __future__ import annotations

from time import perf_counter_ns

import torch
from torch import nn

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

from .metrics import RequestRecord
from .workload import RequestSpec
from .accounting import merge_interval


class GreedySampler(nn.Module):
    def forward(self, logits, temperatures):
        return logits.argmax(dim=-1)


class ObservedEngine(LLMEngine):
    def __init__(
        self,
        model: str,
        observation: str,
        sampler: str,
        **kwargs,
    ):
        self._closed = False
        super().__init__(model, **kwargs)

        self.observation = observation
        self.sampler_mode = sampler
        if sampler == "greedy":
            self.model_runner.sampler = GreedySampler()

        self.origin_ns = 0
        self.by_seq: dict[int, RequestRecord] = {}
        self.sequences = {}
        self.history: dict[int, list[tuple[int, int]]] = {}
        self.steps: list[dict] = []
        self._pending_record = None
        self.step_count = 0
        self.total_preemptions = 0

        original_add = self.scheduler.add

        def observed_add(seq):
            original_add(seq)
            record = self._pending_record
            if record is None:
                raise RuntimeError("V0请求必须通过submit提交")

            record.seq_id = seq.seq_id
            record.enqueue_ns = self.now()
            record.status = "queued"
            self.by_seq[seq.seq_id] = record
            self.sequences[seq.seq_id] = seq

            if observation == "full":
                record.preemptions = 0
                record.recomputed_tokens = 0
                record.waiting_spans.append({
                    "start_ns": record.enqueue_ns,
                    "end_ns": None,
                    "reason": "initial_prefill",
                })
                self.history[seq.seq_id] = []

        self.scheduler.add = observed_add

        if observation == "full":
            original_preempt = self.scheduler.preempt

            def observed_preempt(seq):
                timestamp = self.now()
                original_preempt(seq)

                record = self.by_seq[seq.seq_id]
                record.preemptions += 1
                self.total_preemptions += 1
                record.waiting_spans.append({
                    "start_ns": timestamp,
                    "end_ns": None,
                    "reason": "preempted",
                })

            self.scheduler.preempt = observed_preempt

    def start_clock(self):
        self.origin_ns = perf_counter_ns()

    def now(self) -> int:
        return perf_counter_ns() - self.origin_ns

    def submit(self, spec: RequestSpec, record: RequestRecord):
        self._pending_record = record
        try:
            super().add_request(
                spec.prompt_token_ids,
                SamplingParams(
                    temperature=1.0,
                    max_tokens=spec.max_output_tokens,
                    ignore_eos=True,
                ),
            )
        finally:
            self._pending_record = None

    def step(self):
        self.step_count += 1

        if self.observation == "native":
            outputs, num_tokens = super().step()
            timestamp = self.now()
            for seq_id, output_ids in outputs:
                record = self.by_seq[seq_id]
                record.output_token_ids = list(output_ids)
                record.finish_ns = timestamp
                record.status = "completed"
            return outputs, num_tokens

        start_ns = self.now()
        preemptions_before = self.total_preemptions
        seqs, is_prefill = self.scheduler.schedule()
        scheduled_ns = self.now()

        before_output_counts = {
            seq.seq_id: seq.num_completion_tokens for seq in seqs
        }

        # 必须在postprocess清零scheduled和释放Block之前保存。
        execution_spans = [
            (
                seq.seq_id,
                seq.num_cached_tokens,
                seq.num_cached_tokens + seq.num_scheduled_tokens,
            )
            for seq in seqs
        ]
        scheduled_tokens = sum(
            seq.num_scheduled_tokens for seq in seqs
        )
        context_sum = sum(len(seq) for seq in seqs)
        context_max = max(len(seq) for seq in seqs)

        for seq in seqs:
            record = self.by_seq[seq.seq_id]
            record.status = "active"
            if record.first_scheduled_ns is None:
                record.first_scheduled_ns = scheduled_ns

            if self.observation == "full":
                for span in reversed(record.waiting_spans):
                    if span["end_ns"] is None:
                        span["end_ns"] = scheduled_ns
                        break

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        run_end_ns = self.now()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        token_visible_ns = self.now()

        outputs = []
        for seq in seqs:
            record = self.by_seq[seq.seq_id]
            previous = before_output_counts[seq.seq_id]
            added = seq.num_completion_tokens - previous

            if added not in (0, 1):
                raise RuntimeError("每请求每Step输出Token数异常")

            if added == 1:
                record.token_times_ns.append(token_visible_ns)
                record.output_token_ids.append(seq.last_token)

            if seq.is_finished:
                record.status = "completed"
                record.finish_ns = token_visible_ns
                outputs.append((seq.seq_id, seq.completion_token_ids))

        if self.observation == "full":
            recomputed = 0
            for seq_id, a, b in execution_spans:
                overlap, merged = merge_interval(
                    self.history[seq_id], a, b
                )
                self.history[seq_id] = merged
                self.by_seq[seq_id].recomputed_tokens += overlap
                recomputed += overlap

            # 部分Prefill执行后仍在waiting队列，开始新的等待段。
            for seq in seqs:
                if not seq.is_finished and seq.status.name == "WAITING":
                    self.by_seq[seq.seq_id].waiting_spans.append({
                        "start_ns": token_visible_ns,
                        "end_ns": None,
                        "reason": "between_prefill_chunks",
                    })

            manager = self.scheduler.block_manager
            runner = self.model_runner
            graph_mode = (
                "cuda_graph"
                if not is_prefill
                and not runner.enforce_eager
                and len(seqs) <= 512
                else "eager"
            )
            end_ns = self.now()
            self.steps.append({
                "step_id": self.step_count,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": (end_ns - start_ns) / 1e6,
                "schedule_ms": (scheduled_ns - start_ns) / 1e6,
                "run_ms": (run_end_ns - scheduled_ns) / 1e6,
                "postprocess_ms": (token_visible_ns - run_end_ns) / 1e6,
                "phase": "prefill" if is_prefill else "decode",
                "num_requests": len(seqs),
                "prefill_tokens": scheduled_tokens if is_prefill else 0,
                "num_decode": 0 if is_prefill else len(seqs),
                "sum_context": context_sum,
                "max_context": context_max,
                "used_blocks": len(manager.used_block_ids),
                "total_blocks": len(manager.blocks),
                "waiting": len(self.scheduler.waiting),
                "running": len(self.scheduler.running),
                "recomputed_tokens": recomputed,
                "preemptions": self.total_preemptions - preemptions_before,
                "token_budget": self.scheduler.max_num_batched_tokens,
                "kv_dtype": str(runner.kv_cache.dtype),
                "kv_payload_bytes": runner.kv_cache.numel() * runner.kv_cache.element_size(),
                "scale_bytes": 0,
                # 按冻结版本run_model的实际分支条件确定。
                "graph_mode": graph_mode,
                "min_slack": None,
                "predicted_ms": None,
                "workspace_bytes": None,
            })

        num_tokens = scheduled_tokens if is_prefill else -len(seqs)
        return outputs, num_tokens

    def snapshot_outputs(self):
        for seq_id, seq in self.sequences.items():
            self.by_seq[seq_id].output_token_ids = list(
                seq.completion_token_ids
            )

    def close_waiting_spans(self, timestamp: int):
        for record in self.by_seq.values():
            for span in record.waiting_spans:
                if span["end_ns"] is None:
                    span["end_ns"] = timestamp

    def exit(self):
        if self._closed:
            return
        self._closed = True
        super().exit()
