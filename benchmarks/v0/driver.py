"""Open-loop driver, with an engine/clock seam for CPU regression."""
from __future__ import annotations

import bisect
import math
import time
import traceback
from dataclasses import dataclass

from .metrics import RequestRecord


@dataclass
class DriverResult:
    records: list[RequestRecord]
    backlog: list[dict]
    end_ns: int
    deadline_ns: int
    error: str | None


def rejection_reason(spec, config):
    total = len(spec.prompt_token_ids) + spec.max_output_tokens
    blocks = (total + config.kvcache_block_size - 1) // config.kvcache_block_size
    if total > config.max_model_len:
        return "request_exceeds_max_model_len"
    if blocks > config.num_kvcache_blocks:
        return "request_exceeds_kv_capacity"
    if any(t < 0 or t >= config.hf_config.vocab_size for t in spec.prompt_token_ids):
        return "invalid_prompt_token_id"
    return None


def execute_trace(engine, config, meta, specs, drain_s, observation, sleep=time.sleep):
    if not math.isfinite(drain_s) or drain_s < 0:
        raise ValueError("drain_s must be finite and nonnegative")
    measurement_end = round((meta["warmup_s"] + meta["measurement_s"]) * 1e9)
    deadline = measurement_end + round(drain_s * 1e9)
    records = [RequestRecord(
        request_id=s.request_id, arrival_ns=s.arrival_ns,
        prompt_tokens=len(s.prompt_token_ids), max_output_tokens=s.max_output_tokens,
        ttft_slo_ms=s.ttft_slo_ms, itl_slo_ms=s.itl_slo_ms,
    ) for s in specs]
    # Validation is outside the measured window, identical across modes.
    rejections = [rejection_reason(s, config) for s in specs]
    arrivals = [s.arrival_ns for s in specs]
    backlog = []
    next_index = 0
    error = None

    def sample(timestamp):
        arrived = bisect.bisect_right(arrivals, timestamp)
        unsubmitted = max(0, arrived - next_index)
        waiting, running = len(engine.scheduler.waiting), len(engine.scheduler.running)
        backlog.append({
            "time_ns": timestamp, "planned_arrived": arrived,
            "submitted_or_rejected": next_index,
            "arrived_not_submitted": unsubmitted,
            "waiting": waiting, "running": running,
            "backlog_total": unsubmitted + waiting + running,
        })

    engine.start_clock()
    try:
        while engine.now() < deadline:
            timestamp = engine.now()
            while next_index < len(specs) and arrivals[next_index] <= timestamp:
                if engine.now() >= deadline:
                    break
                record = records[next_index]
                reason = rejections[next_index]
                if reason:
                    record.status, record.error = "rejected", reason
                else:
                    engine.submit(specs[next_index], record)
                next_index += 1

            if observation == "full":
                sample(engine.now())
            if engine.now() >= deadline:
                break
            if not engine.is_finished():
                engine.step()
                if observation == "full":
                    sample(engine.now())
                continue

            timestamp = engine.now()
            if next_index == len(specs) and timestamp >= measurement_end:
                break
            next_event = arrivals[next_index] if next_index < len(specs) else measurement_end
            sleep_s = min(0.01, max(0, min(next_event, deadline) - timestamp) / 1e9)
            if sleep_s:
                sleep(sleep_s)
    except Exception:
        error = traceback.format_exc()

    end_ns = engine.now()
    engine.snapshot_outputs()
    engine.close_waiting_spans(end_ns)
    for record in records:
        if record.status == "completed":
            if record.finish_ns >= deadline:
                record.status = "timeout"
                record.error = "completed_at_or_after_drain_deadline"
        elif record.status != "rejected":
            record.status = "failed" if error else "unfinished"
    if observation == "full":
        sample(end_ns)
    return DriverResult(records, backlog, end_ns, deadline, error)
