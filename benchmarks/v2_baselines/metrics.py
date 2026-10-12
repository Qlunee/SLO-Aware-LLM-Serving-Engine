"""Explicit cohort/window scopes; incomplete requests are never discarded."""
import statistics

from benchmarks.v0.metrics import percentiles, request_metrics, summarize
from .workload import validate_trace


def analyze_online(trace, records, steps, backlog, end_ns, starvation_ms):
    specs = validate_trace(trace)
    by_id = {r.request_id: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {s.request_id for s in specs}:
        raise ValueError("records must include every unique planned request")
    for s in specs:
        r = by_id[s.request_id]
        if (r.arrival_ns != s.arrival_ns or r.prompt_tokens != len(s.prompt_token_ids)
                or r.max_output_tokens != s.max_output_tokens
                or r.ttft_slo_ms != s.ttft_slo_ms or r.itl_slo_ms != s.itl_slo_ms):
            raise ValueError("record differs from frozen trace")
        if len(r.output_token_ids) != len(r.token_times_ns):
            raise ValueError("visible token count and timestamps disagree")
        if r.status == "completed" and len(r.output_token_ids) != r.max_output_tokens:
            raise ValueError("completed request has wrong fixed output length")
    start = round(trace["warmup_s"]*1e9)
    end = round((trace["warmup_s"]+trace["measurement_s"])*1e9)
    base = summarize(records, start, end, True)
    waits, starved = [], set()
    tpots = []
    for r in records:
        tpot = request_metrics(r)["tpot_ms"]
        if r.status == "completed" and tpot is not None:
            tpots.append(tpot)
        for span in r.waiting_spans:
            a, b = span["start_ns"], span["end_ns"]
            b = end_ns if b is None else b
            if not 0 <= a <= b <= end_ns:
                raise ValueError("invalid waiting span")
            duration = (b-a)/1e6
            waits.append(duration)
            if duration >= starvation_ms:
                starved.add(r.request_id)
    measured_steps = [s for s in steps if s["start_ns"] < end and s["end_ns"] > start]
    water = [s["used_blocks"]/s["total_blocks"] for s in measured_steps]
    return {**base,
            "all_completed": bool(records) and all(r.status == "completed" for r in records),
            "all_requests_tpot_mean_ms": statistics.mean(tpots) if tpots else None,
            **percentiles("all_requests_tpot", tpots),
            "all_requests_tpot_scope": "completed requests, including warmup arrivals; full generation, not window-only",
            "prefill_wait_max_ms": max(waits) if waits else None,
            "prefill_wait_scope": "all recorded initial/inter-chunk/preemption waiting segments, INCLUDING drain and unfinished",
            "starvation_request_count": len(starved),
            "starvation_threshold_ms": starvation_ms,
            "preemptions": sum(r.preemptions or 0 for r in records),
            "recomputed_tokens": sum(r.recomputed_tokens or 0 for r in records),
            "kv_utilization_sampled_peak": max(water) if water else None,
            "kv_utilization_scope": "postprocess samples of Steps overlapping measurement window, not instantaneous allocation peak",
            "backlog_sample_count": len(backlog),
            "v1_acceptance_passed": False}
