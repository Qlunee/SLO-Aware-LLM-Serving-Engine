"""Unclipped boundary-overlap ITL, validity checks, paired statistics and branches."""
from dataclasses import fields
import statistics

from benchmarks.v0.metrics import RequestRecord, quantile, request_metrics, summarize


def records_from_rows(rows):
    names = {f.name for f in fields(RequestRecord)}
    return [RequestRecord(**{k: v for k, v in row.items() if k in names}) for row in rows]


def stats(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def readiness(record, inject_ns, config):
    times = [t for t in record.token_times_ns if t <= inject_ns]
    gaps = [b-a for a, b in zip(times, times[1:])][-config.stable_intervals:]
    mean = statistics.mean(gaps) if gaps else 0
    cv = statistics.pstdev(gaps) / mean if len(gaps) >= 2 and mean > 0 else None
    fresh = bool(times and mean > 0 and inject_ns-times[-1] <= max(
        4*mean, config.max_enqueue_lag_ms*1e6))
    alive = record.finish_ns is None or record.finish_ns > inject_ns
    return {"tokens_before_injection": len(times), "intervals": len(gaps), "cv": cv,
            "fresh": fresh, "alive": alive, "stable": len(gaps) == config.stable_intervals
            and cv is not None and cv <= config.stable_cv_max and fresh and alive}


def analyze_run(trace, records, config, error=None):
    start, end = trace["inject_ns"], trace["window_end_ns"]
    ids = [r.request_id for r in records]
    expected = {s["request_id"]: s for s in trace["requests"]}
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("duplicate, missing or unexpected request records")
    by_id = {r.request_id: r for r in records}
    invalid = ["driver_error"] if error else []
    for r in records:
        spec = expected[r.request_id]
        if any(getattr(r, name) != spec[name] for name in
               ("arrival_ns", "max_output_tokens", "ttft_slo_ms", "itl_slo_ms")) or r.prompt_tokens != len(spec["prompt_token_ids"]):
            raise ValueError("request fields differ from frozen trace")
        request_metrics(r)  # Validates monotonic and causally ordered timestamps.
        if r.status != "completed":
            invalid.append(f"not_completed:{r.request_id}")
        if len(r.output_token_ids) != r.max_output_tokens or len(r.token_times_ns) != len(r.output_token_ids):
            invalid.append(f"output_length_mismatch:{r.request_id}")
        if r.preemptions:
            invalid.append(f"preemption_changes_fixed_decode_population:{r.request_id}")
    long = by_id[trace["long_id"]]
    lag = (long.enqueue_ns-start)/1e6 if long.enqueue_ns is not None else None
    if lag is None or not 0 <= lag <= config.max_enqueue_lag_ms:
        invalid.append("long_enqueue_lag_exceeds_bound")
    details, pooled, before = [], [], []
    for request_id in trace["decode_ids"]:
        r = by_id[request_id]
        ready = readiness(r, start, config)
        if not ready["stable"]:
            invalid.append(f"decode_not_stable:{request_id}")
        if not r.token_times_ns or r.token_times_ns[-1] < end or (r.finish_ns is not None and r.finish_ns < end):
            invalid.append(f"decode_window_censored:{request_id}")
        # Intervals overlapping [start, end) are included whole. In particular,
        # include a pause starting before injection and ending AFTER window end.
        values = [(b-a)/1e6 for a, b in zip(r.token_times_ns, r.token_times_ns[1:])
                  if a < end and b > start]
        prior = [(b-a)/1e6 for a, b in zip(r.token_times_ns, r.token_times_ns[1:])
                 if max(0, start-1_000_000_000) <= a and b <= start]
        if not values:
            invalid.append(f"no_decode_interval:{request_id}")
        pooled.extend(values)
        before.extend(prior)
        details.append({"request_id": request_id, "readiness": ready, "itl_count": len(values),
                        "stall_max_ms": max(values) if values else None,
                        "itl_p99_ms": quantile(values, .99)})
    base = summarize(records, start, end, True)
    return {**base, "valid_experiment": not invalid, "invalid_reasons": invalid,
            "decode_stall_max_ms": max(pooled) if pooled else None,
            "decode_itl_p99_ms": quantile(pooled, .99), "decode_itl_count": len(pooled),
            "pre_injection_itl_p50_ms": quantile(before, .5),
            "long_ttft_ms": request_metrics(long)["ttft_ms"], "long_enqueue_lag_ms": lag,
            "per_decode_request": details,
            "note": "Boundary-overlap ITLs are not clipped. All fixed Decode requests remain in validity checks. Not a saturation-capacity experiment."}


def compare_tokens(trace, baseline, candidate):
    specs = {s["request_id"]: s for s in trace["requests"]}
    if set(baseline) != set(specs) or set(candidate) != set(specs):
        raise ValueError("Token comparison requires every request, including failures")
    rows = []
    for request_id, spec in specs.items():
        a, b = baseline[request_id], candidate[request_id]
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        exact = a == b
        branch = not exact and prefix < min(len(a), len(b))
        rows.append({"request_id": request_id, "exact_match": exact,
                     "baseline_tokens": len(a), "candidate_tokens": len(b),
                     "common_prefix_tokens": prefix,
                     "first_divergence_token_1based": prefix+1 if branch else None,
                     "length_mismatch": len(a) != len(b),
                     "baseline_token": a[prefix] if branch else None,
                     "candidate_token": b[prefix] if branch else None,
                     "same_history_token_ids": spec["prompt_token_ids"] + a[:prefix] if branch else None,
                     "review_status": "NEEDS_REVIEW" if not exact else "EXACT_ON_THIS_TRACE"})
    return rows


def aggregate_pairs(pairs, expected_seeds):
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("invalid expected seeds")
    by_seed = {p["seed"]: p for p in pairs}
    if len(by_seed) != len(pairs) or set(by_seed) != set(expected_seeds):
        raise ValueError("missing, duplicate or unexpected paired runs")
    invalid = [p["seed"] for p in pairs if not p["baseline"]["valid_experiment"]
               or not p["candidate"]["valid_experiment"]]
    result = {"expected_pairs": len(expected_seeds), "invalid_seeds": invalid,
              "valid_for_performance_comparison": not invalid,
              "v1_acceptance_passed": False,
              "note": "No invalid pair is discarded. Numerical/generation review is separate from timing validity."}
    if invalid:
        return result
    keys = ("decode_stall_max_ms", "decode_itl_p99_ms", "long_ttft_ms", "window_throughput_tok_s",
            "ttft_p99_ms", "itl_p99_ms", "window_goodput_rps", "cohort_violation_rate")
    result["baseline"] = {k: stats(p["baseline"][k] for p in pairs) for k in keys}
    result["candidate"] = {k: stats(p["candidate"][k] for p in pairs) for k in keys}
    result["paired_stall_reduction_pct"] = stats(
        100*(1-p["candidate"]["decode_stall_max_ms"]/p["baseline"]["decode_stall_max_ms"])
        for p in pairs)
    result["paired_throughput_change_pct"] = stats(
        100*(p["candidate"]["window_throughput_tok_s"]/p["baseline"]["window_throughput_tok_s"]-1)
        for p in pairs if p["baseline"]["window_throughput_tok_s"] > 0)
    if result["paired_throughput_change_pct"]["n"] != len(pairs):
        result["valid_for_performance_comparison"] = False
        result["zero_baseline_window_throughput"] = True
    return result


def choose_chunk(tuning_reports, config):
    """Predeclared objective; input is tuning reports ONLY, never eval metrics."""
    eligible, rejected = [], {}
    for chunk, report in tuning_reports.items():
        reason = None
        if not report["valid_for_performance_comparison"]:
            reason = "invalid_or_missing_pair"
        elif report["paired_throughput_change_pct"]["mean"] < -100*config.throughput_loss_limit:
            reason = "mean_throughput_loss_exceeds_frozen_bound"
        elif report["candidate"]["long_ttft_ms"]["max"] > config.ttft_slo_ms:
            reason = "long_TTFT_exceeds_frozen_SLO_in_a_tuning_run"
        if reason:
            rejected[str(chunk)] = reason
        else:
            eligible.append((report["candidate"]["decode_stall_max_ms"]["mean"], chunk))
    return {"chunk": min(eligible)[1] if eligible else None, "rejected": rejected,
            "objective": "Minimum mean per-run maximum Decode ITL, within mean throughput-loss bound and per-run long TTFT SLO; ties choose smaller chunk.",
            "selection_split": "tune", "v1_acceptance_passed": False}
