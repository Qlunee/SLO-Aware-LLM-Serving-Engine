from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    request_id: str
    arrival_ns: int
    prompt_tokens: int
    max_output_tokens: int
    ttft_slo_ms: float
    itl_slo_ms: float

    seq_id: int | None = None
    enqueue_ns: int | None = None
    first_scheduled_ns: int | None = None
    finish_ns: int | None = None
    status: str = "pending"
    error: str | None = None

    output_token_ids: list[int] = field(default_factory=list)
    token_times_ns: list[int] = field(default_factory=list)
    waiting_spans: list[dict] = field(default_factory=list)

    preemptions: int | None = None
    recomputed_tokens: int | None = None


def quantile(values: list[float], q: float) -> float | None:
    """线性插值分位数，与常见的(n-1)*q定义一致。"""
    if not 0 <= q <= 1:
        raise ValueError("q必须位于[0, 1]")
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("分位数输入包含非有限值")

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def request_metrics(record: RequestRecord) -> dict:
    times = record.token_times_ns

    if record.arrival_ns < 0:
        raise ValueError("计划到达不能为负")
    if record.finish_ns is not None and record.finish_ns < record.arrival_ns:
        raise ValueError("完成早于计划到达")

    if any(t < record.arrival_ns for t in times):
        raise ValueError("Token输出早于计划到达")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Token时间不单调")
    if record.enqueue_ns is not None:
        if record.enqueue_ns < record.arrival_ns:
            raise ValueError("入队早于计划到达")
    if record.first_scheduled_ns is not None:
        if record.enqueue_ns is None:
            raise ValueError("已调度请求缺少入队时间")
        if record.first_scheduled_ns < record.enqueue_ns:
            raise ValueError("首次调度早于入队")
    if times and record.enqueue_ns is not None and times[0] < record.enqueue_ns:
        raise ValueError("Token输出早于入队")
    if times and record.first_scheduled_ns is not None and times[0] < record.first_scheduled_ns:
        raise ValueError("Token输出早于调度")
    if times and record.finish_ns is not None and record.finish_ns < times[-1]:
        raise ValueError("请求完成早于最后Token")
    if record.status == "completed" and record.finish_ns is None:
        raise ValueError("完成请求缺少finish_ns")

    ttft_ms = (
        (times[0] - record.arrival_ns) / 1e6 if times else None
    )
    itls = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    itl_p99_ms = quantile(itls, 0.99)
    tpot_ms = (
        (times[-1] - times[0]) / 1e6 / (len(times) - 1)
        if len(times) >= 2 else None
    )

    if record.status != "completed":
        slo_pass = False
    elif ttft_ms is None or itl_p99_ms is None:
        slo_pass = None
    else:
        slo_pass = (
            ttft_ms <= record.ttft_slo_ms
            and itl_p99_ms <= record.itl_slo_ms
        )

    return {
        "enqueue_lag_ms": (
            (record.enqueue_ns - record.arrival_ns) / 1e6
            if record.enqueue_ns is not None else None
        ),
        "ttft_ms": ttft_ms,
        "itl_p99_ms": itl_p99_ms,
        "max_itl_ms": max(itls) if itls else None,
        "tpot_ms": tpot_ms,
        "slo_pass": slo_pass,
    }


def percentiles(name: str, values: list[float]) -> dict:
    return {
        f"{name}_p{label}_ms": quantile(values, q)
        for label, q in [(50, 0.5), (95, 0.95), (99, 0.99)]
    }


def summarize(
    records: list[RequestRecord],
    warmup_ns: int,
    measurement_end_ns: int,
    observed: bool,
) -> dict:
    seconds = (measurement_end_ns - warmup_ns) / 1e9
    if seconds <= 0:
        raise ValueError("测量窗口必须为正")
    if len({r.request_id for r in records}) != len(records):
        raise ValueError("请求ID重复")

    calculated = {
        r.request_id: request_metrics(r) for r in records
    }
    cohort = [
        r for r in records
        if warmup_ns <= r.arrival_ns < measurement_end_ns
    ]

    # 单Token请求单列，不放入双SLO队列。
    dual_cohort = [r for r in cohort if r.max_output_tokens >= 2]
    completed_cohort = [
        r for r in dual_cohort if r.status == "completed"
    ]
    window_completed = [
        r for r in records
        if r.status == "completed"
        and r.finish_ns is not None
        and warmup_ns <= r.finish_ns < measurement_end_ns
    ]

    counts = Counter(r.status for r in records)
    result = {
        "request_count": len(records),
        "completed_count": counts["completed"],
        "rejected_count": counts["rejected"],
        "unfinished_count": counts["unfinished"],
        "timeout_count": counts["timeout"],
        "failed_count": counts["failed"],
        "cohort_arrivals": len(cohort),
        "cohort_completion_rate": (
            sum(r.status == "completed" for r in cohort) / len(cohort)
            if cohort else None
        ),
        "cohort_arrival_rps": len(cohort) / seconds,
        "dual_slo_cohort_arrivals": len(dual_cohort),
        "single_token_cohort_arrivals": (
            len(cohort) - len(dual_cohort)
        ),
        "cohort_rejected": sum(r.status == "rejected" for r in cohort),
        "cohort_unfinished": sum(
            r.status in {"unfinished", "timeout"} for r in cohort
        ),
        "window_completed_rps": len(window_completed) / seconds,
        "window_completed_count": len(window_completed),
        "window_goodput_rps": None,
        "window_throughput_tok_s": None,
        "cohort_success_rate": None,
        "cohort_violation_rate": None,
        "completed_cohort_violation_rate": None,
        "max_itl_ms": None,
        "itl_exceed_fraction": None,
        "all_completed_makespan_s": None,
        "observed_ttft_count": 0,
        "observed_itl_count": 0,
    }

    if records and counts["completed"] == len(records):
        result["all_completed_makespan_s"] = (
            max(r.finish_ns for r in records)
            - min(r.arrival_ns for r in records)
        ) / 1e9

    ttfts, itls, lags = [], [], []
    request_itl_p99s, tpots = [], []

    if observed:
        successful = sum(
            calculated[r.request_id]["slo_pass"] is True
            for r in dual_cohort
        )
        if dual_cohort:
            rate = successful / len(dual_cohort)
            result["cohort_success_rate"] = rate
            result["cohort_violation_rate"] = 1 - rate

        if completed_cohort:
            result["completed_cohort_violation_rate"] = sum(
                calculated[r.request_id]["slo_pass"] is not True
                for r in completed_cohort
            ) / len(completed_cohort)

        result["window_goodput_rps"] = sum(
            calculated[r.request_id]["slo_pass"] is True
            for r in window_completed
            if r.max_output_tokens >= 2
        ) / seconds

        result["window_throughput_tok_s"] = sum(
            warmup_ns <= timestamp < measurement_end_ns
            for r in records
            for timestamp in r.token_times_ns
        ) / seconds

        exceeded = 0
        for r in cohort:
            values = calculated[r.request_id]
            if values["ttft_ms"] is not None:
                ttfts.append(values["ttft_ms"])
            if values["enqueue_lag_ms"] is not None:
                lags.append(values["enqueue_lag_ms"])
            if values["itl_p99_ms"] is not None:
                request_itl_p99s.append(values["itl_p99_ms"])
            if values["tpot_ms"] is not None:
                tpots.append(values["tpot_ms"])

            request_itls = [
                (b - a) / 1e6
                for a, b in zip(
                    r.token_times_ns, r.token_times_ns[1:]
                )
            ]
            itls.extend(request_itls)
            exceeded += sum(x > r.itl_slo_ms for x in request_itls)

        result["observed_ttft_count"] = len(ttfts)
        result["observed_itl_count"] = len(itls)
        result["max_itl_ms"] = max(itls) if itls else None
        result["itl_exceed_fraction"] = (
            exceeded / len(itls) if itls else None
        )

    result.update(percentiles("ttft", ttfts))
    result.update(percentiles("itl", itls))
    result.update(percentiles("enqueue_lag", lags))
    result.update(percentiles("request_itl_p99", request_itl_p99s))
    result.update(percentiles("tpot", tpots))
    return result
