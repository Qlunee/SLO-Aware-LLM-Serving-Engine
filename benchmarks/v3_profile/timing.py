"""Timing seam shared by real GPU collection and CPU contract tests."""
from time import perf_counter_ns


def timed_step(engine, clock=perf_counter_ns):
    """Stop at CPU-visible postprocess, before features, statistics or disk IO.

    run_mixed already synchronizes via token .tolist(), or explicitly for an
    intermediate chunk with no sampling. Do not add another measured sync.
    """
    start = clock()
    batch = engine.scheduler.schedule()
    scheduled = clock()
    result = engine.model_runner.call("run_mixed", batch, False, False, engine.greedy)
    ran = clock()
    engine.scheduler.postprocess(batch, result["tokens"])
    visible = clock()
    engine.last_step = {"batch": batch, **result}
    return batch, result, {"start_ns": start, "scheduled_ns": scheduled,
        "run_end_ns": ran, "visible_ns": visible,
        "step_time_ms": (visible-start)/1e6,
        "schedule_ms": (scheduled-start)/1e6,
        "run_ms": (ran-scheduled)/1e6, "postprocess_ms": (visible-ran)/1e6}
