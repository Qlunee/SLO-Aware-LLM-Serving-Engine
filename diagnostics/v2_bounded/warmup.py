"""Eight representative requests, independent of measured rate/duration/count."""
from dataclasses import asdict
from collections import Counter
import math
from pathlib import Path
import traceback

from benchmarks.v0.workload import RequestSpec
from benchmarks.v0.driver import execute_trace
from benchmarks.v2_stall.protocol import save, seal


def build_warmup(short_tokens, long_tokens, revision, token_factory):
    if (type(short_tokens) is not int or type(long_tokens) is not int
            or not 0 < short_tokens <= long_tokens or not revision.strip()):
        raise ValueError("invalid warmup lengths/revision")
    rows = []
    for i, length in enumerate([short_tokens] * 4 + [long_tokens] * 4):
        tokens = token_factory(length, i)
        if len(tokens) != length or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid warmup tokens")
        rows.append(asdict(RequestSpec(f"warm-{i}", 0, list(tokens), 16, 1000., 100.)))
    return {"schema": "bounded-warmup-v1", "model_revision": revision,
            "warmup_s": 0, "measurement_s": .001, "drain_s": 120,
            "requests": rows}


def reset_cold(engine, config):
    from benchmarks.v2_stall.run import check_state
    check_state(engine)
    scheduler = engine.scheduler
    if not engine.is_finished() or scheduler.block_manager.used_block_ids:
        raise AssertionError("warmup left active requests/pages")
    if getattr(scheduler, "pending", None) is not None:
        raise AssertionError("warmup left an uncommitted batch")
    # Only CPU page/prefix metadata is rebuilt. The GPU KV tensor is unchanged.
    old_manager = scheduler.block_manager
    scheduler.block_manager = type(old_manager)(config.num_kvcache_blocks, config.kvcache_block_size)
    for owner, names in ((engine, ("by_seq", "history", "steps", "sequences")),
                         (scheduler, ("sequences", "wait_since", "last_decision"))):
        for name in names:
            if hasattr(owner, name):
                getattr(owner, name).clear()
    for name in ("step_count", "total_preemptions"):
        setattr(engine, name, 0)
    if hasattr(scheduler, "preemption_count"):
        scheduler.preemption_count = 0
    if hasattr(engine, "last_step"):
        engine.last_step = None
    check_state(engine)
    if (scheduler.waiting or scheduler.running or engine.by_seq or engine.steps
            or scheduler.block_manager.hash_to_block_id
            or len(scheduler.block_manager.free_block_ids) != config.num_kvcache_blocks):
        raise AssertionError("cold reset incomplete")


def prewarm(engine, config, trace, out):
    """Run the frozen eight-request trace and retain diagnostics on every outcome."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "trace.json", trace)
    report = {"passed": False, "diagnostic_only": True}
    pool = engine.model_runner.kv_cache
    try:
        specs = [RequestSpec(**row) for row in trace["requests"]]
        required = sum(math.ceil((len(s.prompt_token_ids) + s.max_output_tokens)
                                 / config.kvcache_block_size) for s in specs)
        if (len(specs) != 8 or any(s.arrival_ns != 0 or s.max_output_tokens != 16 for s in specs)
                or required > config.num_kvcache_blocks):
            raise ValueError("warmup must be eight fitting requests; no silent truncation")
        result = execute_trace(engine, config, trace, specs, trace["drain_s"], "full")
        report.update(request_count=len(specs), required_blocks_upper_bound=required,
            status_counts=dict(Counter(r.status for r in result.records)),
            driver_error=result.error, elapsed_s=result.end_ns / 1e9,
            deadline_s=result.deadline_ns / 1e9,
            deadline_reached=result.end_ns >= result.deadline_ns,
            last_backlog=result.backlog[-1] if result.backlog else None,
            step_count=len(engine.steps), phases=sorted({s["phase"] for s in engine.steps}))
        save(out / "requests.json", [asdict(r) for r in result.records])
        save(out / "steps.json", engine.steps)
        save(out / "backlog.json", result.backlog)
        if (result.error or any(r.status != "completed" or len(r.output_token_ids) != 16
                                for r in result.records)):
            raise RuntimeError("bounded warmup failed; see warmup/report.json for internal cause")
        reset_cold(engine, config)
        if engine.model_runner.kv_cache is not pool:
            raise AssertionError("warmup reallocated GPU KV tensor")
        report.update(passed=True, cold_prefix_metadata_reset=True,
                      observations_reset=True, no_gpu_memory_reallocation=True)
    except Exception:
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save(out / "report.json", report)
        seal(out)
    return report
