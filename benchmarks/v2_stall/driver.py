"""Controlled inputs on the existing open-loop driver, without moving arrivals."""
from benchmarks.v0.driver import execute_trace
from benchmarks.v0.workload import RequestSpec
from .metrics import analyze_run


def execute_experiment(engine, gpu_config, trace, config, **kwargs):
    specs = [RequestSpec(**s) for s in trace["requests"]]
    meta = {"warmup_s": config.inject_s, "measurement_s": config.window_s}
    result = execute_trace(engine, gpu_config, meta, specs, config.drain_s, "full", **kwargs)
    return result, analyze_run(trace, result.records, config, result.error)
