"""Paired Slack, optional Aging/Hysteresis/KV Guard, and fixed Chunks on a frozen V3 trace.

The reused trace is suitable for correctness and diagnostic comparison, not
an independent SLO performance claim.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

from benchmarks.v0.environment import collect_environment, command_output
from benchmarks.v0.driver import execute_trace
from benchmarks.v0.run import write_csv
from benchmarks.v2_baselines.metrics import analyze_online
from benchmarks.v2_baselines.workload import validate_trace
from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources, verify_artifacts
from benchmarks.v2_stall.run import check_state, warm_engine
from benchmarks.v3_profile.protocol import runtime_identity


NEW_FILES = (
    "nanovllm/engine/slack_scheduler.py",
    "benchmarks/v3_slack/observed_engine.py", "benchmarks/v3_slack/run.py",
)
POLICY_SOURCES = frozenset((
    "nanovllm/engine/step_predictor.py",
    "nanovllm/engine/slack_scheduler.py",
))


def profiled_execution_drift(frozen, current):
    """Reject changes to the V1 path measured by the frozen Step profiles."""
    return [name for name, digest in frozen.items()
            if name.startswith("nanovllm/") and name not in POLICY_SOURCES
            and current.get(name) != digest]


def runtime_mismatches(frozen, current, allow_cross_gpu_diagnostic=False):
    """Compare JSON identities; allow a tiny same-model GPU size delta only by opt-in."""
    current = json.loads(json.dumps(current, allow_nan=False))
    frozen = json.loads(json.dumps(frozen, allow_nan=False))
    differences = [key for key in frozen if key != "source_sha256" and
                   current.get(key) != frozen[key]]
    if allow_cross_gpu_diagnostic and "gpus" in differences:
        old, new = frozen["gpus"], current["gpus"]
        if (len(old) == len(new) == 1 and
                all(old[0].get(k) == new[0].get(k)
                    for k in ("index", "name", "compute_capability")) and
                abs(old[0]["total_memory_bytes"] - new[0]["total_memory_bytes"]) <= 1024 * 1024):
            differences.remove("gpus")
    return differences


def prepare_warmup(engine, config, trace):
    """Prime singleton Decode before the full trace warmup for both policies."""
    if not trace["requests"]:
        raise ValueError("warmup trace has no requests")
    # The full zero-arrival warmup can keep the fixed policy in multi-request
    # Decode throughout. Its first singleton Decode would then JIT during the
    # measured window, while Slack has already exercised that shape.
    warm_engine(engine, config, {"requests": [trace["requests"][0]]})
    result = warm_engine(engine, config, trace)
    return {**result, "singleton_decode_primed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="online_plan/rN from the completed terminal suite")
    parser.add_argument("--mode", choices=("poisson", "burst", "long_short"), required=True)
    parser.add_argument("--policy", choices=("slack", "slack-hysteresis",
                    "slack-aging", "slack-aging-hysteresis", "slack-kv",
                    "slack-aging-kv", "slack-hysteresis-kv",
                    "slack-aging-hysteresis-kv", "chunk-256", "chunk-512",
                    "chunk-1024"), required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--safety-margin-ms", type=float, default=15.0)
    parser.add_argument("--rise-steps", type=int, default=2)
    parser.add_argument("--rise-margin-ms", type=float, default=10.0)
    parser.add_argument("--aging-warning-ms", type=float, default=300.0)
    parser.add_argument("--aging-hard-ms", type=float, default=800.0)
    parser.add_argument("--kv-low-free-blocks", type=int, default=8)
    parser.add_argument("--kv-high-free-blocks", type=int, default=16)
    parser.add_argument("--allow-cross-gpu-diagnostic", action="store_true",
                        help="same-model GPU with <=1 MiB size difference; diagnostic only")
    args = parser.parse_args()
    plan_dir, out = Path(args.plan).resolve(), Path(args.out).resolve()
    verify_artifacts(plan_dir)
    plan = read(plan_dir/"plan.json")
    if plan.get("schema") != "v3-refine-plan-v1":
        raise ValueError("wrong V3 online plan")
    trace_path = plan_dir/"traces"/f"{args.mode}.json"
    trace = read(trace_path)
    specs = validate_trace(trace)
    if trace["seed"] != plan["online_seeds"][args.mode]:
        raise ValueError("online trace seed differs from plan")
    predictor_path = Path(args.predictor).resolve()
    verify_artifacts(predictor_path.parent)
    table = read(predictor_path)
    if table.get("identity") != plan["identity"]:
        raise ValueError("predictor and frozen online plan identities differ")
    frozen = plan["identity"]["source_sha256"]
    current = sources()
    # Policy files are versioned in each run; the profiled V1 execution path
    # must remain byte-for-byte identical to the predictor training source.
    altered = profiled_execution_drift(frozen, current)
    if altered:
        raise ValueError(f"profiled execution source changed: {altered}")
    source_start = {name: sha(ROOT/name) for name in NEW_FILES}
    predictor_sha = sha(predictor_path)
    trace_sha = sha(trace_path)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"run_config.json", {**vars(args), "trace_sha256": trace_sha,
                                  "predictor_sha256": predictor_sha,
                                  "plan_sha256": sha(plan_dir/"plan.json"),
                                  "source_sha256": source_start,
                                  "predictor_source_drift": current.get("nanovllm/engine/step_predictor.py") !=
                                      frozen.get("nanovllm/engine/step_predictor.py"),
                                  "scope": "reused online trace; correctness/diagnostic smoke only"})
    engine = None
    try:
        import torch
        from nanovllm.engine.step_predictor import StepPredictor
        from benchmarks.v2_baselines.observed_engine import ObservedFixedEngine
        from .observed_engine import ObservedSlackEngine
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU")
        torch.manual_seed(trace["seed"])
        torch.cuda.manual_seed_all(trace["seed"])
        s = plan["settings"]
        kwargs = dict(tensor_parallel_size=1, enforce_eager=True,
            max_model_len=s["max_model_len"], max_num_seqs=s["max_num_seqs"],
            max_num_batched_tokens=s["token_budget"],
            gpu_memory_utilization=s["memory"], kvcache_block_size=256,
            kv_block_cap=s["kv_block_cap"])
        predictor = StepPredictor(table, plan["identity"])
        fixed_chunk = int(args.policy.split("-", 1)[1]) if args.policy.startswith("chunk-") else None
        slack_policy = fixed_chunk is None
        if slack_policy:
            engine = ObservedSlackEngine(plan["model_path"], predictor,
                safety_margin_ms=args.safety_margin_ms,
                hysteresis="hysteresis" in args.policy,
                rise_steps=args.rise_steps, rise_margin_ms=args.rise_margin_ms,
                aging="aging" in args.policy,
                aging_warning_ms=args.aging_warning_ms,
                aging_hard_ms=args.aging_hard_ms,
                kv_guard="kv" in args.policy,
                kv_low_free_blocks=args.kv_low_free_blocks,
                kv_high_free_blocks=args.kv_high_free_blocks,
                **kwargs)
        else:
            engine = ObservedFixedEngine(plan["model_path"], policy="chunk",
                chunk_size=fixed_chunk, **kwargs)
        if engine.config.num_kvcache_blocks != s["kv_block_cap"]:
            raise ValueError("common KV cap unavailable")
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 KV required")
        environment = collect_environment(ROOT, plan["model_path"],
                                          plan["identity"]["model_revision"])
        environment["v3_driver"] = command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        runtime = runtime_identity(environment, engine.config, engine.model_runner.kv_cache)
        mismatches = runtime_mismatches(plan["identity"], runtime,
                                        args.allow_cross_gpu_diagnostic)
        if mismatches:
            raise ValueError(f"predictor runtime identity differs: {mismatches}")
        save(out/"environment.json", environment)
        warmup = prepare_warmup(engine, engine.config, trace)
        if slack_policy:
            engine.scheduler.last_visible_ns.clear()
            engine.scheduler.itl_slo_ms.clear()
            engine.scheduler.ttft_deadline_ns.clear()
            engine.scheduler.small_prefill_ms.clear()
            engine.scheduler.prefill_skip_steps = 0
            engine.scheduler.reset_hysteresis()
            engine.scheduler.reset_aging()
            engine.scheduler.reset_kv_guard()
            for residuals in engine.scheduler.residuals.values():
                residuals.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = execute_trace(engine, engine.config, trace, specs, plan["drain_s"], "full")
        torch.cuda.synchronize()
        check_state(engine)
        summary = analyze_online(trace, result.records, engine.steps, result.backlog,
                                 result.end_ns, starvation_ms=1000)
        source_unchanged = (source_start == {name: sha(ROOT/name) for name in NEW_FILES}
                            and sha(predictor_path) == predictor_sha
                            and sha(trace_path) == trace_sha)
        summary.update(policy=args.policy, mode=args.mode, seed=trace["seed"],
            valid_measurement=bool(not result.error and source_unchanged),
            source_unchanged=source_unchanged, warmup=warmup,
            end_ns=result.end_ns, deadline_ns=result.deadline_ns,
            predictor_sha256=predictor_sha,
            kv_blocks=engine.config.num_kvcache_blocks,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cross_gpu_diagnostic=args.allow_cross_gpu_diagnostic,
            runtime_gpu_identity=runtime["gpus"],
            scope="reused V3 online trace; diagnostic comparison, no independent SLO claim")
        if slack_policy:
            summary["hysteresis_hold_steps"] = sum(
                row.get("decision_reason") == "hysteresis_hold" for row in engine.steps)
            summary["hysteresis_raise_steps"] = sum(
                row.get("decision_reason") == "hysteresis_raise" for row in engine.steps)
            summary["mixed_prefill_budget_switches"] = sum(
                row.get("num_decode", 0) > 0 and
                row.get("hysteresis_previous_budget", 0) is not None and
                row.get("hysteresis_previous_budget", 0) > 0 and
                row.get("selected_prefill_budget") != row["hysteresis_previous_budget"]
                for row in engine.steps)
            summary["aging_warning_steps"] = sum(
                row.get("aging_phase") == "warning" for row in engine.steps)
            summary["aging_hard_steps"] = sum(
                row.get("aging_phase") == "hard" for row in engine.steps)
            summary["aging_priority_steps"] = sum(
                row.get("aging_served") and
                row.get("aging_promoted_position") is not None and
                row["aging_promoted_position"] < row["aging_original_position"]
                for row in engine.steps)
            summary["aging_conflict_steps"] = sum(
                row.get("aging_conflict") is not None for row in engine.steps)
            summary["aging_max_wait_age_ms"] = max(
                (row.get("aging_oldest_wait_ms") or 0 for row in engine.steps),
                default=0)
            summary["kv_guard_held_steps"] = sum(
                row.get("kv_guard_active", False) for row in engine.steps)
            summary["kv_guard_enter_steps"] = sum(
                row.get("kv_guard_entered", False) for row in engine.steps)
            summary["kv_guard_exit_steps"] = sum(
                row.get("kv_guard_exited", False) for row in engine.steps)
            summary["kv_guard_blocked_steps"] = sum(
                bool(row.get("kv_guard_blocked_seq_ids")) for row in engine.steps)
            summary["kv_guard_blocked_admissions"] = sum(
                len(row.get("kv_guard_blocked_seq_ids", [])) for row in engine.steps)
            summary["kv_guard_conflict_steps"] = sum(
                row.get("kv_guard_conflict") is not None for row in engine.steps)
        save(out/"summary.json", summary)
        save(out/"requests.json", [asdict(r) for r in result.records])
        save(out/"steps.json", engine.steps)
        save(out/"backlog.json", result.backlog)
        write_csv(out/"steps.csv", engine.steps)
        if result.error:
            save(out/"error.json", {"message": result.error})
        print({k: summary[k] for k in ("policy", "mode", "valid_measurement",
            "all_completed", "ttft_p99_ms", "itl_p99_ms", "window_goodput_rps",
            "window_throughput_tok_s")}, flush=True)
        if not summary["valid_measurement"] or not summary["all_completed"]:
            raise SystemExit(2)
    except BaseException:
        if not (out/"error.json").exists():
            save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)


if __name__ == "__main__":
    main()
