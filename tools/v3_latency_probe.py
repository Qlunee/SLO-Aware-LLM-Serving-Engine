"""Short, diagnostic-only CPU/CUDA trace for the V3 fixed-shape drift.

This deliberately writes no V3 profile protocol/steps files: profiler runs
change timing and must never enter the predictor's training/evaluation pool.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import traceback

from benchmarks.defaults import model_path_default, revision_for
from benchmarks.v0.environment import collect_environment, command_output
from benchmarks.v3_profile.protocol import (Settings, planned_cases,
                                            predictor_module, runtime_identity)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def stats(rows, key):
    values = [row[key] for row in rows]
    return {"count": len(values), "p50_ms": statistics.median(values),
            "p95_ms": percentile(values, .95), "max_ms": max(values)}


def summarize_profile(profiler, out, case_id):
    trace = out / f"trace-{case_id}.json"
    profiler.export_chrome_trace(str(trace))
    averages = profiler.key_averages()
    (out / f"operators-{case_id}.txt").write_text(
        "CPU self time (microseconds):\n" +
        averages.table(sort_by="self_cpu_time_total", row_limit=20) +
        "\nCUDA self time (microseconds):\n" +
        averages.table(sort_by="self_device_time_total", row_limit=20))
    # Kineto emits a CPU and a GPU annotation for each record_function range.
    # Count only CPU-side ranges, and use CPU operations for attributed CUDA
    # time so the GPU annotation cannot double-count an entire Step.
    cpu_ops = [e for e in averages if str(e.device_type) == "DeviceType.CPU"]
    markers = [e for e in cpu_ops if e.key == "V3_DIAGNOSTIC_STEP"]
    cuda_ops = [e for e in cpu_ops if e.self_device_time_total > 0]
    return {"profiled_steps": sum(e.count for e in markers),
            "cuda_timed_op_calls": sum(e.count for e in cuda_ops),
            "marker_cpu_ms_total": sum(e.cpu_time_total for e in markers) / 1000,
            "cuda_self_ms_sum": sum(e.self_device_time_total for e in cpu_ops) / 1000,
            "trace": trace.name,
            "operators": f"operators-{case_id}.txt",
            "interpretation": "CUDA self-time is an operator sum, not end-to-end GPU elapsed time; inspect trace for CPU gaps and waits"}


def collect(args):
    settings = Settings(decode_counts=(6,), prefill_budgets=(0, 256),
                        contexts=(1024,), prefill_request_counts=(2,),
                        repeats=args.repeats, warmup=args.warmup)
    settings.validate()
    cases = planned_cases(settings)
    if args.profile_steps > settings.repeats:
        raise ValueError("profile-steps must not exceed repeats (pure Decode has one Step per episode)")
    if args.plan_only:
        print(json.dumps({"settings": settings.__dict__, "cases": cases,
                          "profile_steps_per_case": args.profile_steps}, indent=2))
        return

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    engine = None
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function
        from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
        from nanovllm.sampling_params import SamplingParams
        from benchmarks.v3_profile import driver

        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        rng = random.Random(args.seed)
        environment = collect_environment(Path(__file__).resolve().parents[1],
                                          args.model, args.model_revision)
        environment["v3_driver"] = command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        engine = MixedLLMEngine(args.model, chunk_size=256,
            kv_block_cap=settings.kv_block_cap, greedy=True, verify_tp_plan=False,
            tensor_parallel_size=1, enforce_eager=True,
            max_model_len=settings.max_model_len, max_num_seqs=settings.max_num_seqs,
            max_num_batched_tokens=settings.token_budget,
            gpu_memory_utilization=settings.memory, kvcache_block_size=256)
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 KV required")
        if engine.config.num_kvcache_blocks != settings.kv_block_cap:
            raise ValueError("requested common KV cap unavailable")
        identity = runtime_identity(environment, engine.config,
                                    engine.model_runner.kv_cache)
        write_json(out / "probe_protocol.json", {
            "schema": "v3-latency-probe-v1", "diagnostic_only": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed, "settings": settings.__dict__, "identity": identity,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "profile_steps_per_case": args.profile_steps,
            "timing": "existing CPU-visible timed_step; profiler runs on first measured Steps only"})
        write_json(out / "environment.json", environment)
        estimator = predictor_module()
        scheduler = engine.scheduler
        vocab = engine.config.hf_config.vocab_size
        special = set(engine.tokenizer.all_special_ids)

        def add(length, outputs):
            tokens = []
            while len(tokens) < length:
                token = rng.randrange(vocab)
                if token not in special:
                    tokens.append(token)
            return engine.add_request(tokens, SamplingParams(max_tokens=outputs,
                                                               ignore_eos=True))

        rng.shuffle(cases)
        profiles = {}
        with (out / "probe_steps.jsonl").open("x") as stream:
            for case in cases:
                case_id = case["case_id"]
                steps_per_episode = 4 if case["prefill_budget"] else 1
                warmup_steps = settings.warmup * steps_per_episode
                last_profile_step = warmup_steps + args.profile_steps
                state = {"count": 0, "profiler": None, "stopped": False}
                original = driver.timed_step

                def measured(engine):
                    index = state["count"]
                    state["count"] += 1
                    if index == warmup_steps:
                        state["profiler"] = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            acc_events=True)
                        state["profiler"].start()
                    if warmup_steps <= index < last_profile_step:
                        with record_function("V3_DIAGNOSTIC_STEP"):
                            result = original(engine)
                        state["profiler"].step()
                        if index == last_profile_step - 1:
                            state["profiler"].stop()
                            state["stopped"] = True
                    else:
                        result = original(engine)
                    return result

                def observe(case, episode, local_step, warmup, batch, result, times):
                    measured_index = (episode - settings.warmup) * steps_per_episode + local_step
                    row = {"case_id": case_id, "episode": episode,
                           "episode_step": local_step, "warmup": warmup,
                           "profiled": 0 <= measured_index < args.profile_steps,
                           "after_profiler": args.profile_steps <= measured_index < args.profile_steps + 2,
                           "features": estimator.batch_features(batch, result["graph_mode"]),
                           **times, "preemptions_total": scheduler.preemption_count}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

                driver.timed_step = measured
                try:
                    driver.run_case(engine, case, settings, add, observe)
                finally:
                    driver.timed_step = original
                    if state["profiler"] is not None and not state["stopped"]:
                        state["profiler"].stop()
                expected = (settings.warmup + settings.repeats) * steps_per_episode
                if state["count"] != expected:
                    raise ValueError(f"incomplete case {case_id}: {state['count']} != {expected}")
                profiles[case_id] = summarize_profile(state["profiler"], out, case_id)
                if profiles[case_id]["profiled_steps"] != args.profile_steps:
                    raise ValueError(f"profiler lost Step markers for {case_id}")
                if profiles[case_id]["cuda_timed_op_calls"] == 0:
                    raise RuntimeError(f"CUDA profiler captured no timed operations for {case_id}")
                print(f"{case_id}: {expected} Steps, {args.profile_steps} profiled", flush=True)
        rows = [json.loads(line) for line in (out / "probe_steps.jsonl").read_text().splitlines()]
        write_json(out / "probe_summary.json", {
            "schema": "v3-latency-probe-v1", "valid": True,
            "recorded_steps": len(rows),
            "measured_steps": sum(not r["warmup"] for r in rows),
            "preemptions": scheduler.preemption_count,
            "profiles": profiles})
        print(json.dumps({"probe": str(out), "measured_steps":
                          sum(not r["warmup"] for r in rows)}, ensure_ascii=False))
    except BaseException:
        write_json(out / "error.json", {"error": traceback.format_exc()})
        raise
    finally:
        if engine is not None:
            engine.exit()


def summarize(paths):
    loaded = []
    for path in paths:
        folder = Path(path).resolve()
        if not (folder / "probe_summary.json").exists():
            raise ValueError(f"probe did not finish: {folder}; inspect error.json or collection log")
        protocol = json.loads((folder / "probe_protocol.json").read_text())
        summary = json.loads((folder / "probe_summary.json").read_text())
        if protocol["schema"] != "v3-latency-probe-v1" or not summary["valid"]:
            raise ValueError(f"invalid diagnostic probe: {folder}")
        rows = [json.loads(line) for line in (folder / "probe_steps.jsonl").read_text().splitlines()]
        if len(rows) != summary["recorded_steps"] or summary["preemptions"]:
            raise ValueError(f"incomplete diagnostic probe: {folder}")
        loaded.append((folder, protocol, summary, rows))
    identity = loaded[0][1]["identity"]
    signature = [(r["case_id"], r["episode"], r["episode_step"], r["features"])
                 for r in loaded[0][3]]
    if any(p["identity"] != identity or p["seed"] != loaded[0][1]["seed"] or
           [(r["case_id"], r["episode"], r["episode_step"], r["features"])
            for r in rows] != signature for _, p, _, rows in loaded[1:]):
        raise ValueError("probe runs differ in execution identity or Step shapes")
    result = []
    for folder, _, summary, rows in loaded:
        groups = defaultdict(list)
        for row in rows:
            if not row["warmup"] and not row["profiled"] and not row["after_profiler"]:
                groups[(row["case_id"], row["episode_step"])].append(row)
        result.append({"folder": str(folder), "ordinary_steps": sum(map(len, groups.values())),
                       "groups": [{"case_id": case, "episode_step": local,
                                   "step": stats(items, "step_time_ms"),
                                   "run": stats(items, "run_ms"),
                                   "schedule": stats(items, "schedule_ms"),
                                   "postprocess": stats(items, "postprocess_ms")}
                                  for (case, local), items in sorted(groups.items())],
                       "profiles": summary["profiles"]})
    print(json.dumps({"schema": "v3-latency-probe-summary-v1", "runs": result},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect")
    run.add_argument("--out", required=True)
    run.add_argument("--model", default=model_path_default())
    run.add_argument("--model-revision")
    run.add_argument("--seed", type=int, default=701)
    run.add_argument("--repeats", type=int, default=50)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--profile-steps", type=int, default=8)
    run.add_argument("--plan-only", action="store_true")
    report = sub.add_parser("summarize")
    report.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        args.model = str(Path(args.model).expanduser().resolve())
        args.model_revision = revision_for(args.model, args.model_revision)
        if args.seed < 0 or args.profile_steps < 1:
            parser.error("seed must be nonnegative and profile-steps positive")
        collect(args)
    else:
        summarize(args.runs)


if __name__ == "__main__":
    main()
