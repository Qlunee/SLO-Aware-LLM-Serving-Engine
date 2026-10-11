"""One fresh-process TP=1/Eager controlled-injection run. No timing assertions."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import traceback

from .driver import execute_experiment
from .protocol import ROOT, load_suite, read, save, seal, sha, sources


def check_state(engine):
    """Diagnostic-only scans, NEVER enabled for performance comparisons."""
    scheduler = engine.scheduler
    if hasattr(scheduler, "assert_consistent"):
        scheduler.assert_consistent()
    sequences = getattr(engine, "sequences", None)
    if sequences is None:
        sequences = scheduler.sequences
    manager = scheduler.block_manager
    refs = [0] * len(manager.blocks)
    for seq in sequences.values():
        if not 0 <= seq.num_cached_tokens <= len(seq):
            raise AssertionError("invalid committed KV progress")
        if seq.num_completion_tokens > seq.max_tokens:
            raise AssertionError("duplicate sampling past fixed output length")
        if len(seq.block_table) != len(set(seq.block_table)):
            raise AssertionError("duplicate physical page within a request")
        for block in seq.block_table:
            if not 0 <= block < len(refs):
                raise AssertionError("invalid physical page")
            refs[block] += 1
        if seq.is_finished and seq.block_table:
            raise AssertionError("finished request retains page references")
    if any(block.ref_count != refs[i] for i, block in enumerate(manager.blocks)):
        raise AssertionError("page reference leak or cross-request ownership mismatch")
    used = {i for i, count in enumerate(refs) if count}
    free = list(manager.free_block_ids)
    if used != manager.used_block_ids or len(set(free)) != len(free) or set(free) != set(range(len(refs)))-used:
        raise AssertionError("allocator partition mismatch")
    for seq_id, record in engine.by_seq.items():
        if len(record.output_token_ids) != sequences[seq_id].num_completion_tokens:
            raise AssertionError("visible output and logical sequence diverged")


def warm_engine(engine, config, trace):
    """Exercise representative Prefill/Mixed/Decode shapes, then reset cold page metadata."""
    from benchmarks.v0.driver import execute_trace
    from benchmarks.v0.workload import RequestSpec
    from nanovllm.engine.block_manager import BlockManager
    specs = [RequestSpec(**{**s, "request_id": "warm-"+s["request_id"], "arrival_ns": 0,
                            "max_output_tokens": 16}) for s in trace["requests"]]
    result = execute_trace(engine, config, {"warmup_s": 0, "measurement_s": .001},
                           specs, 120, "full")
    if result.error or any(r.status != "completed" for r in result.records):
        raise RuntimeError("pre-measurement warmup did not complete")
    check_state(engine)
    if not engine.is_finished() or engine.scheduler.block_manager.used_block_ids:
        raise AssertionError("warmup left active requests or pages")
    # No model weights or KV tensor reallocation. Discard warm Prefix identities;
    # stale physical bytes are unreachable until valid positions are overwritten.
    engine.scheduler.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
    for name in ("by_seq", "history", "steps"):
        getattr(engine, name).clear()
    if hasattr(engine, "sequences"):
        engine.sequences.clear()
    if hasattr(engine.scheduler, "sequences"):
        engine.scheduler.sequences.clear()
    engine.step_count = engine.total_preemptions = 0
    if hasattr(engine.scheduler, "preemption_count"):
        engine.scheduler.preemption_count = 0
    return {"requests": len(specs), "output_tokens_each": 16,
            "cold_prefix_metadata_reset": True, "no_gpu_memory_reallocation": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--split", choices=["tune", "eval"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=["baseline", "chunk"], required=True)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-state", action="store_true")
    args = parser.parse_args()
    folder = Path(args.suite).resolve()
    protocol, experiment = load_suite(folder)
    if args.seed not in getattr(experiment, f"{args.split}_seeds"):
        parser.error("seed not in frozen split")
    if args.policy == "chunk" and args.chunk not in experiment.chunks:
        parser.error("chunk not in frozen candidate set")
    if args.policy == "baseline" and args.chunk is not None:
        parser.error("baseline has no chunk")
    trace_file = folder / args.split / f"seed-{args.seed}.json"
    trace = read(trace_file)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "trace.json", trace)
    save(out / "run_config.json", {**vars(args), "tp": 1, "graph": False, "observation": "full",
                                  "protocol_sha256": sha(folder / "protocol.json")})
    engine = None
    exit_code = 1
    try:
        import torch
        from benchmarks.v0.environment import collect_environment, model_metadata_hashes
        from benchmarks.v0.run import write_csv
        from benchmarks.v0.observed_engine import ObservedEngine
        from benchmarks.v1.observed_engine import ObservedMixedEngine
        model = protocol["model"]
        if model_metadata_hashes(Path(model["path"])) != protocol["model_metadata_sha256"]:
            raise ValueError("model metadata differs from frozen suite")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        environment = collect_environment(ROOT, model["path"], model["revision"])
        save(out / "environment.json", environment)
        kwargs = dict(tensor_parallel_size=1, enforce_eager=True,
                      max_model_len=experiment.max_model_len,
                      max_num_seqs=max(32, experiment.decode_count+1),
                      max_num_batched_tokens=experiment.token_budget,
                      gpu_memory_utilization=experiment.memory, kvcache_block_size=256)
        if args.policy == "baseline":
            engine = ObservedEngine(model["path"], observation="full", sampler="greedy", **kwargs)
            config = engine.model_runner.config
        else:
            engine = ObservedMixedEngine(model["path"], observation="full", sampler="greedy",
                                         chunk_size=args.chunk, **kwargs)
            config = engine.config
        cache = engine.model_runner.kv_cache
        if cache.dtype != torch.bfloat16:
            raise ValueError("this protocol requires BF16 KV")
        max_total = max(len(s["prompt_token_ids"])+s["max_output_tokens"] for s in trace["requests"])
        required_pages = sum(math.ceil((len(s["prompt_token_ids"])+s["max_output_tokens"])/256)
                             for s in trace["requests"])
        if max_total > config.max_model_len or required_pages > config.num_kvcache_blocks:
            raise ValueError("fixed Decode population plus long request cannot fit; no truncation/preemption allowed")
        warmup = warm_engine(engine, config, trace)
        if args.verify_state:
            step = engine.step
            def checked_step():
                before = {i: len(r.output_token_ids) for i, r in engine.by_seq.items()}
                result = step()
                for i, r in engine.by_seq.items():
                    if len(r.output_token_ids)-before[i] not in (0, 1):
                        raise AssertionError("more than one token sampled per request per step")
                check_state(engine)
                return result
            engine.step = checked_step
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result, summary = execute_experiment(engine, config, trace, experiment)
        torch.cuda.synchronize()
        end_state_passed = True
        try:
            check_state(engine)  # After the measured trace; excluded from timing.
        except AssertionError:
            end_state_passed = False
            summary["invalid_reasons"].append("end_state_check_failed")
            summary["valid_experiment"] = False
            save(out / "state_error.json", {"message": traceback.format_exc()})
        unchanged = sources() == protocol["source_sha256"]
        if not unchanged:
            summary["invalid_reasons"].append("source_changed_during_run")
            summary["valid_experiment"] = False
        summary.update(policy=args.policy, chunk=args.chunk, seed=args.seed, split=args.split,
                       diagnostic_only=args.verify_state, source_unchanged=unchanged,
                       end_state_check_passed=end_state_passed, driver_elapsed_s=result.end_ns/1e9,
                       torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       memory_scope="rank0 PyTorch allocator peak, measured trace INCLUDING drain; not total NVML process VRAM",
                       kv_blocks=config.num_kvcache_blocks,
                       kv_payload_bytes=cache.numel()*cache.element_size(),
                       v1_acceptance_passed=False, warmup=warmup)
        save(out / "requests.json", [asdict(r) for r in result.records])
        save(out / "steps.json", engine.steps)
        save(out / "backlog.json", result.backlog)
        write_csv(out / "steps.csv", engine.steps)
        save(out / "summary.json", summary)
        if result.error:
            save(out / "error.json", {"message": result.error})
        print({k: summary[k] for k in ("valid_experiment", "invalid_reasons", "decode_stall_max_ms",
                                      "long_ttft_ms", "window_throughput_tok_s", "diagnostic_only")}, flush=True)
        exit_code = 0 if summary["valid_experiment"] else 2
    except Exception:
        if not (out / "error.json").exists():
            save(out / "error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
