"""One fresh-process online smoke run, or separate per-Step diagnostics."""
import argparse
from dataclasses import asdict
from pathlib import Path
import traceback

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources
from .protocol import load, Settings, POLICIES
from .workload import validate_trace
from .metrics import analyze_online


def make_engine(protocol, policy, kv_block_cap=None):
    from benchmarks.v0.observed_engine import ObservedEngine
    from .observed_engine import ObservedFixedEngine
    s = Settings(**protocol["settings"])
    kwargs = dict(tensor_parallel_size=1, enforce_eager=True, max_model_len=s.max_model_len,
                  max_num_seqs=s.max_num_seqs, max_num_batched_tokens=s.token_budget,
                  gpu_memory_utilization=s.memory, kvcache_block_size=256)
    if policy == "baseline":
        if kv_block_cap is not None:
            raise ValueError("diagnostic KV cap is supported only for Mixed")
        engine = ObservedEngine(protocol["model"]["path"], observation="full", sampler="greedy", **kwargs)
        return engine, engine.model_runner.config
    chunk = int(policy.split("-")[1]) if policy.startswith("chunk-") else 256
    name = "chunk" if policy.startswith("chunk-") else policy
    engine = ObservedFixedEngine(protocol["model"]["path"], policy=name, chunk_size=chunk,
        max_prefill_wait_ms=s.max_prefill_wait_ms, kv_block_cap=kv_block_cap, **kwargs)
    return engine, engine.config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--mode", choices=("poisson", "burst", "long_short"), required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    protocol = load(suite)
    settings = Settings(**protocol["settings"])
    trace = read(suite/f"traces/{args.mode}-seed-{settings.seed}.json")
    specs = validate_trace(trace)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"trace.json", trace)
    identity = {"policy": args.policy, "mode": args.mode, "seed": settings.seed,
                "tp": 1, "graph": False, "observation": "full",
                "diagnostic_only": args.diagnostic, "protocol_sha256": sha(suite/"protocol.json")}
    save(out/"run_config.json", identity)
    engine = None
    code = 1
    try:
        import torch
        from benchmarks.v0.environment import collect_environment
        from benchmarks.v0.driver import execute_trace
        from benchmarks.v0.run import write_csv
        from benchmarks.v2_stall.run import check_state, warm_engine
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU")
        torch.manual_seed(settings.seed)
        torch.cuda.manual_seed_all(settings.seed)
        save(out/"environment.json", collect_environment(ROOT, protocol["model"]["path"],
                                                         protocol["model"]["revision"]))
        engine, config = make_engine(protocol, args.policy)
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 KV required")
        warmup = warm_engine(engine, config, trace)
        if args.diagnostic:
            original_step = engine.step
            def checked_step():
                before = {i: len(r.output_token_ids) for i, r in engine.by_seq.items()}
                result = original_step()
                if any(len(r.output_token_ids)-before.get(i, 0) not in (0, 1)
                       for i, r in engine.by_seq.items()):
                    raise AssertionError("duplicate sampling")
                check_state(engine)
                return result
            engine.step = checked_step
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = execute_trace(engine, config, trace, specs, settings.drain_s, "full")
        torch.cuda.synchronize()
        end_state = True
        try:
            check_state(engine)
        except AssertionError:
            end_state = False
            save(out/"state_error.json", {"message": traceback.format_exc()})
        unchanged = sources() == protocol["source_sha256"]
        summary = analyze_online(trace, result.records, engine.steps, result.backlog,
                                 result.end_ns, settings.starvation_ms)
        summary.update(**identity, valid_measurement=bool(not result.error and end_state and unchanged),
            source_unchanged=unchanged, end_state_check_passed=end_state,
            end_ns=result.end_ns, deadline_ns=result.deadline_ns,
            kv_blocks=config.num_kvcache_blocks,
            kv_payload_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            memory_scope="PyTorch allocator measured trace INCLUDING drain; not total NVML VRAM",
            warmup=warmup, stage="V2-stage1-smoke")
        for name, value in (("summary.json", summary), ("requests.json", [asdict(r) for r in result.records]),
                            ("steps.json", engine.steps), ("backlog.json", result.backlog)):
            save(out/name, value)
        write_csv(out/"steps.csv", engine.steps)
        if result.error:
            save(out/"error.json", {"message": result.error})
        print({k: summary[k] for k in ("policy", "mode", "valid_measurement", "all_completed",
                                      "ttft_p99_ms", "itl_p99_ms", "window_throughput_tok_s")}, flush=True)
        code = 0 if summary["valid_measurement"] and summary["all_completed"] else 2
    except Exception:
        if not (out/"error.json").exists():
            save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
