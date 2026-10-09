"""管理实验生命周期、有限Drain及产物保存；重负载依赖仅在运行时导入。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments

from .driver import execute_trace
from .environment import collect_environment, save_json, sha256_file
from .metrics import request_metrics, summarize
from .workload import load_trace, trace_sha256


def write_csv(path: Path, rows: list[dict], empty_fields=()):
    fields = list(rows[0]) if rows else list(empty_fields)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--observation", choices=["native", "minimal", "full"], default="full")
    parser.add_argument("--sampler", choices=["greedy", "original"], default="greedy")
    parser.add_argument("--drain-s", type=float, default=60)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return resolve_model_arguments(parser.parse_args(), parser)


def main():
    args = parse_args()
    if not math.isfinite(args.drain_s) or args.drain_s < 0:
        raise ValueError("drain-s必须有限且非负")
    if min(args.max_model_len, args.max_num_seqs, args.token_budget) < 1:
        raise ValueError("长度、序列数和Token Budget必须为正")
    if args.block_size <= 0 or args.block_size % 256:
        raise ValueError("冻结上游要求Block Size为256的正整数倍")
    if not 1 <= args.tp <= 8 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("TP或显存比例超出有效范围")
    if args.graph and args.max_num_seqs < 8:
        raise ValueError("冻结上游Graph包含bs=8，max-num-seqs至少为8")

    meta, specs = load_trace(args.trace)
    if meta["model_revision"] != args.model_revision:
        raise ValueError("Trace与模型revision不一致")
    if meta.get("model_config_sha256") is not None:
        if sha256_file(Path(args.model) / "config.json") != meta["model_config_sha256"]:
            raise ValueError("Trace生成后模型config.json已变化")

    from .preflight import check_environment
    checks = check_environment(args.model, args.tp)
    if checks["errors"]:
        raise SystemExit(json.dumps(checks, indent=2, ensure_ascii=False))
    import torch
    from .observed_engine import ObservedEngine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(args.trace, out / "trace.jsonl")
    save_json(out / "run_config.json", vars(args))
    engine_kwargs = {
        "tensor_parallel_size": args.tp, "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": args.token_budget,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kvcache_block_size": args.block_size, "enforce_eager": not args.graph,
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine = None
    try:
        engine = ObservedEngine(args.model, observation=args.observation,
                                sampler=args.sampler, **engine_kwargs)
        environment = collect_environment(project_root, args.model, args.model_revision)
        environment.update(run_config=vars(args), engine_kwargs=engine_kwargs,
                           trace_sha256=trace_sha256(args.trace), time_origin="relative_perf_counter_ns")
        config, kv_cache = engine.model_runner.config, engine.model_runner.kv_cache
        environment["runtime"] = {
            "effective_max_model_len": config.max_model_len, "kv_dtype": str(kv_cache.dtype),
            "rank0_kv_allocated_bytes": kv_cache.numel() * kv_cache.element_size(),
            "num_kvcache_blocks": config.num_kvcache_blocks,
            "graph_capture_count": len(getattr(engine.model_runner, "graphs", {})),
        }
        result = execute_trace(engine, config, meta, specs, args.drain_s, args.observation)
        warmup_ns = round(meta["warmup_s"] * 1e9)
        end_ns = round((meta["warmup_s"] + meta["measurement_s"]) * 1e9)
        summary = summarize(result.records, warmup_ns, end_ns, args.observation != "native")
        summary.update({
            "schema_version": 2, "valid_run": result.error is None, "empty_trace": not specs,
            "observation": args.observation, "sampler": args.sampler,
            "seed": args.seed, "tp": args.tp, "rps": meta["rps"],
            "driver_elapsed_s": result.end_ns / 1e9,
            "deadline_overrun_ms": max(0, result.end_ns - result.deadline_ns) / 1e6,
            "step_count": engine.step_count,
            "preemptions": sum(r.preemptions or 0 for r in result.records) if args.observation == "full" else None,
            "recomputed_tokens": sum(r.recomputed_tokens or 0 for r in result.records) if args.observation == "full" else None,
        })
        environment["schema_version"] = 2
        save_json(out / "environment.json", environment)
        save_json(out / "summary.json", summary)
        write_csv(out / "summary.csv", [summary])
        write_csv(out / "steps.csv", engine.steps, ["step_id", "start_ns", "end_ns"])
        write_csv(out / "backlog.csv", result.backlog, ["time_ns", "backlog_total"])
        with (out / "requests.jsonl").open("w", encoding="utf-8") as file:
            for record in result.records:
                row = asdict(record)
                row["schema_version"] = 2
                row["metrics"] = request_metrics(record) if args.observation != "native" else None
                if args.observation == "native":
                    row["token_times_ns"] = None
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if result.error:
            (out / "error.txt").write_text(result.error)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if result.error:
            raise SystemExit(1)
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    main()
