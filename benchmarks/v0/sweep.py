## 逐个启动独立进程，避免引擎退出、NCCL和CUDA状态影响下一次运行。
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments


def execute(arguments: list[str], log_path: Path):
    with log_path.open("w") as log:
        subprocess.run(
            arguments,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--warmup-s", type=float, default=10)
    parser.add_argument("--measurement-s", type=float, default=60)
    parser.add_argument("--drain-s", type=float, default=60)
    parser.add_argument("--input-lengths", type=int, nargs="+",
                        default=[256, 1024, 2048])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--itl-slo-ms", type=float, default=100)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--observation", choices=["native", "minimal", "full"], default="full")
    parser.add_argument("--sampler", choices=["greedy", "original"], default="greedy")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--out", required=True)
    args = resolve_model_arguments(parser.parse_args(), parser)

    if any(not math.isfinite(rate) or rate <= 0 for rate in args.rates):
        parser.error("rates必须有限且为正")
    if len(set(args.rates)) != len(args.rates) or len(set(args.seeds)) != len(args.seeds):
        parser.error("rates和seeds不得重复")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    summaries = []

    for rate in args.rates:
        for seed in args.seeds:
            name = f"rps_{rate:g}_seed_{seed}"
            trace = out / f"{name}.jsonl"
            run_dir = out / name

            workload_command = [
                sys.executable, "-m", "benchmarks.v0.workload",
                "--model", args.model,
                "--model-revision", args.model_revision,
                "--rps", str(rate),
                "--seed", str(seed),
                "--warmup-s", str(args.warmup_s),
                "--measurement-s", str(args.measurement_s),
                "--output-tokens", str(args.output_tokens),
                "--ttft-slo-ms", str(args.ttft_slo_ms),
                "--itl-slo-ms", str(args.itl_slo_ms),
                "--input-lengths",
                *map(str, args.input_lengths),
                "--out", str(trace),
            ]
            execute(workload_command, out / f"{name}_trace.log")

            run_command = [
                sys.executable, "-m", "benchmarks.v0.run",
                "--model", args.model,
                "--model-revision", args.model_revision,
                "--trace", str(trace),
                "--out", str(run_dir),
                "--seed", str(seed),
                "--tp", str(args.tp),
                "--drain-s", str(args.drain_s),
                "--observation", args.observation,
                "--sampler", args.sampler,
                "--max-model-len", str(args.max_model_len),
                "--max-num-seqs", str(args.max_num_seqs),
                "--token-budget", str(args.token_budget),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            ]
            if args.graph:
                run_command.append("--graph")

            execute(run_command, out / f"{name}_run.log")
            summary = json.loads(
                (run_dir / "summary.json").read_text()
            )
            if not summary["valid_run"]:
                raise RuntimeError(f"无效实验：{run_dir}")
            summaries.append(summary)
            # 每次成功都更新，后续运行失败不丢失已完成结果。
            with (out / "sweep_summary.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)

    with (out / "sweep_summary.csv").open("w", newline="") as file:
        if summaries:
            writer = csv.DictWriter(
                file, fieldnames=list(summaries[0])
            )
            writer.writeheader()
            writer.writerows(summaries)

    print(f"完成{len(summaries)}次运行：{out}")


if __name__ == "__main__":
    main()
