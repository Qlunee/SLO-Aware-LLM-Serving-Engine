"""Compare audited V0/V1 runs with identical trace/config; no performance claim implied."""
import argparse
import bisect
import csv
import json
from pathlib import Path

from benchmarks.v0.metrics import quantile
from benchmarks.v0.workload import load_trace, trace_sha256


def prefill_overlap_itls(records, steps, start_ns, end_ns):
    """ITLs ending in the measurement window, overlapping a Prefill-containing step."""
    prefill = sorted((int(s["start_ns"]), int(s["end_ns"])) for s in steps
                     if int(s.get("prefill_tokens") or 0) > 0)
    starts, ends = [x[0] for x in prefill], [x[1] for x in prefill]
    if any(a >= b for a, b in prefill) or any(b > c for (_, b), (c, _) in zip(prefill, prefill[1:])):
        raise ValueError("step intervals overlap or have invalid duration")
    values = []
    for record in records:
        times = record.get("token_times_ns") or []
        for a, b in zip(times, times[1:]):
            if start_ns <= b < end_ns and bisect.bisect_left(starts, b) > bisect.bisect_right(ends, a):
                values.append((b - a) / 1e6)
    return {"count": len(values), "max_ms": max(values) if values else None,
            "p99_ms": quantile(values, 0.99)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    folders = [Path(args.baseline), Path(args.candidate)]
    if trace_sha256(folders[0] / "trace.jsonl") != trace_sha256(folders[1] / "trace.jsonl"):
        raise ValueError("trace differs")
    envs = [json.loads((p / "environment.json").read_text()) for p in folders]
    for key in ["model_revision_declared", "model_metadata_sha256"]:
        if envs[0].get(key) != envs[1].get(key):
            raise ValueError(f"model differs: {key}")
    configs = [e["run_config"] for e in envs]
    for key in ["observation", "sampler", "drain_s", "tp", "max_model_len", "max_num_seqs",
                "token_budget", "gpu_memory_utilization", "block_size", "graph", "seed"]:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"unfair configuration difference: {key}")
    if configs[0]["observation"] == "native":
        raise ValueError("native has no token timing")
    meta, _ = load_trace(folders[0] / "trace.jsonl")
    start, end = round(meta["warmup_s"] * 1e9), round((meta["warmup_s"] + meta["measurement_s"]) * 1e9)
    keys = ["completed_count", "unfinished_count", "cohort_violation_rate", "window_goodput_rps",
            "window_throughput_tok_s", "ttft_p99_ms", "itl_p99_ms", "max_itl_ms",
            "preemptions", "recomputed_tokens"]
    result = {"trace_sha256": trace_sha256(folders[0] / "trace.jsonl"),
              "note": "Run validate_results on both first. Prefill-overlap ITL is temporal overlap, not causal proof. Correctness-check timings are not benchmark timings."}
    for label, folder in zip(["baseline", "candidate"], folders):
        summary = json.loads((folder / "summary.json").read_text())
        if not summary["valid_run"]:
            raise ValueError(f"invalid run: {folder}")
        result[label] = {k: summary.get(k) for k in keys}
        if configs[0]["observation"] == "full":
            records = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
            with (folder / "steps.csv").open() as stream:
                steps = list(csv.DictReader(stream))
            result[label]["prefill_overlap_itl"] = prefill_overlap_itls(records, steps, start, end)
    result["candidate_minus_baseline"] = {
        k: result["candidate"][k] - result["baseline"][k] for k in keys
        if isinstance(result["candidate"][k], (int, float)) and isinstance(result["baseline"][k], (int, float))}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with Path(args.out).open("x") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
