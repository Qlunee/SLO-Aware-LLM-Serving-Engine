"""Candidate run, legacy Greedy reference and fixed-history logits comparison."""
import argparse
import atexit
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter_ns

from benchmarks.defaults import add_model_arguments, resolve_model_arguments


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def candidate(args):
    import torch
    from benchmarks.v0.accounting import merge_interval
    from benchmarks.v0.environment import collect_environment
    from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
    from nanovllm.sampling_params import SamplingParams

    out = new_directory(args.out)
    root = Path(__file__).resolve().parents[2]
    engine_kwargs = dict(
        tensor_parallel_size=args.tp, enforce_eager=not args.graph,
        max_num_seqs=max(32, args.batch), max_num_batched_tokens=args.token_budget,
        max_model_len=4096, gpu_memory_utilization=args.memory,
    )
    environment = collect_environment(root, args.model, args.model_revision)
    environment["v1_benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment["v1_source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))}
    save(out / "environment.json", environment)
    save(out / "config.json", {**vars(args), "engine_kwargs": engine_kwargs})
    torch.manual_seed(args.seed)
    engine = MixedLLMEngine(args.model, chunk_size=args.chunk, kv_block_cap=args.kv_block_cap,
                            verify_tp_plan=True, **engine_kwargs)
    specs, steps, teacher, events, intervals = [], [], [], {}, {}
    recomputed = 0
    model_calls = [0]
    def count_forward(module, inputs):
        model_calls[0] += 1
    forward_handle = engine.model_runner.model.register_forward_pre_hook(count_forward)
    rng = random.Random(args.seed)
    prompt_index = 0

    def prompt(length):
        nonlocal prompt_index
        prompt_index += 1
        if args.prompt_kind == "random":
            return [rng.randrange(100, 2000) for _ in range(length)]
        from benchmarks.v1.fixtures import language_prompt
        return language_prompt(engine.tokenizer, length, prompt_index)

    def add(tokens):
        seq_id = engine.add_request(tokens, SamplingParams(max_tokens=8, ignore_eos=True))
        specs.append({"seq_id": seq_id, "prompt_token_ids": tokens, "max_output_tokens": 8})
        events[seq_id], intervals[seq_id] = [], []

    def one_step():
        nonlocal recomputed
        if len(steps) >= 10000:
            raise RuntimeError("step limit reached")
        start = perf_counter_ns()
        previous_calls = model_calls[0]
        engine.step(return_logits=True)
        visible = perf_counter_ns()
        engine.scheduler.assert_consistent()
        state = engine.last_step
        batch = state["batch"]
        forwards = model_calls[0] - previous_calls
        expected_forwards = 0 if state["graph_mode"] == "cuda_graph" else 1
        if forwards != expected_forwards:
            raise AssertionError(f"expected {expected_forwards} Python model forwards, got {forwards}")
        steps.append({"step": len(steps), "start_ns": start, "visible_ns": visible,
                      "prefill_tokens": batch.num_prefill_tokens,
                      "decode_tokens": batch.num_decode_tokens,
                      "graph_mode": state["graph_mode"],
                      "python_model_forwards": forwards,
                      "requests": [asdict(r) for r in batch.requests]})
        for row in batch.requests:
            overlap, merged = merge_interval(intervals[row.seq_id], row.start_pos, row.end_pos)
            intervals[row.seq_id] = merged
            recomputed += overlap
        for i, seq_id in enumerate(batch.sample_seq_ids):
            events[seq_id].append(visible)
            row = next(r for r in batch.requests if r.seq_id == seq_id)
            new_phase = ((seq_id, row.phase) not in
                         {(s["seq_id"], s["phase"]) for s in teacher})
            if (len(teacher) < args.logit_samples or new_phase) and state["logits"] is not None:
                history = engine.scheduler.sequences[seq_id].token_ids[:row.logical_tokens]
                teacher.append({"seq_id": seq_id, "history": list(history),
                                "phase": row.phase, "logits": state["logits"][i].clone()})

    def drain():
        while not engine.is_finished():
            one_step()

    try:
        if args.case == "mixed":
            add(prompt(256))
            one_step()
            for i in range(args.batch - 1):
                add(prompt([255, 257, 513, 1025][i % 4]))
            drain()
        elif args.case == "prefix":
            shared = prompt(512)
            add(shared + [101])
            drain()
            add(shared + [102])
            drain()
        else:
            add(prompt(256))
            add(prompt(256))
            drain()
        for spec in specs:
            seq = engine.scheduler.sequences[spec["seq_id"]]
            spec["output_token_ids"] = list(seq.completion_token_ids)
            if not seq.is_finished or len(spec["output_token_ids"]) != spec["max_output_tokens"]:
                raise AssertionError("request lost or duplicate output")
        mixed_steps = sum(s["decode_tokens"] > 0 and s["prefill_tokens"] > 0 for s in steps)
        if args.case == "mixed" and args.batch > 1 and mixed_steps == 0:
            raise AssertionError("test never exercised mixed execution")
        if args.case == "prefix" and not any(r["start_pos"] >= 512
                and r["phase"] == "prefill" for s in steps for r in s["requests"]):
            raise AssertionError("test did not exercise cached prefix")
        if args.case == "preempt" and engine.scheduler.preemption_count == 0:
            raise AssertionError("test did not force preemption")
        if args.graph and not any(s["graph_mode"] == "cuda_graph" for s in steps):
            raise AssertionError("graph test never replayed a decode graph")
        report = {"passed_state_checks": True, "requests": len(specs),
                  "mixed_steps": mixed_steps, "steps": len(steps),
                  "preemptions": engine.scheduler.preemption_count,
                  "actual_recomputed_tokens": recomputed,
                  "graph_replays": sum(s["graph_mode"] == "cuda_graph" for s in steps),
                  "note": "Correctness run includes logits copies / assertions / TP digest checks; NOT a latency benchmark."}
        save(out / "requests.json", specs)
        save(out / "steps.json", steps)
        save(out / "token_events_ns.json", events)
        torch.save(teacher, out / "teacher_forced.pt")
        save(out / "report.json", report)
        # Hash the exact inputs and evidence consumed by the reference command.
        save(out / "artifact_hashes.json", {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
             for name in ["requests.json", "teacher_forced.pt", "config.json", "environment.json"]})
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as exc:
        save(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        forward_handle.remove()
        engine.exit()


def reference(args):
    import torch
    import torch.nn.functional as F
    from benchmarks.v0.environment import source_hashes, model_metadata_hashes
    from benchmarks.v0.observed_engine import GreedySampler
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams
    from benchmarks.v1.numerics import numerical_pass

    folder = Path(args.candidate)
    out = new_directory(args.out)
    hashes = json.loads((folder / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"candidate artifact changed: {name}")
    config = json.loads((folder / "config.json").read_text())
    environment = json.loads((folder / "environment.json").read_text())
    if source_hashes(Path(__file__).resolve().parents[2]) != environment["source_sha256"]:
        raise ValueError("source changed after candidate run")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != environment["v1_benchmark_sha256"]:
        raise ValueError("V1 checker changed after candidate run")
    current_support = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(__file__).parent.glob("*.py"))}
    if current_support != environment["v1_source_sha256"]:
        raise ValueError("V1 support source changed after candidate run")
    if model_metadata_hashes(Path(config["model"])) != environment["model_metadata_sha256"]:
        raise ValueError("model metadata changed")
    specs = json.loads((folder / "requests.json").read_text())
    teacher = torch.load(folder / "teacher_forced.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(config["seed"])
    engine = LLMEngine(config["model"], **config["engine_kwargs"])
    atexit.unregister(engine.exit)
    engine.model_runner.sampler = GreedySampler()
    try:
        expected = engine.generate([s["prompt_token_ids"] for s in specs],
                                  [SamplingParams(max_tokens=8, ignore_eos=True) for _ in specs],
                                  use_tqdm=False)
        greedy_cases = [{"seq_id": s["seq_id"], "match": s["output_token_ids"] == ref["token_ids"],
                         "candidate": s["output_token_ids"], "reference": ref["token_ids"]}
                        for s, ref in zip(specs, expected)]
        if len(expected) != len(specs):
            raise AssertionError("reference request count")
        rows = []
        captured = []

        def hook(module, inputs, output):
            if output is not None:
                captured.append(output[-1].detach().float().cpu())

        handle = engine.model_runner.model.lm_head.register_forward_hook(hook)
        try:
            for sample in teacher:
                captured.clear()
                engine.generate([sample["history"]], SamplingParams(max_tokens=1, ignore_eos=True),
                                use_tqdm=False)
                ref, actual = captured[-1], sample["logits"]
                diff = (actual - ref).abs()
                best = ref.topk(2).values
                rows.append({"seq_id": sample["seq_id"], "phase": sample["phase"],
                             "history_length": len(sample["history"]),
                             "mae": diff.mean().item(), "max_abs_error": diff.max().item(),
                             "relative_l2": ((actual - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
                             "cosine": F.cosine_similarity(actual, ref, dim=0).item(),
                             "reference_argmax_margin": (best[0] - best[1]).item(),
                             "top1_match": actual.argmax().item() == ref.argmax().item(),
                             "allclose": torch.allclose(actual, ref, atol=args.atol, rtol=args.rtol)})
        finally:
            handle.remove()
    finally:
        engine.exit()
    passed = (bool(rows) and all(r["match"] for r in greedy_cases)
              and all(numerical_pass(r, args.numerical_policy, args.relative_l2_limit,
                                    args.cosine_min, args.max_abs_limit) for r in rows))
    report = {"passed": passed, "atol": args.atol, "rtol": args.rtol,
              "numerical_policy": args.numerical_policy,
              "relative_l2_limit": args.relative_l2_limit, "cosine_min": args.cosine_min,
              "max_abs_limit": args.max_abs_limit,
              "strict_allclose_passed": all(r["allclose"] for r in rows),
              "greedy_cases": greedy_cases, "teacher_forced": rows,
              "note": "Greedy/Top-1 disagreement always fails. BF16 policy uses explicit provisional norm/cosine/max-absolute limits; elementwise allclose is still reported. Strict policy gates on allclose. Neither substitutes for state/attention correctness."}
    save(out / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidate")
    add_model_arguments(c)
    c.add_argument("--tp", type=int, default=1)
    c.add_argument("--graph", action="store_true")
    c.add_argument("--batch", type=int, choices=[1, 2, 8, 32], default=8)
    c.add_argument("--chunk", type=int, choices=[256, 512, 1024, 2048], default=256)
    c.add_argument("--token-budget", type=int, default=512)
    c.add_argument("--memory", type=float, default=0.5)
    c.add_argument("--seed", type=int, default=41)
    c.add_argument("--prompt-kind", choices=["natural", "random"], default="natural")
    c.add_argument("--logit-samples", type=int, default=8)
    c.add_argument("--kv-block-cap", type=int)
    c.add_argument("--case", choices=["mixed", "prefix", "preempt"], default="mixed")
    c.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--candidate", required=True)
    r.add_argument("--atol", type=float, default=0.15)
    r.add_argument("--rtol", type=float, default=0.03)
    r.add_argument("--numerical-policy", choices=["strict", "bf16"], default="strict")
    r.add_argument("--relative-l2-limit", type=float, default=0.02)
    r.add_argument("--cosine-min", type=float, default=0.999)
    r.add_argument("--max-abs-limit", type=float, default=0.5)
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "candidate":
        resolve_model_arguments(args, parser)
        if args.token_budget < 256 or args.tp not in [1, 2] or args.logit_samples < 1:
            parser.error("V1 GPU checks require budget>=256, TP=1/2 and logit-samples>=1")
        if not 0 < args.memory < 1:
            parser.error("memory must be in (0,1)")
        if args.case == "preempt" and (args.kv_block_cap != 2 or args.token_budget < 512):
            parser.error("preemption test requires --kv-block-cap 2 --token-budget >=512")
        candidate(args)
    else:
        import math
        if (not all(math.isfinite(v) for v in [args.atol, args.rtol, args.relative_l2_limit,
                                               args.cosine_min, args.max_abs_limit])
                or min(args.atol, args.rtol, args.relative_l2_limit, args.max_abs_limit) < 0
                or not -1 <= args.cosine_min <= 1):
            parser.error("invalid tolerance")
        reference(args)


if __name__ == "__main__":
    main()
