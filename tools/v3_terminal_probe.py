"""Real-engine collector for terminal mixed Prefill chunks.

The frozen V3 collector inserts prompts longer than its chunk, so its first
full Prefill chunks do not sample a token.  This probe inserts a prompt exactly
one chunk long: its first mixed Step both fills KV and samples the first token.
Diagnostic runs are never training inputs.  A prospective suite may designate
new runs as train/eval and fit them with its separate, validated fitter.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import traceback

from benchmarks.v0.environment import collect_environment, command_output
from benchmarks.v3_profile.protocol import (ROOT, Settings, predictor_module,
    runtime_identity, read, save, seal, sha, sources, verify_artifacts)
from benchmarks.v3_profile.timing import timed_step
from tools.v3_refine import load_suite


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def csv_ints(value):
    try:
        values = tuple(int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(x < 1 for x in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct positive integers")
    return values


def prepare(args):
    plan = load_suite(args.suite)
    predictor_path = Path(args.predictor).resolve()
    verify_artifacts(predictor_path.parent)
    table = read(predictor_path)
    if table["identity"] != plan["identity"]:
        raise ValueError("predictor and frozen suite have different runtime identities")
    if args.seed in {x["seed"] for x in table["training_profiles"]}:
        raise ValueError("diagnostic seed overlaps predictor training")
    settings = Settings(**plan["settings"])
    settings.validate()
    cases = []
    for d in args.decode_counts:
        for p in args.prefill_tokens:
            for c in args.contexts:
                if args.context_pattern == "one_long" and c <= 256:
                    raise ValueError("one_long requires a context above 256 tokens")
                decode_contexts = ([c] * d if args.context_pattern == "uniform"
                                   else [c] + [256] * (d - 1))
                cases.append({"decode_count": d, "context_tokens": c,
                    "decode_contexts": decode_contexts, "prefill_tokens": p,
                    "context_pattern": args.context_pattern,
                    "case_id": f"terminal-d{d}-p{p}-c{c}-{args.context_pattern}"})
    for case in cases:
        d, p, c = (case[x] for x in ("decode_count", "prefill_tokens", "context_tokens"))
        output_limit = 4 * (args.repeats + args.warmup) + math.ceil(c / 256) * d + 16
        if (d + p > settings.token_budget or d + 1 > settings.max_num_seqs
                or max(c + output_limit, p + 1) > settings.max_model_len):
            raise ValueError(f"case exceeds frozen model/budget limits: {case['case_id']}")
        blocks = sum(math.ceil((length + output_limit) / 256)
                     for length in case["decode_contexts"]) + math.ceil((p + 1) / 256)
        if blocks > settings.kv_block_cap:
            raise ValueError(f"case exceeds frozen KV cap: {case['case_id']}")
        case["base_output_limit"] = output_limit
    return plan, predictor_path, table, settings, cases


def run_case(engine, case, args, settings, add, observe):
    scheduler = engine.scheduler
    d, p, c = (case[x] for x in ("decode_count", "prefill_tokens", "context_tokens"))
    scheduler.max_num_batched_tokens = settings.token_budget
    scheduler.chunk_size = 256
    base = [add(length, case["base_output_limit"])
            for length in case["decode_contexts"]]
    for _ in range(10000):
        if all(scheduler.sequences[i].status.name == "RUNNING" for i in base):
            break
        engine.step()
    else:
        raise RuntimeError("Decode bootstrap exhausted")
    scheduler.max_num_batched_tokens = d + p
    scheduler.chunk_size = p
    for episode in range(args.warmup + args.repeats):
        inserted = add(p, 1)
        batch, result, times = timed_step(engine)
        if batch.num_decode_tokens != d or batch.num_prefill_tokens != p:
            raise RuntimeError("target terminal mixed shape was not scheduled")
        if len(batch.sample_indices) != d + 1:
            raise RuntimeError("terminal chunk did not sample the first output token")
        observe(case, episode, batch, result, times)
        if not scheduler.sequences[inserted].is_finished:
            raise RuntimeError("terminal Prefill request did not finish")
        if any(scheduler.sequences[i].is_finished for i in base):
            raise RuntimeError("Decode base finished during probe")
        if scheduler.preemption_count:
            raise RuntimeError("probe unexpectedly preempted")
    scheduler.max_num_batched_tokens = settings.token_budget
    scheduler.chunk_size = 256
    for _ in range(10000):
        if engine.is_finished():
            break
        engine.step()
    if not engine.is_finished():
        raise RuntimeError("finite drain exhausted")
    scheduler.assert_consistent()
    if scheduler.block_manager.used_block_ids:
        raise AssertionError("KV blocks leaked")
    scheduler.sequences.clear()


def collect(args):
    plan, predictor_path, table, settings, cases = prepare(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    engine = None
    recorded, completed = 0, []
    initial_sources = sources()
    try:
        import torch
        from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
        from nanovllm.sampling_params import SamplingParams
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        rng = random.Random(args.seed)
        environment = collect_environment(ROOT, plan["model_path"], plan["identity"]["model_revision"])
        environment["v3_driver"] = command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        engine = MixedLLMEngine(plan["model_path"], chunk_size=256,
            kv_block_cap=settings.kv_block_cap, greedy=True, verify_tp_plan=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=settings.max_model_len,
            max_num_seqs=settings.max_num_seqs, max_num_batched_tokens=settings.token_budget,
            gpu_memory_utilization=settings.memory, kvcache_block_size=256)
        identity = json.loads(json.dumps(runtime_identity(
            environment, engine.config, engine.model_runner.kv_cache)))
        if identity != plan["identity"]:
            raise ValueError("runtime differs from frozen V3 suite")
        predictor = predictor_module().StepPredictor(table, plan["identity"])
        estimator = predictor_module()
        save(out / "environment.json", environment)
        save(out / "protocol.json", {"schema": "v3-terminal-probe-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "split": args.split,
            "cases": cases, "repeats": args.repeats, "warmup": args.warmup,
            "settings": asdict(settings), "identity": plan["identity"],
            "predictor_sha256": sha(predictor_path),
            "probe_source_sha256": sha(Path(__file__)),
            "scope": ("diagnostic; do not pool with training" if args.split == "diagnostic"
                      else "prospective terminal mixed profile for independent suite")})
        vocab = engine.config.hf_config.vocab_size
        special = set(engine.tokenizer.all_special_ids)

        def add(length, outputs):
            tokens = []
            while len(tokens) < length:
                token = rng.randrange(vocab)
                if token not in special:
                    tokens.append(token)
            return engine.add_request(tokens, SamplingParams(max_tokens=outputs, ignore_eos=True))

        rng.shuffle(cases)
        with (out / "steps.jsonl").open("x", encoding="utf-8") as stream:
            def observe(case, episode, batch, result, times):
                nonlocal recorded
                features = estimator.batch_features(batch, result["graph_mode"])
                prediction = predictor.predict(features)
                row = {"step_id": recorded, "case_id": case["case_id"],
                    "episode": episode, "warmup": episode < args.warmup,
                    "features": features, "prediction": asdict(prediction), **times}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                recorded += 1

            for case in cases:
                run_case(engine, case, args, settings, add, observe)
                completed.append(case["case_id"])
                print(f"[{len(completed)}/{len(cases)}] {case['case_id']}", flush=True)
        if sources() != initial_sources:
            raise ValueError("execution sources changed during probe")
        save(out / "summary.json", {"valid": True, "completed_cases": completed,
            "recorded_steps": recorded, "measured_steps": len(cases) * args.repeats,
            "preemptions": engine.scheduler.preemption_count, "source_unchanged": True})
    except BaseException:
        save(out / "error.json", {"error": traceback.format_exc(),
            "recorded_steps": recorded, "completed_cases": completed})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=715)
    parser.add_argument("--split", choices=("diagnostic", "train", "eval"),
                        default="diagnostic")
    parser.add_argument("--decode-counts", type=csv_ints, default=(1, 2, 3))
    parser.add_argument("--prefill-tokens", type=csv_ints, default=(256,))
    parser.add_argument("--contexts", type=csv_ints, default=(256,))
    parser.add_argument("--context-pattern", choices=("uniform", "one_long"),
                        default="uniform")
    parser.add_argument("--repeats", type=positive_int, default=40)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not args.plan_only and not args.out:
        parser.error("--out is required for collection")
    plan, predictor_path, table, settings, cases = prepare(args)
    if args.plan_only:
        print(json.dumps({"cases": cases, "measured_steps": len(cases) * args.repeats,
            "predictor": str(predictor_path), "suite": str(Path(args.suite).resolve()),
            "runtime_identity_matches": True}, indent=2))
    else:
        print(json.dumps({"out": str(collect(args))}, indent=2))


if __name__ == "__main__":
    main()
