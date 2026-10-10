"""Independent semantics, observational BF16 calibration, and long-output sanity.

No command here overrides benchmarks.v1.check_v1 or certifies all V1 matrices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import torch

from benchmarks.defaults import add_model_arguments, resolve_model_arguments
from benchmarks.v1_validation.metrics import distribution_metrics, sequence_metrics, summarize

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validation_hashes():
    files = [*Path(__file__).parent.glob("*.py"), *ROOT.glob("tests/test_v1*py"),
             ROOT / "scripts/test_v1_validation.sh"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}


def make_engine(args, fp32=False):
    from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
    from benchmarks.v0.observed_engine import GreedySampler
    engine = MixedLLMEngine(args.model, chunk_size=args.chunk, greedy=True,
        verify_tp_plan=True, tensor_parallel_size=args.tp, enforce_eager=not args.graph,
        max_num_seqs=32, max_num_batched_tokens=args.token_budget,
        max_model_len=4096, gpu_memory_utilization=args.memory,
        kv_block_cap=8 if fp32 else None)
    engine.model_runner.sampler = GreedySampler()
    return engine


def cold_allocator(engine):
    bm = engine.scheduler.block_manager
    if not engine.is_finished() or bm.used_block_ids:
        raise RuntimeError("cannot reset prefix lookup while requests are active")
    bm.hash_to_block_id.clear()
    return bm


def legacy_logits(engine, history, chunk):
    """Original Prefill execution, no Mixed Attention, cold KV, fixed history.

    The inherited runner/weights are shared, NOT a second independent model.
    Manual chunk boundaries isolate original execution-path numerical drift.
    """
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    if len(history) > engine.config.max_model_len:
        raise ValueError("history exceeds model length")
    bm = cold_allocator(engine)
    seq = Sequence(history, SamplingParams(max_tokens=1, ignore_eos=True))
    if seq.num_blocks > len(bm.blocks):
        raise ValueError("history exceeds physical KV capacity")
    bm.allocate(seq, 0)
    captured = []
    def hook(module, inputs, output):
        if output is not None:
            captured.append(output[-1].detach().float().cpu().clone())
    handle = engine.model_runner.model.lm_head.register_forward_hook(hook)
    try:
        for start in range(0, len(history), chunk):
            seq.num_cached_tokens = start
            seq.num_scheduled_tokens = min(chunk, len(history)-start)
            captured.clear()
            engine.model_runner.call("run", [seq], True)
        if len(captured) != 1:
            raise AssertionError("missing teacher-forced logits")
        return captured[0]
    finally:
        handle.remove()
        bm.deallocate(seq)


def mixed_logits(engine, history, chunk):
    from nanovllm.sampling_params import SamplingParams
    cold_allocator(engine)
    engine.scheduler.chunk_size = chunk
    seq_id = engine.add_request(history, SamplingParams(max_tokens=1, ignore_eos=True))
    for _ in range(10000):
        engine.step(return_logits=True)
        engine.scheduler.assert_consistent()
        if engine.is_finished():
            batch = engine.last_step["batch"]
            index = batch.sample_seq_ids.index(seq_id)
            return engine.last_step["logits"][index].clone()
    raise RuntimeError("teacher-forced step limit")


def histories(args, tokenizer):
    from benchmarks.v1.fixtures import language_prompt
    rows = [{"id": f"seed-{seed}-len-{length}", "tokens": language_prompt(tokenizer, length, seed),
             "source": "synthetic_language_fixture"}
            for seed in args.seeds for length in args.lengths]
    if args.artifact:
        folder = Path(args.artifact)
        manifest = json.loads((folder / "artifact_hashes.json").read_text())
        for name, digest in manifest.items():
            if Path(name).name != name:
                raise ValueError("invalid artifact manifest path")
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"tampered candidate artifact: {name}")
        config = json.loads((folder / "config.json").read_text())
        if Path(config["model"]).resolve() != Path(args.model).resolve() or config["model_revision"] != args.model_revision:
            raise ValueError("artifact model does not match this validation run")
        # Inputs only: do NOT compare old source-version logits as new acceptance evidence.
        teacher = torch.load(folder / "teacher_forced.pt", map_location="cpu", weights_only=True)
        sample = next((s for s in teacher if s["seq_id"] == args.seq_id and s["phase"] == "prefill"), None)
        if sample is None:
            raise ValueError("requested artifact Prefill sample not found")
        rows.append({"id": f"artifact-seq-{args.seq_id}", "tokens": sample["history"],
                     "source": str(folder.resolve())})
    return rows


def legacy_generate(engine, prompts, output_tokens):
    """Original Scheduler + original runner.run, sharing loaded weights only."""
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    cold_allocator(engine)
    scheduler = Scheduler(engine.config)
    sequences = [Sequence(p, SamplingParams(max_tokens=output_tokens, ignore_eos=True)) for p in prompts]
    for seq in sequences:
        scheduler.add(seq)
    for _ in range(100000):
        if scheduler.is_finished():
            break
        seqs, prefill = scheduler.schedule()
        tokens = engine.model_runner.call("run", seqs, prefill)
        scheduler.postprocess(seqs, tokens, prefill)
    else:
        raise RuntimeError("legacy generation step limit")
    if scheduler.block_manager.used_block_ids:
        raise AssertionError("legacy KV leak")
    return [list(s.completion_token_ids) for s in sequences]


def mixed_generate(engine, prompts, output_tokens):
    from nanovllm.sampling_params import SamplingParams
    cold_allocator(engine)
    ids, mixed_steps, graphs = [], 0, 0
    for i, prompt in enumerate(prompts):
        ids.append(engine.add_request(prompt, SamplingParams(max_tokens=output_tokens, ignore_eos=True)))
        if i == 0:
            engine.step()  # Stagger arrival so later requests can overlap ongoing Decode.
            engine.scheduler.assert_consistent()
    for _ in range(100000):
        if engine.is_finished():
            break
        engine.step()
        engine.scheduler.assert_consistent()
        batch = engine.last_step["batch"]
        mixed_steps += bool(batch.num_decode_tokens and batch.num_prefill_tokens)
        graphs += engine.last_step["graph_mode"] == "cuda_graph"
    else:
        raise RuntimeError("mixed generation step limit")
    outputs = [list(engine.scheduler.sequences[i].completion_token_ids) for i in ids]
    if any(len(x) != output_tokens for x in outputs):
        raise AssertionError("lost/duplicated output tokens")
    return outputs, {"mixed_steps": mixed_steps, "graph_replays": graphs}


def semantics(args, out):
    from benchmarks.v1_validation.reference import install_fp32_diagnostic
    engine = make_engine(args, fp32=True)
    restore = None
    rows, evidence = [], []
    try:
        restore = install_fp32_diagnostic(engine)
        inputs = histories(args, engine.tokenizer)
        save(out / "inputs.json", inputs)
        for item in inputs:
            ref = legacy_logits(engine, item["tokens"], len(item["tokens"]))
            for chunk in args.chunks:
                actual = mixed_logits(engine, item["tokens"], chunk)
                row = {"id": item["id"], "chunk": chunk, **distribution_metrics(actual, ref)}
                row["semantic_gate_passed"] = (row["top1_match"] and row["relative_l2"] <= args.fp32_relative_limit
                                               and row["max_abs_error"] <= args.fp32_max_abs)
                rows.append(row)
                evidence.append({"id": item["id"], "chunk": chunk, "actual": actual, "reference": ref})
        from benchmarks.v1.fixtures import language_prompt
        prompts = [language_prompt(engine.tokenizer, 255, 991),
                   language_prompt(engine.tokenizer, 513, 992)]
        baseline = legacy_generate(engine, prompts, 4)
        actual, execution = mixed_generate(engine, prompts, 4)
        completions = [sequence_metrics(a, r) for a, r in zip(actual, baseline)]
        save(out / "mixed_semantics.json", {"candidate": actual, "reference": baseline,
             "comparisons": completions, **execution})
        if execution["mixed_steps"] == 0:
            raise AssertionError("semantic scenario did not exercise a mixed step")
    finally:
        if restore is not None:
            restore()
        engine.exit()
    torch.save(evidence, out / "logits.pt")
    passed = all(r["semantic_gate_passed"] for r in rows) and all(r["exact_match"] for r in completions)
    return {"status": "PASS" if passed else "FAIL", "scope": "FP32 scheduling/cache semantics, not BF16 FA/TP/Graph acceptance",
            "limits": {"relative_l2": args.fp32_relative_limit, "max_abs": args.fp32_max_abs},
            "comparisons": rows, "summary": summarize(rows), "mixed_execution": execution,
            "v1_acceptance_passed": False}, 0 if passed else 1


def calibration(args, out):
    engine = make_engine(args)
    rows, evidence = [], []
    try:
        inputs = histories(args, engine.tokenizer)
        save(out / "inputs.json", inputs)
        for item in inputs:
            ref = legacy_logits(engine, item["tokens"], len(item["tokens"]))
            for chunk in args.chunks:
                baseline = legacy_logits(engine, item["tokens"], chunk)
                actual = mixed_logits(engine, item["tokens"], chunk)
                pairs = (("legacy_chunk_vs_legacy_full", baseline, ref),
                         ("mixed_chunk_vs_legacy_full", actual, ref),
                         ("mixed_chunk_vs_legacy_chunk", actual, baseline))
                for kind, a, r in pairs:
                    rows.append({"id": item["id"], "chunk": chunk, "comparison": kind, **distribution_metrics(a, r)})
                evidence.append({"id": item["id"], "chunk": chunk, "full": ref,
                                 "legacy_chunk": baseline, "mixed_chunk": actual})
    finally:
        engine.exit()
    torch.save(evidence, out / "logits.pt")
    kinds = sorted({r["comparison"] for r in rows})
    return {"status": "MEASURED_NEEDS_REVIEW", "v1_acceptance_passed": False,
            "note": "No automatic threshold fitting or exemption. Exit 0 means measurement completed, NOT acceptance passed.",
            "summaries": {kind: summarize([r for r in rows if r["comparison"] == kind]) for kind in kinds},
            "comparisons": rows}, 0


def quality(args, out):
    from benchmarks.v1_validation.corpus import load_corpus
    corpus, provenance = load_corpus(args.corpus)
    if len(corpus) < args.cases:
        raise ValueError("corpus has fewer rows than --cases")
    engine = make_engine(args)
    records, teacher, evidence = [], [], []
    try:
        for item in corpus[:args.cases]:
            tokens = engine.tokenizer.encode(item["text"])
            if len(tokens) < 32 or len(tokens) + args.output_tokens > engine.config.max_model_len:
                raise ValueError(f"corpus {item['id']} must have >=32 tokens and fit max_model_len")
            start = max(16, len(tokens)//2)
            positions = torch.linspace(start, len(tokens)-1, steps=min(args.teacher_samples, len(tokens)-start)).long().unique().tolist()
            records.append({"id": item["id"], "prompt_token_ids": tokens[:start],
                            "text_token_ids": tokens, "teacher_positions": positions})
        save(out / "inputs.json", {"provenance": provenance, "records": records})
        prompts = [r["prompt_token_ids"] for r in records]
        baseline_batch = legacy_generate(engine, prompts, args.output_tokens)
        baseline_serial = [legacy_generate(engine, [p], args.output_tokens)[0] for p in prompts]
        actual, execution = mixed_generate(engine, prompts, args.output_tokens)
        generations = []
        for item, a, rb, rs in zip(records, actual, baseline_batch, baseline_serial):
            generations.append({"id": item["id"], "mixed_vs_legacy_batch": sequence_metrics(a, rb),
                "legacy_serial_vs_legacy_batch": sequence_metrics(rs, rb),
                "mixed_token_ids": a, "legacy_batch_token_ids": rb, "legacy_serial_token_ids": rs,
                "mixed_text": engine.tokenizer.decode(a), "legacy_batch_text": engine.tokenizer.decode(rb),
                "legacy_serial_text": engine.tokenizer.decode(rs)})
        save(out / "generations.json", generations)
        for item in records:
            for position in item["teacher_positions"]:
                history = item["text_token_ids"][:position]
                target = int(item["text_token_ids"][position])
                full = legacy_logits(engine, history, len(history))
                baseline = legacy_logits(engine, history, args.chunk)
                a = mixed_logits(engine, history, args.chunk)
                for name, x, y in (("mixed_vs_legacy_full", a, full),
                                   ("legacy_chunk_vs_legacy_full", baseline, full),
                                   ("mixed_vs_legacy_chunk", a, baseline)):
                    teacher.append({"id": item["id"], "position": position, "comparison": name,
                                    **distribution_metrics(x, y, target=target)})
                evidence.append({"id": item["id"], "position": position, "target": target,
                                 "full": full, "legacy_chunk": baseline, "mixed_chunk": a})
    finally:
        engine.exit()
    torch.save(evidence, out / "logits.pt")
    kinds = sorted({r["comparison"] for r in teacher})
    return {"status": "MEASURED_NEEDS_REVIEW", "v1_acceptance_passed": False,
            "fixed_output_tokens_ignore_eos": args.output_tokens, "requests": len(records),
            "mixed_vs_legacy_exact_sequence_rate": sum(r["mixed_vs_legacy_batch"]["exact_match"] for r in generations)/len(generations),
            "legacy_serial_vs_batch_exact_sequence_rate": sum(r["legacy_serial_vs_legacy_batch"]["exact_match"] for r in generations)/len(generations),
            "teacher_summaries": {kind: summarize([r for r in teacher if r["comparison"] == kind]) for kind in kinds},
            "teacher_comparisons": teacher, "execution": execution, "corpus": provenance,
            "note": "Fixed-history comparisons use source-text targets, never compare logits after generation branches diverge. Sampled conditional PPL is NOT full-corpus PPL. Curated fixtures do NOT establish real-world task quality. Exit 0 only means measurement completed."}, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["semantic", "calibrate", "quality"])
    add_model_arguments(parser)
    parser.add_argument("--out")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--memory", type=float, default=.5)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--chunks", type=int, nargs="+", default=[128, 249, 256, 512])
    parser.add_argument("--lengths", type=int, nargs="+", default=[255, 257, 513])
    parser.add_argument("--seeds", type=int, nargs="+", default=[41])
    parser.add_argument("--artifact", help="Optional old candidate directory, read INPUTS only")
    parser.add_argument("--seq-id", type=int, default=7)
    parser.add_argument("--corpus", help="JSONL rows: {id: string, text: continuous human-written text}")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--teacher-samples", type=int, default=2)
    parser.add_argument("--fp32-relative-limit", type=float, default=.001)
    parser.add_argument("--fp32-max-abs", type=float, default=.02)
    args = parser.parse_args()
    resolve_model_arguments(args, parser)
    if not 0 < args.memory < 1 or args.token_budget < 256:
        parser.error("memory must be in (0,1), token-budget>=256")
    if min(args.chunk, args.cases, args.output_tokens, args.teacher_samples, *args.chunks, *args.lengths) < 1:
        parser.error("lengths/counts/chunks must be positive")
    if max(args.lengths) > 1536 or args.output_tokens > 1024:
        parser.error("diagnostic limits: lengths<=1536, output-tokens<=1024")
    if args.command != "quality" and (args.tp != 1 or args.graph):
        parser.error("semantic/calibrate are isolated Eager TP=1 diagnostics")
    if args.command == "semantic" and (args.fp32_relative_limit != .001 or args.fp32_max_abs != .02):
        parser.error("FP32 diagnostic gates are frozen; do not tune them to pass")
    if args.artifact and args.command == "quality":
        parser.error("quality uses --corpus, not --artifact")
    raw = ROOT / "results/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if args.out:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix=f"v1-{args.command}.", dir=raw))
    print(f"V1 validation artifacts: {out}", flush=True)
    from benchmarks.v0.environment import collect_environment, source_hashes
    before = validation_hashes()
    environment = collect_environment(ROOT, args.model, args.model_revision)
    environment["validation_source_sha256"] = before
    environment["precision_controls"] = {"bf16_reduced_precision_reduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                                          "allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    save(out / "config.json", vars(args))
    save(out / "environment.json", environment)
    try:
        torch.manual_seed(args.seeds[0])
        report, code = {"semantic": semantics, "calibrate": calibration, "quality": quality}[args.command](args, out)
        if validation_hashes() != before or source_hashes(ROOT) != environment["source_sha256"]:
            raise RuntimeError("source changed during validation")
        report["command"] = args.command
        save(out / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"comparisons", "teacher_comparisons"}},
                         indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        save(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save(out / "artifact_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
