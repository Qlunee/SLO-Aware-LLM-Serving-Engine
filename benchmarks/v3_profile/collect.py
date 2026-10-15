"""Controlled real-engine batches; no fake KV and no adaptive budget controller."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import traceback

from benchmarks.v0.environment import collect_environment, command_output
from .protocol import (ROOT, TIMING_CONTRACT,
                       planned_cases, predictor_module, runtime_identity,
                       save, seal, sources)
from .driver import run_case


def collect(args, settings):
    settings.validate()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    engine = None
    measured = recorded = 0
    cases_done = []
    initial_sources = sources()
    try:
        import torch
        from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
        from nanovllm.sampling_params import SamplingParams
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        rng = random.Random(args.seed)
        environment = collect_environment(ROOT, args.model, args.model_revision)
        environment["v3_driver"] = command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        engine = MixedLLMEngine(args.model, chunk_size=256, kv_block_cap=settings.kv_block_cap,
            greedy=True, verify_tp_plan=False, tensor_parallel_size=1, enforce_eager=True,
            max_model_len=settings.max_model_len, max_num_seqs=settings.max_num_seqs,
            max_num_batched_tokens=settings.token_budget, gpu_memory_utilization=settings.memory,
            kvcache_block_size=256)
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 KV required")
        if engine.config.num_kvcache_blocks != settings.kv_block_cap:
            raise ValueError("requested common KV cap unavailable; reduce cap/matrix prospectively")
        if engine.config.max_model_len != settings.max_model_len:
            raise ValueError("model context is smaller than configured profile context")
        save(out/"environment.json", environment)
        identity = runtime_identity(environment, engine.config, engine.model_runner.kv_cache)
        protocol = {"schema": "v3-step-profile-v1", "split": args.split, "seed": args.seed,
            "smoke_only": args.smoke, "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": asdict(settings), "model_path": args.model, "identity": identity,
            "timing_contract": TIMING_CONTRACT,
            "scope": "TP1/Eager/BF16 controlled synthetic tokens; NOT online SLO or numerical acceptance"}
        save(out/"protocol.json", protocol)
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
            return engine.add_request(tokens, SamplingParams(max_tokens=outputs, ignore_eos=True))

        cases = planned_cases(settings)
        rng.shuffle(cases)
        # Constructor warmup does not cover all mixed shapes. Mark per-case
        # episode warmup separately and exclude it when fitting/evaluating.
        with (out/"steps.jsonl").open("x", encoding="utf-8") as stream:
            def observe(case, episode, local_step, warmup, batch, result, times):
                nonlocal recorded, measured
                features = estimator.batch_features(batch, result["graph_mode"])
                row = {"step_id": recorded, "case_id": case["case_id"],
                    "episode": episode, "episode_step": local_step, "warmup": warmup,
                    "configured_prefill_budget": case["prefill_budget"], "features": features, **times,
                    "preemptions_total": scheduler.preemption_count,
                    "rows": [{"phase": r.phase, "start_pos": r.start_pos,
                              "end_pos": r.end_pos, "should_sample": r.should_sample}
                             for r in batch.requests]}
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                recorded += 1
                measured += not warmup

            for case in cases:
                run_case(engine, case, settings, add, observe)
                cases_done.append(case["case_id"])
                print(f"[{len(cases_done)}/{len(cases)}] {case['case_id']} measured_steps={measured}", flush=True)
        if sources() != initial_sources:
            raise ValueError("source changed during collection")
        save(out/"summary.json", {"valid": True, "recorded_steps": recorded,
            "measured_steps": measured, "completed_cases": cases_done,
            "preemptions": scheduler.preemption_count, "source_unchanged": True,
            "state_and_drain_passed": True, "smoke_only": args.smoke})
    except BaseException:
        save(out/"error.json", {"error": traceback.format_exc(), "recorded_steps": recorded,
                               "completed_cases": cases_done})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    return out
