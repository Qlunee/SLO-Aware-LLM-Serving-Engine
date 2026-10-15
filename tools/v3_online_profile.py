"""Independent open-loop estimator probe; fixed V1 scheduler, no Slack policy."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
import traceback

from benchmarks.v3_profile.protocol import (ROOT, Settings, predictor_module, read, save,
    seal, sha, sources, verify_artifacts, runtime_identity)
from benchmarks.v3_profile.timing import timed_step
from benchmarks.v2_baselines.workload import validate_trace
from tools.v3_refine import load_suite, prediction_summary


class OnlineAdapter:
    """Keep token visibility timestamps separate from diagnostic bookkeeping."""
    def __init__(self, engine, parameter_factory, predictor, warmup_s, measurement_s,
                 clock=perf_counter_ns):
        self.engine, self.scheduler = engine, engine.scheduler
        self.parameter_factory, self.predictor, self.clock = parameter_factory, predictor, clock
        self.warmup_ns = round(warmup_s*1e9)
        self.measurement_end_ns = round((warmup_s+measurement_s)*1e9)
        self.origin_ns = 0
        self.by_seq, self.steps = {}, []

    def start_clock(self):
        self.origin_ns = self.clock()

    def now(self):
        return self.clock()-self.origin_ns

    def is_finished(self):
        return self.engine.is_finished()

    def submit(self, spec, record):
        seq_id = self.engine.add_request(spec.prompt_token_ids, self.parameter_factory(spec.max_output_tokens))
        record.seq_id, record.enqueue_ns, record.status = seq_id, self.now(), "queued"
        self.by_seq[seq_id] = record

    def step(self):
        batch, result, times = timed_step(self.engine, self.clock)
        visible = times["visible_ns"]-self.origin_ns
        scheduled = times["scheduled_ns"]-self.origin_ns
        outputs = []
        for row in batch.requests:
            seq = self.scheduler.sequences[row.seq_id]
            record = self.by_seq[row.seq_id]
            if record.first_scheduled_ns is None:
                record.first_scheduled_ns = scheduled
            record.status = "active"
            if row.should_sample:
                record.token_times_ns.append(visible)
                record.output_token_ids.append(seq.last_token)
            if seq.is_finished:
                record.status, record.finish_ns = "completed", visible
                outputs.append((seq.seq_id, list(seq.completion_token_ids)))
        estimator = predictor_module()
        features = estimator.batch_features(batch, result["graph_mode"])
        # Shadow prediction is AFTER visible_ns; no predictor feedback affects
        # the fixed scheduler. Future candidate construction cost needs profiling.
        started = self.clock()
        prediction = self.predictor.predict(features)
        cost_us = (self.clock()-started)/1000
        relative_start = times["start_ns"]-self.origin_ns
        window = ("warmup" if relative_start < self.warmup_ns else
                  "drain" if relative_start >= self.measurement_end_ns else "window")
        self.steps.append({"step_id": len(self.steps), "window": window, "features": features,
            "prediction": asdict(prediction), "predict_cpu_us": cost_us,
            "actual_ms": times["step_time_ms"], "relative_start_ns": relative_start, **times})
        return outputs, {"decode_tokens": batch.num_decode_tokens, "prefill_tokens": batch.num_prefill_tokens}

    def snapshot_outputs(self):
        for seq_id, record in self.by_seq.items():
            record.output_token_ids = list(self.scheduler.sequences[seq_id].completion_token_ids)

    def close_waiting_spans(self, timestamp):
        pass


def collect(suite, mode, predictor_path, out):
    suite, predictor_path, out = Path(suite).resolve(), Path(predictor_path).resolve(), Path(out).resolve()
    plan = load_suite(suite)
    trace_path = suite/f"traces/{mode}.json"
    trace = read(trace_path)
    specs = validate_trace(trace)
    verify_artifacts(predictor_path.parent)
    table = read(predictor_path)
    if trace["seed"] in {p["seed"] for p in table["training_profiles"]}:
        raise ValueError("online seed overlaps training")
    out.mkdir(parents=True, exist_ok=False)
    engine = None
    try:
        import torch
        from nanovllm.engine.mixed_llm_engine import MixedLLMEngine
        from nanovllm.sampling_params import SamplingParams
        from benchmarks.v0.driver import execute_trace
        from benchmarks.v0.environment import collect_environment, command_output
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU")
        torch.manual_seed(trace["seed"])
        torch.cuda.manual_seed_all(trace["seed"])
        s = Settings(**plan["settings"])
        environment = collect_environment(ROOT, plan["model_path"], plan["identity"]["model_revision"])
        environment["v3_driver"] = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        engine = MixedLLMEngine(plan["model_path"], chunk_size=plan["online_chunk"],
            kv_block_cap=s.kv_block_cap, greedy=True, verify_tp_plan=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=s.max_model_len,
            max_num_seqs=s.max_num_seqs, max_num_batched_tokens=s.token_budget,
            gpu_memory_utilization=s.memory, kvcache_block_size=256)
        identity = runtime_identity(environment, engine.config, engine.model_runner.kv_cache)
        if identity != plan["identity"]:
            # JSON normalizes tuple-valued NCCL version metadata.
            import json
            if json.loads(json.dumps(identity)) != plan["identity"]:
                raise ValueError("online runtime differs from frozen controlled environment")
        predictor = predictor_module().StepPredictor(table, identity)
        save(out/"environment.json", environment)
        save(out/"trace.json", trace)
        save(out/"run_config.json", {"policy": "fixed-chunk-256", "identity": plan["identity"],
            "workflow_sources": plan["workflow_sources"], "plan_sha256": sha(suite/"plan.json"),
            "timing": "same timed_step helper as controlled profiles; shadow predictions outside interval"})
        params = lambda count: SamplingParams(max_tokens=count, ignore_eos=True)
        # Warm new/history Prefill, Decode and mixed execution without changing
        # trace arrivals or recording initialization as an online sample.
        engine.add_request([3]*257, params(4))
        engine.step()
        engine.step()
        engine.add_request([4]*513, params(2))
        for _ in range(100):
            if engine.is_finished(): break
            engine.step()
        if not engine.is_finished():
            raise RuntimeError("online warmup failed to drain")
        engine.scheduler.assert_consistent()
        engine.scheduler.sequences.clear()
        adapter = OnlineAdapter(engine, params, predictor, trace["warmup_s"], trace["measurement_s"])
        result = execute_trace(adapter, engine.config, trace, specs, plan["drain_s"], "full")
        state_ok = True
        try:
            engine.scheduler.assert_consistent()
            if engine.is_finished() and engine.scheduler.block_manager.used_block_ids:
                raise AssertionError("KV blocks remain after complete drain")
            for record in result.records:
                if len(record.token_times_ns) != len(record.output_token_ids):
                    raise AssertionError("token/time count differs")
                if record.status == "completed" and len(record.output_token_ids) != record.max_output_tokens:
                    raise AssertionError("completed request output length differs")
        except AssertionError:
            state_ok = False
            save(out/"state_error.json", {"message": traceback.format_exc()})
        unchanged = sources() == plan["execution_sources"]
        valid = not result.error and state_ok and unchanged
        statuses = dict(Counter(r.status for r in result.records))
        report = {"schema": "v3-online-prediction-v1", "valid_measurement": valid,
            "all_completed": all(r.status == "completed" for r in result.records),
            "request_statuses": statuses, "preemptions": engine.scheduler.preemption_count,
            "predictor_sha256": sha(predictor_path), "trace_sha256": sha(trace_path),
            "workflow_sources": plan["workflow_sources"], "state_passed": state_ok,
            "source_unchanged": unchanged, "seed": trace["seed"], "mode": mode,
            "end_ns": result.end_ns, "deadline_ns": result.deadline_ns,
            "window": prediction_summary([r for r in adapter.steps if r["window"] == "window"]),
            "drain": prediction_summary([r for r in adapter.steps if r["window"] == "drain"]),
            "scope": "single-seed fixed-policy distribution probe; not SLO optimization evidence"}
        save(out/"report.json", report)
        save(out/"steps.json", adapter.steps)
        save(out/"requests.json", [asdict(r) for r in result.records])
        save(out/"backlog.json", result.backlog)
        if result.error:
            save(out/"error.json", {"message": result.error})
        if not valid:
            raise RuntimeError("invalid online measurement; retained diagnostics")
        print({"mode": mode, "window_coverage": report["window"]["coverage"],
               "window_steps": report["window"]["total"], "all_completed": report["all_completed"]}, flush=True)
    except BaseException:
        if not (out/"error.json").exists():
            save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None: engine.exit()
        finally:
            seal(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--mode", required=True, choices=("poisson", "burst", "long_short"))
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    collect(args.suite, args.mode, args.predictor, args.out)


if __name__ == "__main__":
    main()
