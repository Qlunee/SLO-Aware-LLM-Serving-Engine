"""Process-local observational hooks. No serving file is edited."""
from dataclasses import asdict
from pathlib import Path
import traceback

import torch

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha
from benchmarks.v2_baselines.protocol import load, Settings
from .core import validate_plan, tool_hashes
from .tensors import pack_context, snapshot_kv, logit_summary, validate_context


class OnlineCapture:
    def __init__(self, engine, policy, targets, output_token, out, max_bytes):
        self.engine, self.runner, self.policy = engine, engine.model_runner, policy
        self.targets, self.output_token, self.out = set(targets), output_token, out
        self.max_bytes, self.active, self.pending = max_bytes, None, None
        self.entries, self.seen = [], set()
        self.method = "run" if policy == "baseline" else "run_mixed"
        self.original = getattr(self.runner, self.method)
        self.handles = []

    def install(self):
        self.handles = [self.runner.model.register_forward_pre_hook(self.before_model),
                        self.runner.model.lm_head.register_forward_hook(self.after_head)]
        setattr(self.runner, self.method, self.observed_call)

    def close(self):
        setattr(self.runner, self.method, self.original)
        for handle in self.handles:
            handle.remove()

    def observed_call(self, first, *args, **kwargs):
        legacy = self.policy == "baseline"
        seqs = first if legacy else [self.engine.scheduler.sequences[r.seq_id] for r in first.requests]
        sample_ids = [s.seq_id for s in seqs] if legacy else list(first.sample_seq_ids)
        selected = [s.seq_id for s in seqs if s.seq_id in sample_ids
                    and self.engine.by_seq[s.seq_id].request_id in self.targets
                    and s.num_completion_tokens == self.output_token-1]
        if not selected:
            return self.original(first, *args, **kwargs)
        rows, offset = [], 0
        for i, seq in enumerate(seqs):
            if legacy:
                prefill = args[0] if args else kwargs["is_prefill"]
                start = seq.num_cached_tokens if prefill else len(seq)-1
                end = start+seq.num_scheduled_tokens if prefill else len(seq)
                phase = ("prefill" if seq.num_completion_tokens == 0 else "recompute") if prefill else "decode"
            else:
                row = first.requests[i]
                start, end, phase = row.start_pos, row.end_pos, row.phase
            rows.append({"request_id": self.engine.by_seq[seq.seq_id].request_id,
                         "seq_id": seq.seq_id, "phase": phase, "start": start, "end": end,
                         "a": offset, "b": offset+end-start, "table": list(seq.block_table),
                         "history": list(seq.token_ids), "output_count": seq.num_completion_tokens,
                         "committed_computed_tokens": seq.num_cached_tokens})
            offset += end-start
        self.active = {"rows": rows, "sample_seq_ids": sample_ids,
                       "sample_indices": [r["b"]-1 for r in rows if r["seq_id"] in sample_ids],
                       "selected_seq_ids": selected, "block_size": self.runner.block_size,
                       "total_blocks": self.runner.config.num_kvcache_blocks,
                       "selection": "indexed" if not legacy and not first.pure_decode else "native",
                       "step_id": self.engine.step_count, "policy": self.policy}
        self.pending = None
        try:
            result = self.original(first, *args, **kwargs)
            if self.pending is None or "target_logits" not in self.pending:
                raise RuntimeError("target model/LMHead hooks did not capture exactly one forward")
            sampled = dict(zip(sample_ids, result)) if legacy else result["tokens"]
            snapshot = self.pending
            descriptions = []
            for seq_id in selected:
                row = next(r for r in rows if r["seq_id"] == seq_id)
                request_id = row["request_id"]
                if request_id in self.seen:
                    raise RuntimeError("target output captured more than once")
                stats = logit_summary(snapshot["target_logits"][str(seq_id)])
                if sampled[seq_id] != stats["argmax"]:
                    raise AssertionError("captured LMHead row does not match the actual greedy output")
                descriptions.append({"request_id": request_id, "seq_id": seq_id,
                                     "output_token": self.output_token, "sampled_token": sampled[seq_id],
                                     "history": row["history"], "logits": stats})
                self.seen.add(request_id)
            name = f"snapshot-{len(self.entries):03d}"
            torch.save(snapshot, self.out/f"{name}.pt")
            entry = {"file": f"{name}.pt", "step_id": self.engine.step_count,
                     "plan": snapshot["plan"], "targets": descriptions,
                     "context_shapes": context_shapes(snapshot["context"]),
                     "kv_snapshot_bytes": snapshot["image"]["kv"].numel()*snapshot["image"]["kv"].element_size()}
            save(self.out/f"{name}.json", entry)
            self.entries.append(entry)
            print("Captured", name, [(r["request_id"], r["sampled_token"]) for r in descriptions], flush=True)
            return result
        finally:
            self.active, self.pending = None, None

    def before_model(self, module, args):
        if self.active is None:
            return
        from nanovllm.utils.context import get_context
        if self.pending is not None:
            raise RuntimeError("more than one model forward for captured Step")
        ids, positions = args
        context = get_context()
        plan = {**self.active, "input_ids": ids.tolist(), "positions": positions.tolist(),
                "slots": context.slot_mapping.tolist()}
        validate_plan(plan)
        encoded = pack_context(context)
        validate_context(plan, encoded)
        if any(r["committed_computed_tokens"] != r["start"] for r in plan["rows"]):
            raise AssertionError("logical progress differs from captured query start")
        self.pending = {"schema": "online-token-snapshot-v1", "plan": plan,
                        "input_ids": ids.detach().cpu().clone(),
                        "positions": positions.detach().cpu().clone(),
                        "context": encoded,
                        "image": snapshot_kv(self.runner.kv_cache, plan, self.max_bytes)}

    def after_head(self, module, args, output):
        if self.active is None:
            return
        if self.pending is None or "target_logits" in self.pending or output is None:
            raise RuntimeError("invalid LMHead capture lifecycle")
        plan = self.pending["plan"]
        self.pending["target_logits"] = {
            str(i): output[plan["sample_seq_ids"].index(i)].detach().float().cpu().clone()
            for i in plan["selected_seq_ids"]}


def context_shapes(value):
    return {k: list(v.shape) if isinstance(v, torch.Tensor)
            else context_shapes(v) if isinstance(v, dict) else v
            for k, v in value.items()}


def capture(args):
    from benchmarks.v0.environment import collect_environment
    from benchmarks.v0.driver import execute_trace
    from benchmarks.v2_baselines.run import make_engine
    from benchmarks.v2_baselines.workload import validate_trace
    from benchmarks.v2_stall.run import check_state, warm_engine
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    protocol, before = load(suite), tool_hashes()
    settings = Settings(**protocol["settings"])
    trace = read(suite/f"traces/{args.mode}-seed-{settings.seed}.json")
    specs = validate_trace(trace)
    by_id = {s.request_id: s for s in specs}
    if (not args.target_ids or len(set(args.target_ids)) != len(args.target_ids)
            or any(i not in by_id or by_id[i].max_output_tokens < args.output_token for i in args.target_ids)):
        raise ValueError("targets must be unique Trace requests with enough output tokens")
    if torch.cuda.device_count() != 1:
        raise ValueError("requires exactly one visible GPU, TP1/Eager only")
    out.mkdir(parents=True, exist_ok=False)
    save(out/"identity.json", {"suite": str(suite), "protocol_sha256": sha(suite/"protocol.json"),
        "mode": args.mode, "policy": args.policy, "target_ids": args.target_ids,
        "output_token": args.output_token, "tool_sha256": before,
        "diagnostic_only": True, "v1_acceptance_passed": False})
    save(out/"trace.json", trace)
    engine = recorder = None
    try:
        torch.manual_seed(settings.seed)
        torch.cuda.manual_seed_all(settings.seed)
        save(out/"environment.json", collect_environment(ROOT, protocol["model"]["path"], protocol["model"]["revision"]))
        engine, config = make_engine(protocol, args.policy)
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise ValueError("requires BF16 KV")
        warm_engine(engine, config, trace)
        recorder = OnlineCapture(engine, args.policy, args.target_ids, args.output_token, out,
                                 round(args.max_snapshot_mib*1024**2))
        recorder.install()
        original_step = engine.step
        def checked_step():
            result = original_step()
            check_state(engine)
            return result
        engine.step = checked_step
        result = execute_trace(engine, config, trace, specs, settings.drain_s, "full")
        save(out/"requests.json", [asdict(r) for r in result.records])
        save(out/"steps.json", engine.steps)
        save(out/"backlog.json", result.backlog)
        check_state(engine)
        complete = all(r.status == "completed" for r in result.records)
        load(suite)
        if before != tool_hashes():
            raise RuntimeError("diagnostic tool changed during capture")
        report = {"captures": recorder.entries, "missing_targets": sorted(set(args.target_ids)-recorder.seen),
                  "all_completed": complete, "step_state_checks_passed": True,
                  "kv_blocks": config.num_kvcache_blocks, "driver_error": result.error,
                  "end_ns": result.end_ns, "deadline_ns": result.deadline_ns,
                  "diagnostic_only": True, "v1_acceptance_passed": False,
                  "scope": "Instrumented online run. Snapshot copying changes arrival/batch timing; NOT the historical run or a performance measurement."}
        save(out/"report.json", report)
        if result.error or not complete or report["missing_targets"]:
            raise RuntimeError("capture incomplete; retain all artifacts and inspect report")
    except Exception:
        save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    print("Capture complete:", out, flush=True)
