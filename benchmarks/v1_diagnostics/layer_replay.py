"""Replay a saved quality case; audit logical KV writes and per-layer attention.

TP=1/Eager only. All hooks are observational and restored on exit. The shadow
uses projected K/V indexed by logical positions, NOT the production page table.
CPU FP32 dense attention avoids changing the GPU's TF32/reduction settings.
No result here overrides the original V1 acceptance checker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_error(actual, reference):
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    if a.shape != b.shape or not a.numel():
        raise ValueError("empty tensor or shape mismatch")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite tensor")
    return {"exact": torch.equal(a, b), "max_abs": (a-b).abs().max().item(),
            "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item()}


def expected_slots(table, start, end, block_size):
    if not 0 <= start < end or block_size <= 0 or len(table)*block_size < end:
        raise ValueError("invalid logical span/table")
    if any(p < 0 for p in table):
        raise ValueError("negative page")
    return [table[p//block_size]*block_size + p % block_size for p in range(start, end)]


def gather_pages(cache, table, length):
    if cache.ndim != 4 or length <= 0:
        raise ValueError("expected paged [pages, block, heads, dim] cache")
    pages = (length + cache.shape[1]-1)//cache.shape[1]
    if len(table) < pages or any(not 0 <= p < cache.shape[0] for p in table[:pages]):
        raise ValueError("missing/out-of-range physical page")
    ids = torch.tensor(table[:pages], dtype=torch.long, device=cache.device)
    return cache.index_select(0, ids).flatten(0, 1)[:length].detach().cpu().clone()


class LogicalKV:
    """Independent contiguous CPU ledger; unobserved prefix entries are errors."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.valid = torch.zeros(capacity, dtype=torch.bool)
        self.k = self.v = None

    def write(self, start, k, v):
        if k.ndim != 3 or v.shape != k.shape or not 0 <= start < start+len(k) <= self.capacity:
            raise ValueError("invalid shadow write")
        if self.k is None:
            self.k = torch.empty((self.capacity, *k.shape[1:]), dtype=k.dtype)
            self.v = torch.empty_like(self.k)
        if k.shape[1:] != self.k.shape[1:] or k.dtype != self.k.dtype:
            raise ValueError("shadow geometry changed")
        end = start + len(k)
        self.k[start:end].copy_(k.detach().cpu())
        self.v[start:end].copy_(v.detach().cpu())
        self.valid[start:end] = True

    def read(self, length):
        if not 0 < length <= self.capacity or not self.valid[:length].all():
            raise ValueError("unobserved logical KV: prefix hit or skipped computation")
        return self.k[:length], self.v[:length]


class Recorder:
    def __init__(self, engine, target_prompt, output_index, out):
        self.engine, self.prompt, self.index, self.out = engine, target_prompt, output_index, out
        self.capacity = len(target_prompt) + output_index
        self.stage, self.active = None, None
        self.paths = {}
        self.pending = {}
        self.layers = [layer.self_attn.attn for layer in engine.model_runner.model.model.layers]
        self.handles = [engine.model_runner.model.register_forward_pre_hook(self.model_pre)]
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda module, args, i=i: self.attention_pre(i, module, args)))
            self.handles.append(layer.register_forward_hook(
                lambda module, args, output, i=i: self.attention_post(i, module, output)))
        self.handles.append(engine.model_runner.model.lm_head.register_forward_hook(self.head))

    def start(self, stage):
        self.stage = stage
        self.shadows = [LogicalKV(self.capacity) for _ in self.layers]
        self.paths[stage] = {"state_checks": [], "kv_checks": [], "layers": [], "snapshots": {}}
        (self.out / stage).mkdir()

    def model_pre(self, module, args):
        if self.active is None:
            return
        s = self.active
        ids, positions = args
        a, b = s["a"], s["b"]
        check = {"output_tokens_before_step": s["output_count"], "phase": s["phase"],
                 "start": s["start"], "end": s["end"],
                 "history_length": len(s["history"]),
                 "input_ids_match": ids[a:b].tolist() == s["history"][s["start"]:s["end"]],
                 "positions_match": positions[a:b].tolist() == list(range(s["start"], s["end"])),
                 "computed_matches_start": s["computed"] == s["start"]}
        if s["plan_ok"] is not None:
            check["mixed_plan_matches_sequence"] = s["plan_ok"]
        self.paths[self.stage]["state_checks"].append(check)

    def attention_pre(self, i, module, args):
        if self.active is None:
            return
        from nanovllm.utils.context import get_context
        s, c = self.active, get_context()
        a, b = s["a"], s["b"]
        q, k, v = args
        self.shadows[i].write(s["start"], k[a:b], v[a:b])
        if i == 0:
            check = self.paths[self.stage]["state_checks"][-1]
            slots = expected_slots(s["table"], s["start"], s["end"], module.k_cache.shape[1])
            check["slots_match"] = c.slot_mapping[a:b].tolist() == slots
            table = None
            if c.mixed is not None:
                m = c.mixed
                if s["phase"] == "decode":
                    length, table = int(m.decode_lens[a]), m.decode_tables[a]
                else:
                    j = s["prefill_index"]
                    length = int(m.prefill_cu_k[j+1]-m.prefill_cu_k[j])
                    check["prefill_query_offsets_match"] = (
                        int(m.prefill_cu_q[j]) == a-m.num_decode
                        and int(m.prefill_cu_q[j+1]) == b-m.num_decode)
                    if m.prefill_has_history:
                        table = m.prefill_tables[j]
            elif c.is_prefill:
                j = s["row_index"]
                length = int(c.cu_seqlens_k[j+1]-c.cu_seqlens_k[j])
                if c.block_tables is not None:
                    table = c.block_tables[j]
            else:
                length, table = int(c.context_lens[a]), c.block_tables[a]
            pages = (s["end"]+module.k_cache.shape[1]-1)//module.k_cache.shape[1]
            check["context_length_matches"] = length == s["end"]
            check["attention_table_matches"] = (
                table[:pages].tolist() == s["table"][:pages] if table is not None else s["start"] == 0)
        if s["final"]:
            self.pending[i] = {"q": q[a:b].detach().cpu().clone(),
                               "projected_k": k[a:b].detach().cpu().clone(),
                               "projected_v": v[a:b].detach().cpu().clone()}

    def attention_post(self, i, module, output):
        if self.active is None:
            return
        from benchmarks.v1_validation.reference import dense_suffix_attention
        s = self.active
        sk, sv = self.shadows[i].read(s["end"])
        ck = gather_pages(module.k_cache, s["table"], s["end"])
        cv = gather_pages(module.v_cache, s["table"], s["end"])
        check = {"layer": i, "output_tokens_before_step": s["output_count"],
                 "start": s["start"], "end": s["end"],
                 "k": tensor_error(ck, sk), "v": tensor_error(cv, sv)}
        self.paths[self.stage]["kv_checks"].append(check)
        if not s["final"]:
            return
        if i in self.paths[self.stage]["snapshots"]:
            raise RuntimeError("target output step was captured twice")
        evidence = self.pending.pop(i)
        # Original Decode returns [B,1,H,D]; mixed returns [T,H,D].
        actual = output.reshape(-1, module.num_heads, module.head_dim)[s["a"]:s["b"]].detach().cpu().clone()
        with torch.inference_mode():
            reference = dense_suffix_attention(evidence["q"], sk, sv, module.scale)
        error = tensor_error(actual, reference)
        # Reuse existing tests/test_v1_reference_gpu.py probe tolerances;
        # this diagnostic never changes production numerical acceptance.
        error["matches_existing_attention_probe_tolerance"] = bool(
            torch.isclose(actual.float(), reference.float(), atol=.02, rtol=.02).all())
        evidence.update(cache_k=ck, cache_v=cv, attention_output=actual,
                        reference_output=reference, history=s["history"])
        torch.save(evidence, self.out / self.stage / f"layer_{i:02d}.pt")
        self.paths[self.stage]["layers"].append({"layer": i, "attention_vs_shadow_reference": error})
        self.paths[self.stage]["snapshots"][i] = evidence

    def head(self, module, args, output):
        s = self.active
        if s is not None and s["final"] and output is not None:
            path = self.paths[self.stage]
            if "logits" in path:
                raise RuntimeError("target LMHead captured twice")
            path["logits"] = output[s["sample_index"]].detach().float().cpu().clone()
            path["history"] = s["history"]

    def spec(self, seq, a, b, start, end, phase, row_index, sample_index, plan_ok=None, prefill_index=None):
        return {"a": a, "b": b, "start": start, "end": end, "phase": phase,
                "row_index": row_index, "prefill_index": prefill_index,
                "sample_index": sample_index, "plan_ok": plan_ok,
                "table": list(seq.block_table), "computed": seq.num_cached_tokens,
                "output_count": seq.num_completion_tokens, "history": list(seq.token_ids),
                "final": seq.num_completion_tokens == self.index and end == len(seq)}

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    from benchmarks.v0.environment import collect_environment, source_hashes
    from benchmarks.v1_validation.run import make_engine, legacy_generate, mixed_generate, validation_hashes
    from benchmarks.v1_validation.metrics import distribution_metrics, sequence_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/raw/v1-quality.jp5c_dbl")
    parser.add_argument("--target-id", default="curated-6")
    parser.add_argument("--output-token", type=int, default=11, help="1-based")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU oracle only; not a performance benchmark")
    cli = parser.parse_args()
    root, source = Path.cwd(), Path(cli.source).resolve()
    for name, digest in json.loads((source / "artifact_hashes.json").read_text()).items():
        if Path(name).name != name or hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"invalid source artifact: {name}")
    config = json.loads((source / "config.json").read_text())
    if config["tp"] != 1 or config["graph"] or not 1 <= cli.output_token <= config["output_tokens"] or cli.cpu_threads < 1:
        parser.error("requires a TP=1 Eager source and a valid output token")
    records = json.loads((source / "inputs.json").read_text())["records"]
    target = next(i for i, r in enumerate(records) if r["id"] == cli.target_id)
    prompts = [r["prompt_token_ids"] for r in records]
    old = json.loads((source / "generations.json").read_text())[target]
    old_env = json.loads((source / "environment.json").read_text())
    before, support_before = source_hashes(root), validation_hashes()
    if before != old_env["source_sha256"] or support_before != old_env["validation_source_sha256"]:
        raise RuntimeError("source no longer matches quality run; do not mix source versions")
    args = SimpleNamespace(**config)
    torch.set_num_threads(cli.cpu_threads)
    out = Path(tempfile.mkdtemp(prefix="v1-layer-replay.", dir=root / "results/raw"))
    print(f"V1 layer replay artifacts: {out}", flush=True)
    environment = collect_environment(root, args.model, args.model_revision)
    environment["validation_source_sha256"] = support_before
    diagnostic_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment["diagnostic_sha256"] = diagnostic_before
    environment["cpu_oracle_threads"] = cli.cpu_threads
    save(out / "environment.json", environment)
    save(out / "config.json", {**vars(cli), "source": str(source), "engine_config": config})
    engine = rec = None
    try:
        torch.manual_seed(args.seeds[0])
        engine = make_engine(args)
        runner = engine.model_runner
        old_run, old_mixed = runner.run, runner.run_mixed
        rec = Recorder(engine, prompts[target], cli.output_token-1, out)

        def observed_run(seqs, is_prefill):
            offset = 0
            for j, seq in enumerate(seqs):
                count = seq.num_scheduled_tokens if is_prefill else 1
                if seq.prompt_token_ids == prompts[target] and seq.num_completion_tokens < cli.output_token:
                    start = seq.num_cached_tokens if is_prefill else len(seq)-1
                    rec.active = rec.spec(seq, offset, offset+count, start, start+count,
                        "prefill" if is_prefill else "decode", j, j)
                offset += count
            try:
                return old_run(seqs, is_prefill)
            finally:
                rec.active = None

        def observed_mixed(batch, return_logits=False, verify_tp_plan=False, greedy=True):
            for j, row in enumerate(batch.requests):
                seq = engine.scheduler.sequences[row.seq_id]
                if seq.prompt_token_ids == prompts[target] and seq.num_completion_tokens < cli.output_token:
                    a, b = batch.query_offsets[j:j+2]
                    sample = batch.sample_seq_ids.index(row.seq_id) if row.seq_id in batch.sample_seq_ids else None
                    plan_ok = (row.start_pos == seq.num_cached_tokens and row.logical_tokens == len(seq)
                               and list(batch.block_tables[j]) == list(seq.block_table))
                    rec.active = rec.spec(seq, a, b, row.start_pos, row.end_pos, row.phase,
                        j, sample, plan_ok, j-batch.num_decode_tokens)
            try:
                return old_mixed(batch, return_logits, verify_tp_plan, greedy)
            finally:
                rec.active = None

        runner.run, runner.run_mixed = observed_run, observed_mixed
        try:
            rec.start("legacy_batch")
            batch_outputs = legacy_generate(engine, prompts, args.output_tokens)
            rec.start("legacy_serial")
            serial_outputs = [legacy_generate(engine, [p], args.output_tokens)[0] for p in prompts]
            rec.start("mixed")
            mixed_outputs, execution = mixed_generate(engine, prompts, args.output_tokens)
        finally:
            runner.run, runner.run_mixed = old_run, old_mixed
            rec.close()
        report = {"status": "MEASURED_NEEDS_REVIEW", "v1_acceptance_passed": False,
                  "scope": f"{cli.target_id}, output token {cli.output_token}, TP1 Eager only",
                  "reproduced_saved_outputs": {}, "paths": {}, "cross_path_layers": {},
                  "execution": execution,
                  "sequence_comparisons": {"mixed_vs_batch": sequence_metrics(mixed_outputs[target], batch_outputs[target]),
                      "serial_vs_batch": sequence_metrics(serial_outputs[target], batch_outputs[target])},
                  "logit_comparisons": {}}
        for name, outputs, key in (("legacy_batch", batch_outputs, "legacy_batch_token_ids"),
                                  ("legacy_serial", serial_outputs, "legacy_serial_token_ids"),
                                  ("mixed", mixed_outputs, "mixed_token_ids")):
            report["reproduced_saved_outputs"][name] = outputs[target] == old[key]
            path = rec.paths[name]
            if len(path["snapshots"]) != len(rec.layers) or "logits" not in path:
                raise RuntimeError(f"incomplete target capture for {name}")
            torch.save({"history": path["history"], "logits": path["logits"]}, out/name/"logits.pt")
            report["paths"][name] = {k: v for k, v in path.items() if k not in {"snapshots", "logits", "history"}}
            values, ids = path["logits"].topk(2)
            report["paths"][name]["top2"] = {"token_ids": ids.tolist(), "logits": values.tolist(),
                "argmax_token_id": int(path["logits"].argmax()), "margin": float(values[0]-values[1])}
            report["paths"][name]["all_state_checks_match"] = all(
                value for row in path["state_checks"] for value in row.values() if isinstance(value, bool))
            report["paths"][name]["all_shadow_kv_exact"] = all(
                row[which]["exact"] for row in path["kv_checks"] for which in ("k", "v"))
            report["paths"][name]["all_attention_within_existing_probe_tolerance"] = all(
                row["attention_vs_shadow_reference"]["matches_existing_attention_probe_tolerance"] for row in path["layers"])
        base = rec.paths["legacy_batch"]
        for name in ("legacy_serial", "mixed"):
            path = rec.paths[name]
            same_history = path["history"] == base["history"]
            report["logit_comparisons"][name] = {"same_history": same_history,
                **(distribution_metrics(path["logits"], base["logits"]) if same_history else {})}
            report["cross_path_layers"][name] = []
            for i in range(len(rec.layers)):
                a, b = path["snapshots"][i], base["snapshots"][i]
                report["cross_path_layers"][name].append({"layer": i, "same_history": a["history"] == b["history"],
                    **({key: tensor_error(a[key], b[key]) for key in
                       ("q", "projected_k", "projected_v", "cache_k", "cache_v", "attention_output")}
                       if a["history"] == b["history"] else {})})
        report["source_unchanged"] = (source_hashes(root) == before and validation_hashes() == support_before
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == diagnostic_before)
        save(out/"report.json", report)
        compact = {k: v for k, v in report.items() if k not in {"paths", "cross_path_layers"}}
        compact["path_summaries"] = {name: {k: v for k, v in path.items() if k.startswith("all_")}
                                     for name, path in report["paths"].items()}
        print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        if not report["source_unchanged"]:
            raise RuntimeError("source changed during diagnosis; evidence is invalid")
        if not all(report["reproduced_saved_outputs"].values()):
            raise RuntimeError("instrumented replay differs from saved outputs; review instrumentation before attribution")
    except BaseException as exc:
        save(out/"error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if rec is not None:
            rec.close()
        if engine is not None:
            engine.exit()
        save(out/"artifact_hashes.json", {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file() and p != out/"artifact_hashes.json"})


if __name__ == "__main__":
    main()
