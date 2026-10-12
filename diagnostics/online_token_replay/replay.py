"""KV-reset, fixed-input eager forwards; no online arrival driver is involved."""
from dataclasses import fields
from pathlib import Path
import traceback

import torch

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, verify_artifacts
from benchmarks.v2_baselines.protocol import load
from .core import validate_plan, valid_prefix_slots, tool_hashes
from .tensors import restore_kv, unpack_context, compare_logits, logit_summary, validate_context


@torch.inference_mode()
def forward_snapshot(runner, snapshot, request_id, isolate=False):
    from nanovllm.utils.context import set_context, reset_context
    plan = snapshot["plan"]
    pages = validate_plan(plan)
    validate_context(plan, snapshot["context"])
    if snapshot["input_ids"].tolist() != plan["input_ids"] or snapshot["positions"].tolist() != plan["positions"]:
        raise ValueError("saved input tensors differ from logical plan")
    image = snapshot["image"]
    if pages != image["block_ids"] or valid_prefix_slots(plan) != image["valid_slots"]:
        raise ValueError("snapshot does not cover the captured logical history")
    row = next(r for r in plan["rows"] if r["request_id"] == request_id)
    if row["seq_id"] not in plan["sample_seq_ids"]:
        raise ValueError("target is not a sampled row")
    restore_kv(runner.kv_cache, image)
    device = runner.kv_cache.device
    ids, positions = snapshot["input_ids"].to(device), snapshot["positions"].to(device)
    try:
        if isolate:
            a, b = row["a"], row["b"]
            ids, positions = ids[a:b].contiguous(), positions[a:b].contiguous()
            slots = torch.tensor(plan["slots"][a:b], dtype=torch.int32, device=device)
            table = torch.tensor([row["table"]], dtype=torch.int32, device=device)
            if row["phase"] == "decode":
                set_context(False, slot_mapping=slots, block_tables=table,
                            context_lens=torch.tensor([row["end"]], dtype=torch.int32, device=device))
            else:
                set_context(True, slot_mapping=slots, block_tables=table if row["start"] else None,
                            cu_seqlens_q=torch.tensor([0, b-a], dtype=torch.int32, device=device),
                            cu_seqlens_k=torch.tensor([0, row["end"]], dtype=torch.int32, device=device),
                            max_seqlen_q=b-a, max_seqlen_k=row["end"])
            hidden = runner.model(ids, positions)
            logits = runner.model.compute_logits(hidden)
            return logits[0].detach().float().cpu().clone()
        context = unpack_context(snapshot["context"], device)
        set_context(**{f.name: getattr(context, f.name) for f in fields(context)})
        hidden = runner.model(ids, positions)
        if plan["selection"] == "indexed":
            indices = torch.tensor(plan["sample_indices"], dtype=torch.int64, device=device)
            hidden = hidden.index_select(0, indices)
        elif plan["selection"] != "native":
            raise ValueError("unknown LMHead selection rule")
        logits = runner.model.compute_logits(hidden)
        if len(logits) != len(plan["sample_seq_ids"]):
            raise ValueError("LMHead replay row count differs")
        return logits[plan["sample_seq_ids"].index(row["seq_id"])].detach().float().cpu().clone()
    finally:
        reset_context()


def replay(args):
    from benchmarks.v2_baselines.run import make_engine
    from benchmarks.v0.environment import collect_environment
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    verify_artifacts(source)
    identity, capture_report = read(source/"identity.json"), read(source/"report.json")
    if (source/"error.json").exists() or capture_report["missing_targets"] or not capture_report["all_completed"]:
        raise ValueError("source capture is incomplete/failed")
    before = tool_hashes()
    if before != identity["tool_sha256"]:
        raise ValueError("tool changed since capture; do not mix versions")
    suite = Path(identity["suite"])
    protocol = load(suite)
    if sha(suite/"protocol.json") != identity["protocol_sha256"]:
        raise ValueError("capture protocol changed")
    if torch.cuda.device_count() != 1:
        raise ValueError("requires exactly one visible GPU")
    out.mkdir(parents=True, exist_ok=False)
    save(out/"identity.json", {"source": str(source), "source_manifest_sha256": sha(source/"artifact_hashes.json"),
        "protocol_sha256": identity["protocol_sha256"], "tool_sha256": before,
        "repeats": args.repeats, "diagnostic_only": True, "v1_acceptance_passed": False})
    engine = None
    try:
        env = collect_environment(ROOT, protocol["model"]["path"], protocol["model"]["revision"])
        previous = read(source/"environment.json")
        for key in ("packages", "gpus", "model_revision_declared", "torch_cuda_version", "model_metadata_sha256"):
            if env.get(key) != previous.get(key):
                raise ValueError(f"replay environment differs: {key}")
        save(out/"environment.json", env)
        torch.manual_seed(protocol["settings"]["seed"])
        torch.cuda.manual_seed_all(protocol["settings"]["seed"])
        engine, config = make_engine(protocol, identity["policy"])
        entries = []
        for entry in capture_report["captures"]:
            # Only locally generated, manifest-verified tensor/primitive snapshots.
            snapshot = torch.load(source/entry["file"], map_location="cpu", weights_only=True)
            if snapshot["schema"] != "online-token-snapshot-v1" or snapshot["plan"] != entry["plan"]:
                raise ValueError("snapshot metadata differs from capture index")
            if snapshot["image"]["pool_shape"] != list(engine.model_runner.kv_cache.shape):
                raise ValueError("actual replay KV capacity differs; retain failure, do not remap silently")
            for target in entry["targets"]:
                request_id, seq_id = target["request_id"], target["seq_id"]
                captured = snapshot["target_logits"][str(seq_id)]
                vectors = {"captured": captured}
                full, isolated = [], []
                for repeat in range(args.repeats):
                    # Every forward restores the pre-Step KV snapshot independently.
                    a = forward_snapshot(engine.model_runner, snapshot, request_id)
                    b = forward_snapshot(engine.model_runner, snapshot, request_id, isolate=True)
                    vectors[f"full_{repeat}"] = a
                    vectors[f"isolated_{repeat}"] = b
                    full.append(a)
                    isolated.append(b)
                key = f"{Path(entry['file']).stem}-{request_id}"
                torch.save(vectors, out/f"{key}.pt")
                row = {"request_id": request_id, "snapshot": entry["file"], "history": target["history"],
                       "captured": logit_summary(captured), "full_batch_tokens": len(snapshot["input_ids"]),
                       "isolated_query_tokens": next(r["b"]-r["a"] for r in entry["plan"]["rows"] if r["seq_id"] == seq_id),
                       "full_vs_captured": [compare_logits(x, captured) for x in full],
                       "full_repeats_vs_first": [compare_logits(x, full[0]) for x in full[1:]],
                       "isolated_vs_full": [compare_logits(x, full[0]) for x in isolated],
                       "isolated_repeats_vs_first": [compare_logits(x, isolated[0]) for x in isolated[1:]],
                       "vectors_file": f"{key}.pt", "acceptance_override": False}
                entries.append(row)
                print(request_id, "captured/full/isolated top1:", row["captured"]["argmax"],
                      [logit_summary(x)["argmax"] for x in full],
                      [logit_summary(x)["argmax"] for x in isolated], flush=True)
        load(suite)
        if before != tool_hashes():
            raise RuntimeError("tool changed during replay")
        save(out/"report.json", {"entries": entries, "diagnostic_only": True,
            "v1_acceptance_passed": False, "status": "MEASURED_NEEDS_REVIEW",
            "scope": "Full repeats keep input/context/KV fixed. Isolation preserves target history/KV but changes batch shape and potentially Attention interface. Neither is an independent model correctness oracle or acceptance waiver."})
    except Exception:
        save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)
    print("Replay complete:", out, flush=True)
