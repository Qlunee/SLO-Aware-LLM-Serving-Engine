"""Prepare a sealed low-KV online plan from a matching controlled profile.

This CPU-only step creates fresh open-loop traces. Profiling, fitting, online
execution and policy comparison remain separate, append-only artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from benchmarks.v0.environment import model_metadata_hashes
from benchmarks.v2_baselines.workload import Workload, build_trace
from benchmarks.v3_profile.protocol import load_profile
from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources, verify_artifacts


POLICY_SOURCES = frozenset((
    "nanovllm/engine/step_predictor.py",
    "nanovllm/engine/slack_scheduler.py",
))
MODES = ("poisson", "burst", "long_short")


def prepare(profile_path, predictor_path, out, *, seed_base=1910,
            rps=3.0, output_tokens=64, long_fraction=.4,
            ttft_slo_ms=1500.0, itl_slo_ms=180.0, drain_s=120.0):
    profile_path = Path(profile_path).resolve()
    predictor_path = Path(predictor_path).resolve()
    out = Path(out).resolve()
    protocol, _ = load_profile(profile_path)
    if protocol["split"] != "train" or protocol["smoke_only"]:
        raise ValueError("a complete non-smoke train profile is required")
    verify_artifacts(predictor_path.parent)
    table = read(predictor_path)
    identity = protocol["identity"]
    if table["identity"] != identity:
        raise ValueError("predictor and low-KV profile identities differ")
    if not any(p["seed"] == protocol["seed"] and
               p["protocol_sha256"] == sha(profile_path / "protocol.json") and
               p["steps_sha256"] == sha(profile_path / "steps.jsonl")
               for p in table["training_profiles"]):
        raise ValueError("the predictor was not trained on this profile")
    current = sources()
    changed = [name for name, digest in identity["source_sha256"].items()
               if name.startswith("nanovllm/") and name not in POLICY_SOURCES
               and current.get(name) != digest]
    if changed:
        raise ValueError(f"profiled execution source changed: {changed}")
    model_path = Path(protocol["model_path"])
    if model_metadata_hashes(model_path) != identity["model_metadata_sha256"]:
        raise ValueError("model metadata changed since profile collection")
    if type(seed_base) is not int or seed_base < 0 or type(output_tokens) is not int:
        raise ValueError("invalid trace seed or output length")
    if (isinstance(drain_s, bool) or not isinstance(drain_s, (int, float))
            or not math.isfinite(drain_s) or drain_s <= 0):
        raise ValueError("drain_s must be positive and finite")
    workload = Workload(warmup_s=2, measurement_s=8, rps=rps,
                        output_tokens=output_tokens, long_fraction=long_fraction,
                        ttft_slo_ms=ttft_slo_ms, itl_slo_ms=itl_slo_ms)
    workload.validate()
    settings = protocol["settings"]
    if workload.long_tokens + workload.output_tokens > settings["max_model_len"]:
        raise ValueError("request exceeds profiled model context")
    if workload.long_tokens + workload.output_tokens > settings["kv_block_cap"] * 256:
        raise ValueError("one request exceeds KV cap")
    seeds = {mode: seed_base + i + 1 for i, mode in enumerate(MODES)}
    if len(set(seeds.values())) != len(seeds) or protocol["seed"] in seeds.values():
        raise ValueError("online and training seeds overlap")
    vocab = read(model_path / "config.json")["vocab_size"]
    if type(vocab) is not int or vocab <= 3:
        raise ValueError("invalid model vocabulary")
    out.mkdir(parents=True, exist_ok=False)
    (out / "traces").mkdir()
    counts = {}
    for mode, seed in seeds.items():
        rng = random.Random(seed + 900_000)
        trace = build_trace(workload, mode, seed,
            lambda length, index: [rng.randrange(3, vocab) for _ in range(length)],
            identity["model_revision"])
        if not any(2_000_000_000 <= r["arrival_ns"] < 10_000_000_000
                   for r in trace["requests"]):
            raise ValueError(f"empty measured cohort: {mode}")
        save(out / "traces" / f"{mode}.json", trace)
        counts[mode] = len(trace["requests"])
    save(out / "plan.json", {
        "schema": "v3-refine-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "settings": settings,
        "model_path": str(model_path),
        "execution_sources": current,
        "training_profile": str(profile_path),
        "training_protocol_sha256": sha(profile_path / "protocol.json"),
        "predictor_sha256": sha(predictor_path),
        "online_seeds": seeds,
        "drain_s": drain_s,
        "scope": "fresh low-KV-capacity diagnostic traces; independent online comparison",
    })
    seal(out)
    verify_artifacts(out)
    return {"plan": str(out), "kv_block_cap": settings["kv_block_cap"],
            "online_seeds": seeds, "request_counts": counts,
            "predictor": str(predictor_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-base", type=int, default=1910)
    parser.add_argument("--rps", type=float, default=3.0)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--long-fraction", type=float, default=.4)
    parser.add_argument("--ttft-slo-ms", type=float, default=1500.0)
    parser.add_argument("--itl-slo-ms", type=float, default=180.0)
    parser.add_argument("--drain-s", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.profile, args.predictor, args.out,
        seed_base=args.seed_base, rps=args.rps,
        output_tokens=args.output_tokens, long_fraction=args.long_fraction,
        ttft_slo_ms=args.ttft_slo_ms, itl_slo_ms=args.itl_slo_ms,
        drain_s=args.drain_s), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
