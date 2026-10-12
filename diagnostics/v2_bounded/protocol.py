"""Independent protocol freeze; old source-set manifests remain valid."""
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import random

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources, verify_artifacts
from benchmarks.v2_baselines.protocol import Settings, POLICIES, metadata
from benchmarks.v2_baselines.workload import Workload, build_trace, validate_trace
from .warmup import build_warmup

MODES = ("poisson", "burst", "long_short")
SCHEMA = "v2-bounded-protocol-v1"
WARMUP_RULE = {"request_count": 8, "short_requests": 4, "long_requests": 4,
               "output_tokens": 16, "seed": 12345, "drain_s": 120,
               "all_arrive_at_zero": True, "reset_cold_prefix_metadata": True,
               "scope": "representative shapes, NOT exhaustive batch/Graph coverage"}


def tool_hashes():
    base = Path(__file__).resolve().parent
    paths = {p for p in base.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".md")}
    paths.add(base.parent / "__init__.py")
    return {str(p.relative_to(base.parent)): sha(p) for p in sorted(paths)}


def matrix(rates, seeds, policies):
    if (not rates or len(set(rates)) != len(rates)
            or any(type(r) not in (float, int) or not math.isfinite(r) or r <= 0 for r in rates)):
        raise ValueError("rates must be distinct positive finite numbers")
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("at least three distinct nonnegative seeds required")
    if (len(policies) < 2 or len(set(policies)) != len(policies)
            or "baseline" not in policies or any(p not in POLICIES for p in policies)):
        raise ValueError("distinct supported policies including baseline required")
    jobs = []
    for rate in rates:
        for seed in seeds:
            for index, mode in enumerate(MODES):
                order = list(policies)
                random.Random(f"{seed}:{rate:g}:{index}").shuffle(order)
                for policy in order:
                    jobs.append({"id": f"rps-{rate:g}-seed-{seed}-{mode}-{policy}",
                                 "rps": rate, "seed": seed, "mode": mode, "policy": policy,
                                 "trace": f"traces/rps-{rate:g}-seed-{seed}-{mode}.json"})
    return jobs


def prepare(folder, model, revision, token_factory, rates=(6, 10), seeds=(71, 72, 73),
            policies=("baseline", "chunk-256", "chunk-512"), settings=None,
            workload=None, parent=None, root=ROOT):
    settings, workload = settings or Settings(), workload or Workload(warmup_s=4, measurement_s=60)
    settings.validate(workload)
    jobs = matrix(rates, seeds, policies)
    folder, model = Path(folder).resolve(), Path(model).resolve()
    if not revision.strip() or settings.max_model_len > read(model / "config.json")["max_position_embeddings"]:
        raise ValueError("invalid model revision/context")
    reference = None
    if parent:
        parent = Path(parent).resolve()
        verify_artifacts(parent)
        old = read(parent / "protocol.json")
        if old["source_sha256"] != sources(root) or old["model"] != {"path": str(model), "revision": revision}:
            raise ValueError("parent model or frozen serving sources differ")
        reference = {"path": str(parent), "protocol_sha256": sha(parent / "protocol.json"),
                     "scope": "old evidence reference ONLY; never pooled with this protocol"}
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("prepare requires a new empty directory")
    (folder / "traces").mkdir()
    traces = {}
    for job in jobs:
        name = job["trace"]
        if name in traces:
            continue
        w = Workload(**(asdict(workload) | {"rps": job["rps"]}))
        settings.validate(w)
        trace = build_trace(w, job["mode"], job["seed"],
                            lambda n, i: token_factory(n, job["seed"], i), revision)
        start, end = round(w.warmup_s * 1e9), round((w.warmup_s + w.measurement_s) * 1e9)
        if not any(start <= r["arrival_ns"] < end for r in trace["requests"]):
            raise ValueError("empty measurement cohort")
        save(folder / name, trace)
        traces[name] = sha(folder / name)
    warm = build_warmup(workload.short_tokens, workload.long_tokens, revision,
                        lambda n, i: token_factory(n, WARMUP_RULE["seed"], i))
    save(folder / "warmup.json", warm)
    protocol = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": {"path": str(model), "revision": revision},
                "model_metadata_sha256": metadata(model), "settings": asdict(settings),
                "workload": asdict(workload), "rates": list(rates), "seeds": list(seeds),
                "policies": list(policies), "jobs": jobs, "warmup_rule": WARMUP_RULE,
                "warmup_sha256": sha(folder / "warmup.json"), "trace_sha256": traces,
                "source_sha256": sources(root), "tool_sha256": tool_hashes(), "reference": reference,
                "scope": "TP1/Eager/BF16/full; new bounded-warmup protocol, independent of old 200 runs",
                "rules": {"retry": "no automatic retry or best-attempt selection; failed attempts retained",
                          "capacity": "valid unfinished runs retained; drain completion is NOT stability",
                          "numerics": "all branches retained; no numerical waiver or V1 acceptance",
                          "selection": "report every configured policy; no automatic universal winner"},
                "v1_acceptance_passed": False}
    save(folder / "protocol.json", protocol)
    seal(folder)
    return protocol


def load_suite(folder, root=ROOT):
    folder = Path(folder).resolve()
    verify_artifacts(folder)
    p = read(folder / "protocol.json")
    if p["schema"] != SCHEMA or p["jobs"] != matrix(p["rates"], p["seeds"], p["policies"]):
        raise ValueError("protocol schema/matrix mismatch")
    if p["source_sha256"] != sources(root) or p["tool_sha256"] != tool_hashes():
        raise ValueError("source/tool set changed; freeze a NEW suite")
    if p["model_metadata_sha256"] != metadata(p["model"]["path"]):
        raise ValueError("model metadata changed")
    if p["warmup_rule"] != WARMUP_RULE or p["warmup_sha256"] != sha(folder / "warmup.json"):
        raise ValueError("warmup protocol changed")
    w = Workload(**p["workload"])
    Settings(**p["settings"]).validate(w)
    warm = read(folder / "warmup.json")
    expected_lengths = [w.short_tokens] * 4 + [w.long_tokens] * 4
    if (warm["schema"] != "bounded-warmup-v1" or warm["model_revision"] != p["model"]["revision"]
            or warm["warmup_s"] != 0 or warm["measurement_s"] != .001 or warm["drain_s"] != 120
            or len(warm["requests"]) != 8):
        raise ValueError("invalid warmup trace")
    for i, (row, length) in enumerate(zip(warm["requests"], expected_lengths)):
        if (row["request_id"] != f"warm-{i}" or row["arrival_ns"] != 0
                or row["max_output_tokens"] != 16 or len(row["prompt_token_ids"]) != length
                or any(type(t) is not int or t < 0 for t in row["prompt_token_ids"])):
            raise ValueError("invalid warmup request")
    expected_names = {job["trace"] for job in p["jobs"]}
    if set(p["trace_sha256"]) != expected_names:
        raise ValueError("trace set changed")
    for name in expected_names:
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != p["trace_sha256"][name]:
            raise ValueError("trace changed")
        trace = read(path)
        validate_trace(trace)
        job = next(j for j in p["jobs"] if j["trace"] == name)
        expected = p["workload"] | {"rps": job["rps"]}
        if (trace["workload"] != expected or trace["seed"] != job["seed"]
                or trace["mode"] != job["mode"] or trace["model_revision"] != p["model"]["revision"]):
            raise ValueError("trace/protocol identity mismatch")
    return p


def find_job(protocol, job_id):
    matches = [j for j in protocol["jobs"] if j["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("unknown job id")
    return matches[0]
