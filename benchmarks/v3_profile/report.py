"""Offline fit and independent evaluation; no GPU imports."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from .protocol import load_profile, predictor_module, read, save, seal, sha, verify_artifacts


WORKFLOW_SOURCE_CHANGES = frozenset((
    "benchmarks/profile_step_time.py",
    "benchmarks/v3_profile/protocol.py",
    "benchmarks/v3_profile/driver.py",
    "benchmarks/v3_profile/report.py",
    "benchmarks/v3_slack/run.py",
    "benchmarks/v3_slack/observed_engine.py",
    "nanovllm/engine/slack_scheduler.py",
))


def compatible_identity(left, right):
    """Permit profile matrix/policy edits while retaining the measured V1 path."""
    if {k: v for k, v in left.items() if k != "source_sha256"} != {
            k: v for k, v in right.items() if k != "source_sha256"}:
        return False
    old, new = left.get("source_sha256", {}), right.get("source_sha256", {})
    return all(old.get(name) == new.get(name) or name in WORKFLOW_SOURCE_CHANGES
               or name.startswith("tests/") or name.startswith("scripts/")
               for name in set(old) | set(new))


def profiles(paths, split):
    loaded = [(Path(p).resolve(), *load_profile(p)) for p in paths]
    if not loaded:
        raise ValueError("no profiles")
    identities = [protocol["identity"] for _, protocol, _ in loaded]
    if any(not compatible_identity(identity, identities[-1]) for identity in identities):
        raise ValueError("execution/model/source identities differ across profiles")
    if any(p["split"] != split or p["smoke_only"] for _, p, _ in loaded):
        raise ValueError(f"requires non-smoke {split} profiles")
    seeds = [p["seed"] for _, p, _ in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed/profile in a split")
    return loaded


def fit(paths, out, min_samples=30, widths=None):
    loaded = profiles(paths, "train")
    estimator = predictor_module()
    table = estimator.fit_table([r for _, _, rows in loaded for r in rows],
                                loaded[-1][1]["identity"], min_samples, widths)
    if not any(b["count"] >= min_samples for b in table["buckets"].values()):
        raise ValueError("no adequately sampled buckets; collect more data before fitting")
    table["training_profiles"] = [{"path": str(folder), "seed": p["seed"],
        "protocol_sha256": sha(folder/"protocol.json"), "steps_sha256": sha(folder/"steps.jsonl")}
        for folder, p, _ in loaded]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"predictor.json", table)
    save(out/"fit_summary.json", {"training_samples": table["training_sample_count"],
        "buckets": len(table["buckets"]),
        "supported_buckets": sum(b["count"] >= min_samples for b in table["buckets"].values()),
        "min_samples": min_samples, "bucket_widths": table["bucket_widths"],
        "independent_evaluation_pending": True})
    seal(out)
    return read(out/"fit_summary.json")


def summarize_predictions(rows):
    if not rows:
        return {"count": 0, "underprediction_rate": None,
                "absolute_error_p95_ms": None, "underprediction_p95_ms": None,
                "max_underprediction_ms": None, "mean_signed_error_ms": None}
    estimator = predictor_module()
    errors = [r["actual_ms"]-r["prediction"]["predicted_ms"] for r in rows]
    return {"count": len(rows), "underprediction_rate": sum(e > 0 for e in errors)/len(errors),
        "absolute_error_p95_ms": estimator.percentile([abs(e) for e in errors], .95),
        "underprediction_p95_ms": estimator.percentile([max(0, e) for e in errors], .95),
        "max_underprediction_ms": max(0, max(errors)),
        "mean_signed_error_ms": sum(errors)/len(errors)}


def evaluate(predictor_path, paths, out):
    predictor_path = Path(predictor_path).resolve()
    verify_artifacts(predictor_path.parent)
    table = read(predictor_path)
    loaded = profiles(paths, "eval")
    estimator = predictor_module()
    if any(not compatible_identity(table["identity"], p["identity"])
           for _, p, _ in loaded):
        raise ValueError("predictor and evaluation execution/model/source identity differs")
    predictor = estimator.StepPredictor(table, table["identity"])
    train_seeds = {p["seed"] for p in table["training_profiles"]}
    if train_seeds.intersection(p["seed"] for _, p, _ in loaded):
        raise ValueError("training/evaluation seeds overlap")
    train_hashes = {p["steps_sha256"] for p in table["training_profiles"]}
    if train_hashes.intersection(sha(folder/"steps.jsonl") for folder, _, _ in loaded):
        raise ValueError("training/evaluation Step records overlap")
    rows, costs = [], []
    # Warm Python's path before timing individual predictor calls.
    for _, _, samples in loaded:
        predictor.predict(samples[0]["features"])
    for folder, protocol, samples in loaded:
        for sample in samples:
            if sample["warmup"]:
                continue
            start = perf_counter_ns()
            prediction = predictor.predict(sample["features"])
            costs.append((perf_counter_ns()-start)/1000)
            rows.append({"profile": str(folder), "seed": protocol["seed"],
                "step_id": sample["step_id"], "case_id": sample["case_id"],
                "features": sample["features"], "actual_ms": sample["step_time_ms"],
                "prediction": asdict(prediction)})
    if not rows:
        raise ValueError("no independent measured samples")
    supported = [r for r in rows if r["prediction"]["supported"]]
    unsupported = [r for r in rows if not r["prediction"]["supported"]]
    per_phase = {}
    for phase in ("decode", "prefill", "mixed"):
        selected = [r for r in rows if ("mixed" if r["features"]["decode_count"] and r["features"]["prefill_tokens"]
                    else "decode" if not r["features"]["prefill_tokens"] else "prefill") == phase]
        per_phase[phase] = {"total": len(selected),
            "supported": summarize_predictions([r for r in selected if r["prediction"]["supported"]])}
    summary = {"schema": "v3-step-evaluation-v1", "independent_seeds": True,
        "total_samples": len(rows), "coverage": len(supported)/len(rows),
        "supported": summarize_predictions(supported),
        "unsupported": summarize_predictions(unsupported),
        "all_including_unsupported": summarize_predictions(rows), "per_phase": per_phase,
        "predict_cpu_p50_us": estimator.percentile(costs, .5),
        "predict_cpu_p95_us": estimator.percentile(costs, .95),
        "predict_cpu_max_us": max(costs), "predictor_sha256": sha(predictor_path),
        "scope": "controlled batches, not independent online workload distribution or SLO acceptance",
        "slo_scheduler_implemented": False, "v1_numerical_acceptance_passed": False}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"report.json", summary)
    save(out/"predictions.json", rows)
    save(out/"evaluation_inputs.json", [{"path": str(folder), "seed": p["seed"],
        "protocol_sha256": sha(folder/"protocol.json"), "steps_sha256": sha(folder/"steps.jsonl")}
        for folder, p, _ in loaded])
    seal(out)
    return summary

