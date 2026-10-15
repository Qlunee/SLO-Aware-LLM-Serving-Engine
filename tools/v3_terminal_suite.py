"""Prospective terminal-Chunk train/eval suite with fresh online holdouts.

The suite never fits from earlier diagnostic runs or online holdouts.  It
reuses the frozen V3 execution path and keeps all outputs append-only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from time import perf_counter_ns

from benchmarks.v0.environment import model_metadata_hashes
from benchmarks.v2_baselines.workload import Workload, build_trace, validate_trace
from benchmarks.v3_profile.protocol import (ROOT, load_profile, predictor_module,
    read, save, seal, sha, sources, verify_artifacts)
from benchmarks.v3_profile.report import summarize_predictions
from tools.v3_refine import load_suite as load_refine_suite, stage


SCHEMA = "v3-terminal-suite-v1"
SOURCE_FILES = ("tools/v3_terminal_suite.py", "tools/v3_terminal_probe.py",
                "tools/v3_drift_replay.py")
SPECS = {
    "train-uniform": {"split": "train", "seed": 720, "pattern": "uniform",
                      "decode_counts": "1,2,3,4,5,6,7", "contexts": "256,1024"},
    "train-one-long": {"split": "train", "seed": 721, "pattern": "one_long",
                       "decode_counts": "2,3,4,5,6,7", "contexts": "1024"},
    "eval-uniform": {"split": "eval", "seed": 820, "pattern": "uniform",
                     "decode_counts": "1,2,3,4,5,6,7", "contexts": "256,1024"},
    "eval-one-long": {"split": "eval", "seed": 821, "pattern": "one_long",
                      "decode_counts": "2,3,4,5,6,7", "contexts": "1024"},
}
ONLINE_SEEDS = {
    "r1": {"poisson": 912, "burst": 913, "long_short": 914},
    "r2": {"poisson": 922, "burst": 923, "long_short": 924},
    "r3": {"poisson": 932, "burst": 933, "long_short": 934},
}
REPEATS, WARMUP, MIN_SAMPLES = 80, 3, 30


def workflow_sources():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def prepare(base_suite, base_predictor, regression_profile, out):
    base_suite = Path(base_suite).resolve()
    base_plan = load_refine_suite(base_suite)
    base_predictor = Path(base_predictor).resolve()
    regression_profile = Path(regression_profile).resolve()
    verify_artifacts(base_predictor.parent)
    table = read(base_predictor)
    regression_protocol, _ = load_profile(regression_profile)
    if table["identity"] != base_plan["identity"] or regression_protocol["identity"] != table["identity"]:
        raise ValueError("base predictor, suite and regression profile identities differ")
    if regression_protocol["split"] != "eval":
        raise ValueError("regression profile must be an eval split")
    used_seeds = {p["seed"] for p in table["training_profiles"]}
    used_seeds.add(regression_protocol["seed"])
    online_seed_values = [seed for replica in ONLINE_SEEDS.values() for seed in replica.values()]
    proposed = {x["seed"] for x in SPECS.values()} | set(online_seed_values)
    if len(proposed) != len(SPECS) + len(online_seed_values) or used_seeds & proposed:
        raise ValueError("new suite seeds overlap")
    if sources() != base_plan["execution_sources"]:
        raise ValueError("frozen execution sources changed")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("prepare requires an empty suite directory")
    workload = Workload(warmup_s=2, measurement_s=8, rps=2, output_tokens=32)
    vocab = read(Path(base_plan["model_path"]) / "config.json")["vocab_size"]
    trace_hashes, online_manifest_hashes = {}, {}
    for replica, seeds in ONLINE_SEEDS.items():
        online = out / "online_plan" / replica
        (online / "traces").mkdir(parents=True)
        online_plan = json.loads(json.dumps(base_plan))
        online_plan["created_utc"] = datetime.now(timezone.utc).isoformat()
        online_plan["online_seeds"] = seeds
        online_plan["scope"] = "fresh held-out online traces for terminal suite"
        trace_hashes[replica] = {}
        for mode, seed in seeds.items():
            rng = random.Random(seed + 900_000)
            trace = build_trace(workload, mode, seed,
                lambda length, index: [rng.randrange(3, vocab) for _ in range(length)],
                base_plan["identity"]["model_revision"])
            validate_trace(trace)
            save(online / f"traces/{mode}.json", trace)
            trace_hashes[replica][mode] = sha(online / f"traces/{mode}.json")
        save(online / "plan.json", online_plan)
        seal(online)
        load_refine_suite(online)
        online_manifest_hashes[replica] = sha(online / "artifact_hashes.json")
    plan = {"schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_suite": str(base_suite), "base_predictor": str(base_predictor),
        "base_predictor_sha256": sha(base_predictor),
        "regression_profile": str(regression_profile),
        "regression_protocol_sha256": sha(regression_profile / "protocol.json"),
        "regression_steps_sha256": sha(regression_profile / "steps.jsonl"),
        "online_plan_manifest_sha256": online_manifest_hashes,
        "online_trace_sha256": trace_hashes, "identity": table["identity"],
        "execution_sources": sources(), "workflow_sources": workflow_sources(),
        "specs": SPECS, "online_seeds": ONLINE_SEEDS, "repeats": REPEATS,
        "warmup": WARMUP, "min_samples": MIN_SAMPLES,
        "review_targets": {"coverage": .95, "underprediction_rate": .10,
                           "max_underprediction_ms": 5},
        "scope": "prospective terminal training, controlled eval and fresh online eval; no Slack claim"}
    save(out / "plan.json", plan)
    seal(out)
    return {"suite": str(out), "terminal_train_steps": (14 + 6) * REPEATS,
            "terminal_eval_steps": (14 + 6) * REPEATS,
            "online_seeds": ONLINE_SEEDS, "workflow_sources": plan["workflow_sources"]}


def load_plan(suite):
    suite = Path(suite).resolve()
    verify_artifacts(suite)
    plan = read(suite / "plan.json")
    if plan.get("schema") != SCHEMA or plan["workflow_sources"] != workflow_sources():
        raise ValueError("suite workflow source changed")
    if plan["execution_sources"] != sources():
        raise ValueError("frozen execution source changed")
    if sha(plan["base_predictor"]) != plan["base_predictor_sha256"]:
        raise ValueError("base predictor changed")
    if (sha(Path(plan["regression_profile"]) / "protocol.json") != plan["regression_protocol_sha256"]
            or sha(Path(plan["regression_profile"]) / "steps.jsonl") != plan["regression_steps_sha256"]):
        raise ValueError("regression evidence changed")
    for replica, seeds in plan["online_seeds"].items():
        online_dir = suite / "online_plan" / replica
        if sha(online_dir / "artifact_hashes.json") != plan["online_plan_manifest_sha256"][replica]:
            raise ValueError("online plan manifest changed")
        online = load_refine_suite(online_dir)
        if online["online_seeds"] != seeds:
            raise ValueError("online seeds changed")
        if model_metadata_hashes(Path(online["model_path"])) != plan["identity"]["model_metadata_sha256"]:
            raise ValueError("model metadata changed")
    return plan


def terminal_rows(folder, plan, name):
    folder = Path(folder)
    verify_artifacts(folder)
    protocol, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    spec = plan["specs"][name]
    if (protocol.get("schema") != "v3-terminal-probe-v1" or protocol.get("split") != spec["split"]
            or protocol.get("seed") != spec["seed"] or protocol.get("identity") != plan["identity"]
            or protocol.get("probe_source_sha256") != plan["workflow_sources"]["tools/v3_terminal_probe.py"]
            or protocol.get("predictor_sha256") != plan["base_predictor_sha256"]
            or protocol.get("repeats") != plan["repeats"] or protocol.get("warmup") != plan["warmup"]
            or not summary.get("valid") or summary.get("preemptions") != 0
            or not summary.get("source_unchanged")):
        raise ValueError(f"invalid terminal profile: {name}")
    rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    if (len(rows) != summary["recorded_steps"] or
            sum(not r["warmup"] for r in rows) != summary["measured_steps"] or
            summary["measured_steps"] != len(protocol["cases"]) * plan["repeats"] or
            len(summary["completed_cases"]) != len(protocol["cases"])):
        raise ValueError(f"incomplete terminal profile: {name}")
    expected = {c["case_id"] for c in protocol["cases"]}
    if set(summary["completed_cases"]) != expected:
        raise ValueError("terminal case list differs")
    counts = Counter(r["case_id"] for r in rows if not r["warmup"])
    if set(counts) != expected or any(n != plan["repeats"] for n in counts.values()):
        raise ValueError("terminal samples missing per case")
    estimator = predictor_module()
    for i, row in enumerate(rows):
        if row["step_id"] != i or row["case_id"] not in expected:
            raise ValueError("terminal Step order/identity differs")
        estimator.validate_features(row["features"])
        if (row["features"]["prefill_tokens"] != 256
                or row["features"]["sample_count"] != row["features"]["decode_count"] + 1
                or row["features"]["prefill_has_history"]):
            raise ValueError("not a terminal mixed sample")
    return protocol, rows


def fit(suite, inputs, out):
    plan = load_plan(suite)
    base_path = Path(plan["base_predictor"])
    base = read(base_path)
    rows = []
    profiles = []
    for item in base["training_profiles"]:
        folder = Path(item["path"])
        protocol, samples = load_profile(folder)
        if (protocol["identity"] != plan["identity"] or protocol["split"] != "train"
                or sha(folder / "protocol.json") != item["protocol_sha256"]
                or sha(folder / "steps.jsonl") != item["steps_sha256"]):
            raise ValueError("base training profile changed")
        rows.extend(samples)
        profiles.append(item)
    for name, folder in zip(("train-uniform", "train-one-long"), inputs):
        protocol, samples = terminal_rows(folder, plan, name)
        rows.extend(samples)
        profiles.append({"path": str(Path(folder).resolve()), "seed": protocol["seed"],
            "protocol_sha256": sha(Path(folder) / "protocol.json"),
            "steps_sha256": sha(Path(folder) / "steps.jsonl"),
            "source": "prospective_terminal"})
    if len(inputs) != 2 or len({x["seed"] for x in profiles}) != len(profiles):
        raise ValueError("train profiles/seeds incomplete")
    estimator = predictor_module()
    table = estimator.fit_table(rows, plan["identity"], plan["min_samples"])
    table["training_profiles"] = profiles
    table["training_plan_sha256"] = sha(Path(suite) / "plan.json")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "predictor.json", table)
    save(out / "fit_summary.json", {"training_samples": table["training_sample_count"],
        "buckets": len(table["buckets"]),
        "supported_buckets": sum(b["count"] >= plan["min_samples"] for b in table["buckets"].values()),
        "new_terminal_measured_samples": 20 * plan["repeats"],
        "scope": "combined frozen base training and new prospective terminal training"})
    seal(out)
    return read(out / "fit_summary.json")


def evaluate_terminal(suite, predictor_path, inputs, out):
    plan = load_plan(suite)
    predictor_path = Path(predictor_path).resolve()
    verify_artifacts(predictor_path.parent)
    predictor = predictor_module().StepPredictor(read(predictor_path), plan["identity"])
    evaluated, costs = [], []
    for name, folder in zip(("eval-uniform", "eval-one-long"), inputs):
        protocol, samples = terminal_rows(folder, plan, name)
        for row in samples:
            if row["warmup"]:
                continue
            start = perf_counter_ns()
            prediction = predictor.predict(row["features"])
            costs.append((perf_counter_ns() - start) / 1000)
            evaluated.append({"case_id": row["case_id"], "seed": protocol["seed"],
                "step_id": row["step_id"], "actual_ms": row["step_time_ms"],
                "features": row["features"], "prediction": vars(prediction)})
    if len(inputs) != 2 or not evaluated:
        raise ValueError("terminal evaluation incomplete")
    supported = [x for x in evaluated if x["prediction"]["supported"]]
    estimator = predictor_module()
    report = {"schema": "v3-terminal-evaluation-v1", "total": len(evaluated),
        "supported": len(supported), "coverage": len(supported) / len(evaluated),
        "supported_errors": summarize_predictions(supported),
        "unsupported_errors": summarize_predictions([x for x in evaluated if not x["prediction"]["supported"]]),
        "predict_cpu_p95_us": estimator.percentile(costs, .95),
        "predictor_sha256": sha(predictor_path),
        "evaluation_seeds": [plan["specs"][x]["seed"] for x in ("eval-uniform", "eval-one-long")],
        "scope": "independent controlled terminal shapes; online SLO acceptance separate"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "report.json", report)
    save(out / "predictions.json", evaluated)
    seal(out)
    return report


def run(suite):
    suite = Path(suite).resolve()
    plan = load_plan(suite)
    terminal = {}
    for name in ("train-uniform", "train-one-long"):
        spec = plan["specs"][name]
        def command(out, spec=spec):
            return [sys.executable, "-B", "-m", "tools.v3_terminal_probe",
                "--suite", str(suite / "online_plan/r1"), "--predictor", plan["base_predictor"],
                "--split", spec["split"], "--seed", str(spec["seed"]),
                "--decode-counts", spec["decode_counts"], "--contexts", spec["contexts"],
                "--context-pattern", spec["pattern"], "--prefill-tokens", "256",
                "--repeats", str(plan["repeats"]), "--warmup", str(plan["warmup"]),
                "--out", str(out)]
        terminal[name] = stage(suite, name, command,
            lambda out, name=name: terminal_rows(out, plan, name))
    fitted = stage(suite, "predictor",
        lambda out: [sys.executable, "-B", "-m", "tools.v3_terminal_suite", "fit",
            "--suite", str(suite), "--inputs", str(terminal["train-uniform"]),
            str(terminal["train-one-long"]), "--out", str(out)],
        lambda out: verify_artifacts(out))
    for name in ("eval-uniform", "eval-one-long"):
        spec = plan["specs"][name]
        terminal[name] = stage(suite, name,
            lambda out, spec=spec: [sys.executable, "-B", "-m", "tools.v3_terminal_probe",
                "--suite", str(suite / "online_plan/r1"), "--predictor", plan["base_predictor"],
                "--split", spec["split"], "--seed", str(spec["seed"]),
                "--decode-counts", spec["decode_counts"], "--contexts", spec["contexts"],
                "--context-pattern", spec["pattern"], "--prefill-tokens", "256",
                "--repeats", str(plan["repeats"]), "--warmup", str(plan["warmup"]),
                "--out", str(out)],
            lambda out, name=name: terminal_rows(out, plan, name))
    terminal_eval = stage(suite, "terminal-evaluation",
        lambda out: [sys.executable, "-B", "-m", "tools.v3_terminal_suite", "evaluate",
            "--suite", str(suite), "--predictor", str(fitted / "predictor.json"),
            "--inputs", str(terminal["eval-uniform"]), str(terminal["eval-one-long"]),
            "--out", str(out)], lambda out: verify_artifacts(out))
    regression = stage(suite, "controlled-regression",
        lambda out: [sys.executable, "-B", "-m", "benchmarks.profile_step_time", "evaluate",
            "--predictor", str(fitted / "predictor.json"),
            "--profiles", plan["regression_profile"], "--out", str(out)],
        lambda out: verify_artifacts(out))
    online, replays = {}, {}
    for replica, seeds in plan["online_seeds"].items():
        for mode in seeds:
            key = (replica, mode)
            online[key] = stage(suite, f"online-{replica}-{mode}",
                lambda out, replica=replica, mode=mode: [sys.executable, "-B", "-m", "tools.v3_online_profile",
                    "--suite", str(suite / "online_plan" / replica), "--mode", mode,
                    "--predictor", str(fitted / "predictor.json"), "--out", str(out)],
                lambda out, replica=replica, mode=mode: valid_online(out, suite, replica, mode, fitted / "predictor.json"))
            replays[key] = stage(suite, f"replay-{replica}-{mode}",
                lambda out, key=key: [sys.executable, "-B", "-m", "tools.v3_drift_replay",
                    "--kind", "online", "--input", str(online[key] / "steps.json"),
                    "--out", str(out)], lambda out: verify_artifacts(out))
    return review(suite, fitted, terminal_eval, regression, online, replays)


def valid_online(folder, suite, replica, mode, predictor_path):
    verify_artifacts(folder)
    report = read(Path(folder) / "report.json")
    if (not report.get("valid_measurement") or not report.get("all_completed")
            or report.get("predictor_sha256") != sha(predictor_path)
            or report.get("trace_sha256") != sha(Path(suite) / "online_plan" / replica / f"traces/{mode}.json")):
        raise ValueError(f"invalid online holdout: {replica}/{mode}")


def review(suite, fitted, terminal_eval, regression, online, replays):
    out = Path(suite) / "review"
    if out.exists():
        verify_artifacts(out)
        return read(out / "report.json")
    plan = load_plan(suite)
    target = plan["review_targets"]
    checks = []
    summaries = {mode: [] for mode in ("poisson", "burst", "long_short")}
    for (replica, mode), folder in online.items():
        report = read(folder / "report.json")
        window = report["window"]
        errors = window["supported"]
        phase = window["per_phase"]
        replay = read(replays[(replica, mode)] / "summary.json")
        summaries[mode].append({"replica": replica, "seed": plan["online_seeds"][replica][mode],
            "steps": window["total"], "coverage": window["coverage"],
            "supported_underprediction_rate": errors["underprediction_rate"],
            "supported_max_underprediction_ms": errors["max_underprediction_ms"],
            "mixed_coverage": phase["mixed"]["coverage"],
            "decode_coverage": phase["decode"]["coverage"],
            "preemptions": report["preemptions"],
            "causal_replay": {k: replay[k] for k in
                ("calibrated_coverage", "calibrated", "cold_start_supported_steps")}})
        if window["coverage"] is None or window["coverage"] < target["coverage"]:
            checks.append(f"{replica}/{mode}:coverage")
        if (errors["underprediction_rate"] is None
                or errors["underprediction_rate"] > target["underprediction_rate"]):
            checks.append(f"{replica}/{mode}:underprediction_rate")
        if (errors["max_underprediction_ms"] is None
                or errors["max_underprediction_ms"] > target["max_underprediction_ms"]):
            checks.append(f"{replica}/{mode}:max_underprediction")
    result = {"schema": "v3-terminal-suite-review-v1",
        "workflow_completed": True, "predictor_review_passed": not checks,
        "failed_checks": checks, "review_targets": target,
        "fit": read(fitted / "fit_summary.json"),
        "terminal_evaluation": read(terminal_eval / "report.json"),
        "controlled_regression": read(regression / "report.json"),
        "online": summaries,
        "scope": "independent predictor evidence only; Slack controller is not implemented"}
    out.mkdir(exist_ok=False)
    save(out / "report.json", result)
    seal(out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--base-suite", required=True)
    p.add_argument("--base-predictor", required=True)
    p.add_argument("--regression-profile", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--suite", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--suite", required=True)
    p.add_argument("--inputs", nargs=2, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--suite", required=True)
    p.add_argument("--predictor", required=True)
    p.add_argument("--inputs", nargs=2, required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.base_suite, args.base_predictor, args.regression_profile, args.out)
    elif args.command == "run":
        result = run(args.suite)
    elif args.command == "fit":
        result = fit(args.suite, args.inputs, args.out)
    else:
        result = evaluate_terminal(args.suite, args.predictor, args.inputs, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
