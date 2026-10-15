"""Append-only V3 experiment orchestration, separately frozen from execution code."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys

from benchmarks.v3_profile.protocol import (ROOT, Settings, load_profile, read, save,
    seal, sha, sources, verify_artifacts)
from benchmarks.v3_profile.report import summarize_predictions
from benchmarks.v2_baselines.workload import Workload, build_trace, validate_trace
from benchmarks.v0.environment import model_metadata_hashes

WORKFLOW_FILES = ("tools/v3_refine.py", "tools/v3_online_profile.py",
                  "tools/test_v3_refine.sh", "tools/tests/test_v3_refine.py")


def workflow_sources(root=ROOT):
    return {name: sha(Path(root)/name) for name in WORKFLOW_FILES}


def prediction_summary(rows):
    supported = [r for r in rows if r["prediction"]["supported"]]
    unsupported = [r for r in rows if not r["prediction"]["supported"]]
    result = {"total": len(rows), "coverage": len(supported)/len(rows) if rows else None,
        "supported": summarize_predictions(supported),
        "unsupported": summarize_predictions(unsupported),
        "unsupported_reasons": dict(Counter(r["prediction"]["reason"] for r in unsupported))}
    result["per_phase"] = {}
    for phase in ("decode", "prefill", "mixed"):
        selected = [r for r in rows if ("mixed" if r["features"]["decode_count"] and r["features"]["prefill_tokens"]
            else "prefill" if r["features"]["prefill_tokens"] else "decode") == phase]
        result["per_phase"][phase] = {"total": len(selected),
            "coverage": sum(r["prediction"]["supported"] for r in selected)/len(selected) if selected else None,
            "supported": summarize_predictions([r for r in selected if r["prediction"]["supported"]])}
    return result


def prepare(parent, suite):
    parent, suite = Path(parent).resolve(), Path(suite).resolve()
    protocol, _ = load_profile(parent/"train-411")
    verify_artifacts(parent/"evaluation")
    old_eval = read(parent/"evaluation/report.json")
    if protocol["seed"] != 411 or protocol["split"] != "train" or protocol["smoke_only"]:
        raise ValueError("parent must contain valid original train-411")
    if sources() != protocol["identity"]["source_sha256"]:
        raise ValueError("parent execution sources changed; cannot pool historical samples")
    if model_metadata_hashes(Path(protocol["model_path"])) != protocol["identity"]["model_metadata_sha256"]:
        raise ValueError("parent model metadata changed")
    settings = Settings(**protocol["settings"])
    settings.validate()
    workload = Workload(warmup_s=2, measurement_s=8, rps=2, output_tokens=32)
    if workload.long_tokens+workload.output_tokens > settings.max_model_len:
        raise ValueError("online probe exceeds frozen model length")
    vocab = read(Path(protocol["model_path"])/"config.json")["vocab_size"]
    if vocab < 4:
        raise ValueError("invalid vocabulary")
    suite.mkdir(parents=True, exist_ok=True)
    if any(suite.iterdir()):
        raise ValueError("prepare requires a new empty suite")
    (suite/"traces").mkdir()
    for mode, seed in (("poisson", 612), ("burst", 613), ("long_short", 614)):
        rng = random.Random(seed+900_000)
        trace = build_trace(workload, mode, seed,
            lambda length, index: [rng.randrange(3, vocab) for _ in range(length)],
            protocol["identity"]["model_revision"])
        if not any(2_000_000_000 <= r["arrival_ns"] < 10_000_000_000 for r in trace["requests"]):
            raise ValueError("empty measured online cohort")
        save(suite/f"traces/{mode}.json", trace)
    plan = {"schema": "v3-refine-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": str(parent), "parent_train_protocol_sha256": sha(parent/"train-411/protocol.json"),
        "parent_train_steps_sha256": sha(parent/"train-411/steps.jsonl"),
        "identity": protocol["identity"], "settings": protocol["settings"],
        "model_path": protocol["model_path"], "execution_sources": sources(),
        "workflow_sources": workflow_sources(), "train_seeds": [412, 413], "eval_seed": 512,
        "online_seeds": {"poisson": 612, "burst": 613, "long_short": 614},
        "min_samples": 30, "online_chunk": 256, "drain_s": 60,
        "review_targets": {"coverage": .95, "underprediction_rate": .10,
                           "max_underprediction_ms": 5, "min_window_steps": 30},
        "parent_diagnostic": {"coverage": old_eval["coverage"],
                              "max_underprediction_ms": old_eval["supported"]["max_underprediction_ms"]},
        "scope": "coverage refinement and independent fixed-policy probes; no Slack controller or SLO benefit claim"}
    save(suite/"plan.json", plan)
    seal(suite)  # Only plan and traces exist now; later stage artifacts seal themselves.
    return plan


def load_suite(suite):
    suite = Path(suite).resolve()
    verify_artifacts(suite)
    plan = read(suite/"plan.json")
    if plan["schema"] != "v3-refine-plan-v1" or sources() != plan["execution_sources"]:
        raise ValueError("execution source freeze changed")
    if workflow_sources() != plan["workflow_sources"]:
        raise ValueError("workflow source freeze changed; prepare a new suite")
    parent = Path(plan["parent"])
    if (sha(parent/"train-411/protocol.json") != plan["parent_train_protocol_sha256"]
            or sha(parent/"train-411/steps.jsonl") != plan["parent_train_steps_sha256"]):
        raise ValueError("parent evidence changed")
    if model_metadata_hashes(Path(plan["model_path"])) != plan["identity"]["model_metadata_sha256"]:
        raise ValueError("model metadata changed")
    for mode, seed in plan["online_seeds"].items():
        trace = read(suite/f"traces/{mode}.json")
        validate_trace(trace)
        if trace["seed"] != seed or trace["mode"] != mode:
            raise ValueError("online trace identity differs")
    return plan


def launch(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(command)+"\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"stage failed (exit {code}); evidence retained, log: {log_path}")


def stage(suite, name, command_for_output, validate, launcher=launch):
    suite = Path(suite).resolve()
    checkpoint = suite/"checkpoints"/f"{name}.json"
    if checkpoint.exists():
        data = read(checkpoint)
        out = Path(data["out"])
        if not out.resolve().is_relative_to((suite/"artifacts"/name).resolve()):
            raise ValueError("checkpoint escapes stage directory")
        validate(out)
        if sha(out/"artifact_hashes.json") != data["manifest_sha256"]:
            raise ValueError("completed stage evidence changed")
        print(f"Reuse completed stage: {name}", flush=True)
        return out
    base = suite/"artifacts"/name
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while (base/f"attempt-{number:03}").exists() or (suite/"logs"/name/f"attempt-{number:03}.log").exists():
        number += 1
    out = base/f"attempt-{number:03}"
    print(f"Run stage: {name} -> {out}", flush=True)
    launcher(command_for_output(out), suite/"logs"/name/f"attempt-{number:03}.log")
    validate(out)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save(checkpoint, {"out": str(out.resolve()), "manifest_sha256": sha(out/"artifact_hashes.json")})
    return out


def valid_profile(path, plan, split, seed):
    protocol, _ = load_profile(path)
    if (protocol["identity"] != plan["identity"] or protocol["settings"] != plan["settings"]
            or protocol["split"] != split or protocol["seed"] != seed or protocol["smoke_only"]):
        raise ValueError("stage profile differs from frozen plan")


def run(suite, launcher=launch):
    suite = Path(suite).resolve()
    plan = load_suite(suite)
    with (suite/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another runner is using this suite") from exc
        base = [sys.executable, "-B", "-m", "benchmarks.profile_step_time"]
        parent_train = Path(plan["parent"])/"train-411"
        training = [parent_train]
        for seed in plan["train_seeds"]:
            load_suite(suite)
            def collect_command(out, split="train", seed=seed):
                command = base+["collect", "--model", plan["model_path"], "--model-revision",
                    plan["identity"]["model_revision"], "--split", split, "--seed", str(seed), "--out", str(out)]
                for name, value in plan["settings"].items():
                    command += ["--"+name.replace("_", "-"),
                                ",".join(map(str, value)) if isinstance(value, list) else str(value)]
                return command
            training.append(stage(suite, f"train-{seed}", collect_command,
                lambda out, seed=seed: valid_profile(out, plan, "train", seed), launcher))

        def validate_fit(out):
            verify_artifacts(out)
            table = read(out/"predictor.json")
            if table["identity"] != plan["identity"] or table["min_samples"] != plan["min_samples"]:
                raise ValueError("predictor identity/threshold differs")
            expected = {sha(p/"steps.jsonl") for p in training}
            if {p["steps_sha256"] for p in table["training_profiles"]} != expected:
                raise ValueError("predictor training inputs differ")
        fitted = stage(suite, "predictor", lambda out: base+["fit", "--profiles", *map(str, training),
            "--min-samples", str(plan["min_samples"]), "--out", str(out)], validate_fit, launcher)
        evaluation = stage(suite, "eval-512", lambda out: collect_command(out, "eval", plan["eval_seed"]),
            lambda out: valid_profile(out, plan, "eval", plan["eval_seed"]), launcher)

        def validate_evaluation(out):
            verify_artifacts(out)
            report = read(out/"report.json")
            if report["predictor_sha256"] != sha(fitted/"predictor.json"):
                raise ValueError("evaluation used a different predictor")
            if read(out/"evaluation_inputs.json")[0]["steps_sha256"] != sha(evaluation/"steps.jsonl"):
                raise ValueError("evaluation input differs")
        controlled = stage(suite, "controlled-evaluation", lambda out: base+["evaluate", "--predictor",
            str(fitted/"predictor.json"), "--profiles", str(evaluation), "--out", str(out)],
            validate_evaluation, launcher)
        online = {}
        for mode in plan["online_seeds"]:
            load_suite(suite)
            def validate_online(out, mode=mode):
                verify_artifacts(out)
                report = read(out/"report.json")
                if (report["predictor_sha256"] != sha(fitted/"predictor.json")
                        or report["trace_sha256"] != sha(suite/f"traces/{mode}.json")
                        or report["workflow_sources"] != plan["workflow_sources"]
                        or not report["valid_measurement"]):
                    raise ValueError("invalid online probe identity/execution")
            online[mode] = stage(suite, f"online-{mode}", lambda out, mode=mode:
                [sys.executable, "-B", "-m", "tools.v3_online_profile", "--suite", str(suite),
                 "--mode", mode, "--predictor", str(fitted/"predictor.json"), "--out", str(out)],
                validate_online, launcher)
        report = finish_report(suite, plan, fitted, controlled, online)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report


def finish_report(suite, plan, fitted, controlled, online):
    controlled_report = read(controlled/"report.json")
    checks = []
    target = plan["review_targets"]
    def inspect(label, coverage, errors, count):
        if count < target["min_window_steps"]: checks.append(f"{label}:insufficient_steps")
        if coverage is None or coverage < target["coverage"]: checks.append(f"{label}:coverage")
        if errors["underprediction_rate"] is None or errors["underprediction_rate"] > target["underprediction_rate"]:
            checks.append(f"{label}:underprediction_rate")
        if errors["max_underprediction_ms"] is None or errors["max_underprediction_ms"] > target["max_underprediction_ms"]:
            checks.append(f"{label}:max_underprediction")
    inspect("controlled", controlled_report["coverage"], controlled_report["supported"], controlled_report["total_samples"])
    online_reports = {}
    for mode, folder in online.items():
        report = read(folder/"report.json")
        inspect(mode, report["window"]["coverage"], report["window"]["supported"], report["window"]["total"])
        if not report["all_completed"]: checks.append(f"{mode}:unfinished_or_rejected_requests")
        online_reports[mode] = report
    result = {"schema": "v3-refine-report-v1", "workflow_completed": True,
        "fit": read(fitted/"fit_summary.json"),
        "parent_coverage": plan["parent_diagnostic"]["coverage"],
        "controlled_coverage": controlled_report["coverage"],
        "controlled_supported": controlled_report["supported"],
        "online": {mode: {k: r[k] for k in ("window", "drain", "all_completed", "request_statuses", "preemptions")}
                   for mode, r in online_reports.items()},
        "review_targets": target, "review_targets_met": not checks, "needs_attention": checks,
        "predictor": str(fitted/"predictor.json"), "slack_implemented": False,
        "scope": "review thresholds are prospective diagnostics, not hard SLO guarantees or final V3 acceptance"}
    out = Path(suite)/"final-report"
    if out.exists():
        verify_artifacts(out)
        if read(out/"report.json") != result:
            raise ValueError("existing final report differs; preserve it and use a new suite")
    else:
        out.mkdir()
        save(out/"report.json", result)
        seal(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent", required=True)
    p.add_argument("--suite", required=True)
    for name in ("run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--suite", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.parent, args.suite)
        print(json.dumps({"suite": str(Path(args.suite).resolve()), "train_seeds": plan["train_seeds"],
                          "eval_seed": plan["eval_seed"], "online_seeds": plan["online_seeds"]}, indent=2))
    elif args.command == "run":
        run(args.suite)
    else:
        load_suite(args.suite)
        count = 0
        for file in (Path(args.suite)/"checkpoints").glob("*.json"):
            checkpoint = read(file)
            output = Path(checkpoint["out"])
            if not output.resolve().is_relative_to((Path(args.suite)/"artifacts"/file.stem).resolve()):
                raise ValueError("checkpoint escapes stage directory")
            verify_artifacts(output)
            if sha(output/"artifact_hashes.json") != checkpoint["manifest_sha256"]:
                raise ValueError("completed stage artifact changed")
            count += 1
        print(f"Frozen inputs verified; {count} completed checkpoint artifact sets verified")


if __name__ == "__main__":
    main()
