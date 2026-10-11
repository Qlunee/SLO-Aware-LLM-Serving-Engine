"""State regressions and retained first-branch teacher-forced diagnostics.

Never converts a numerical failure or a near tie into an acceptance pass.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .protocol import ROOT, load_suite, read, save, seal, sha, verify_artifacts
from .suite import audit_run, launch, launch_run, pair_reports, selected


def regress(folder, protocol, experiment):
    chunk = selected(folder)
    directory = folder / "regression"
    if directory.exists():
        verify_artifacts(directory)
        print(json.dumps(read(directory / "report.json"), ensure_ascii=False, indent=2))
        return
    directory.mkdir()
    unit_checks = []
    for pattern in ("test_v1_state.py", "test_v1_attention_gpu.py", "test_v1_reference_gpu.py"):
        code = launch([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                       "-p", pattern, "-v"], directory / f"{pattern}.log")
        unit_checks.append({"pattern": pattern, "exit": code})
    reports = []
    for case in ("mixed", "prefix", "preempt"):
        candidate = directory / f"{case}-candidate"
        command = [sys.executable, "-B", "-m", "benchmarks.v1.check_v1", "candidate",
                   "--model", protocol["model"]["path"], "--model-revision", protocol["model"]["revision"],
                   "--tp", "1", "--batch", str(experiment.decode_count), "--chunk", str(chunk),
                   "--token-budget", str(experiment.token_budget), "--seed", str(experiment.tune_seeds[0]),
                   "--memory", str(experiment.memory), "--prompt-kind", "natural", "--case", case,
                   "--out", str(candidate)]
        if case == "preempt":
            command += ["--kv-block-cap", "2"]
        code = launch(command, directory / f"{case}-candidate.log")
        entry = {"case": case, "candidate_exit": code, "passed_state_checks": False,
                 "reference_exit": None, "reference_passed": False}
        if code == 0:
            verify_artifacts(candidate)
            entry["passed_state_checks"] = read(candidate / "report.json")["passed_state_checks"]
            reference = directory / f"{case}-reference"
            code = launch([sys.executable, "-B", "-m", "benchmarks.v1.check_v1", "reference",
                           "--numerical-policy", "bf16", "--candidate", str(candidate),
                           "--out", str(reference)], directory / f"{case}-reference.log")
            entry["reference_exit"] = code
            if (reference / "report.json").exists():
                entry["reference_passed"] = read(reference / "report.json")["passed"]
        reports.append(entry)
    # Full actual injection trace, with per-Step scans. This is NOT a benchmark.
    checks = []
    for policy in (None, chunk):
        path = launch_run(folder, "eval", experiment.eval_seeds[0], policy,
                          base=directory / "injection-state", verify_state=True)
        passed = False
        if (path / "summary.json").exists():
            trace = read(folder / "eval" / f"seed-{experiment.eval_seeds[0]}.json")
            summary, _, _ = audit_run(path, protocol, experiment, trace, diagnostic=True)
            passed = (summary["end_state_check_passed"] and summary["failed_count"] == 0
                      and summary["completed_count"] == experiment.decode_count+1)
        checks.append({"policy": "baseline" if policy is None else f"chunk-{policy}",
                       "state_and_completion_passed": passed, "path": str(path)})
    report = {"chunk": chunk, "protocol_sha256": sha(folder / "protocol.json"), "cases": reports,
              "unit_checks": unit_checks,
              "injection_state_runs": checks,
              "state_checks_passed": all(r["exit"] == 0 for r in unit_checks)
                                     and all(r["passed_state_checks"] for r in reports)
                                     and all(r["state_and_completion_passed"] for r in checks),
              "legacy_numerical_checks_passed": all(r["reference_passed"] for r in reports),
              "v1_acceptance_passed": False,
              "note": "State and original numerical gates reported separately; original thresholds unchanged. State success permits exploratory V2 only, NOT V1 acceptance."}
    save(directory / "report.json", report)
    seal(directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["state_checks_passed"]:
        raise SystemExit(2)


def diagnose(folder, protocol, experiment):
    chunk = selected(folder)
    directory = folder / "divergence-review"
    if directory.exists():
        verify_artifacts(directory)
        print(f"Preserving prior diagnostic: {directory}")
        return
    pairs = pair_reports(folder, "eval", chunk, experiment.eval_seeds, protocol, experiment)
    if any(not p[k]["valid_experiment"] for p in pairs for k in ("baseline", "candidate")):
        raise ValueError("finish valid paired evaluation first; failed attempts remain visible in report")
    branches = [{"seed": p["seed"], **r} for p in pairs for r in p["divergences"] if not r["exact_match"]]
    directory.mkdir()
    save(directory / "branches.json", branches)
    save(directory / "protocol_identity.json", {"protocol_sha256": sha(folder / "protocol.json")})
    engine = None
    try:
        if branches:
            import torch
            from benchmarks.v1_validation.run import make_engine, legacy_logits, mixed_logits
            from benchmarks.v1_validation.metrics import distribution_metrics
            from benchmarks.v0.environment import collect_environment
            if torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one GPU")
            environment = collect_environment(ROOT, protocol["model"]["path"], protocol["model"]["revision"])
            if environment["model_metadata_sha256"] != protocol["model_metadata_sha256"]:
                raise ValueError("model metadata changed before divergence diagnostic")
            save(directory / "environment.json", environment)
            args = SimpleNamespace(model=protocol["model"]["path"], chunk=chunk, tp=1, graph=False,
                                   token_budget=experiment.token_budget, memory=experiment.memory)
            engine = make_engine(args)
            if engine.config.max_model_len < experiment.max_model_len:
                raise ValueError("diagnostic helper context is too short for this experiment; no truncation")
            evidence, rows = [], []
            for branch in branches:
                if branch["same_history_token_ids"] is None:
                    rows.append({**branch, "status": "LENGTH_MISMATCH_NEEDS_REVIEW"})
                    continue
                history = branch["same_history_token_ids"]
                full = legacy_logits(engine, history, len(history))
                legacy_chunk = legacy_logits(engine, history, chunk)
                mixed = mixed_logits(engine, history, chunk)
                def top2(x):
                    best = x.topk(2)
                    return {"argmax": int(x.argmax()), "top2_ids": best.indices.tolist(),
                            "top2_values": best.values.tolist(), "margin": float(best.values[0]-best.values[1])}
                rows.append({"seed": branch["seed"], "request_id": branch["request_id"],
                    "first_divergence_token_1based": branch["first_divergence_token_1based"],
                    "history_length": len(history), "observed_baseline_token": branch["baseline_token"],
                    "observed_candidate_token": branch["candidate_token"],
                    "mixed_vs_legacy_chunk": distribution_metrics(mixed, legacy_chunk),
                    "legacy_chunk_vs_full": distribution_metrics(legacy_chunk, full),
                    "top2": {"full": top2(full), "legacy_chunk": top2(legacy_chunk), "mixed": top2(mixed)},
                    "status": "NEEDS_REVIEW_NOT_AUTOMATICALLY_EXEMPT"})
                evidence.append({"seed": branch["seed"], "request_id": branch["request_id"],
                                 "history": history, "full": full, "legacy_chunk": legacy_chunk, "mixed": mixed})
            torch.save(evidence, directory / "logits.pt")
        else:
            rows = []
        load_suite(folder)  # Detect source changes during diagnostic.
        report = {"branches": len(branches), "comparisons": rows, "v1_acceptance_passed": False,
                  "status": "MEASURED_NEEDS_REVIEW" if branches else "NO_BRANCH_ON_THESE_TRACES",
                  "scope": "Recompute common history using cold single-request Prefill: full vs original chunk vs Mixed chunk. Does NOT reproduce original online Decode batch shapes or prove the first-branch cause.",
                  "note": "Keep all branches, even when same-chunk logits agree; near-tie is not an exemption. State scans and legacy numerical regression are separate evidence."}
        save(directory / "report.json", report)
        print(f"Divergence review: {directory}")
        print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, ensure_ascii=False, indent=2))
    except Exception as exc:
        save(directory / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(directory)
