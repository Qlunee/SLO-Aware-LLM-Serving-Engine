"""Recompute metrics from sealed records; never trust the outcome flag alone."""
from pathlib import Path
from collections import Counter

from benchmarks.v2_stall.protocol import ROOT, read, sha, verify_artifacts
from benchmarks.v2_stall.metrics import records_from_rows
from benchmarks.v0.metrics import request_metrics
from benchmarks.v2_baselines.metrics import analyze_online
from .protocol import load_suite
from .run import identity

VALID = {"VALID_COMPLETE", "VALID_INCOMPLETE"}
FAILED = {"SETUP_FAILED", "WARMUP_FAILED", "MEASUREMENT_FAILED", "CLEANUP_FAILED", "INTERRUPTED"}


def audit_attempt(suite, job, out, protocol=None, root=ROOT):
    suite, out = Path(suite).resolve(), Path(out).resolve()
    protocol = protocol or load_suite(suite, root=root)
    verify_artifacts(out)
    if read(out / "identity.json") != identity(suite, protocol, job):
        raise ValueError("run identity differs from frozen protocol")
    outcome = read(out / "outcome.json")
    status = outcome["status"]
    if status not in VALID | FAILED:
        raise ValueError("unknown outcome")
    if status not in VALID:
        return {"status": status, "path": str(out), "outcome": outcome,
                "valid_for_performance": False}
    warm = out / "warmup"
    verify_artifacts(warm)
    if read(warm / "trace.json") != read(suite / "warmup.json"):
        raise ValueError("warmup trace differs")
    wr = read(warm / "report.json")
    records = records_from_rows(read(warm / "requests.json"))
    expected = read(suite / "warmup.json")["requests"]
    if len(records) != 8 or len({r.request_id for r in records}) != 8:
        raise ValueError("warmup must retain all eight records")
    by_id = {r.request_id: r for r in records}
    for spec in expected:
        r = by_id.get(spec["request_id"])
        if (r is None or r.status != "completed" or len(r.output_token_ids) != 16
                or r.max_output_tokens != 16 or r.arrival_ns != 0
                or r.prompt_tokens != len(spec["prompt_token_ids"])):
            raise ValueError("warmup incomplete/identity mismatch")
        request_metrics(r)
    warm_steps, warm_backlog = read(warm / "steps.json"), read(warm / "backlog.json")
    if (not wr["passed"] or wr["driver_error"] is not None
            or wr["status_counts"] != dict(Counter(r.status for r in records))
            or wr["step_count"] != len(warm_steps)
            or wr["last_backlog"] != (warm_backlog[-1] if warm_backlog else None)
            or wr["phases"] != sorted({s["phase"] for s in warm_steps})
            or not all(wr[k] for k in ("cold_prefix_metadata_reset", "observations_reset", "no_gpu_memory_reallocation"))):
        raise ValueError("warmup report audit failed")
    trace = read(out / "trace.json")
    if trace != read(suite / job["trace"]):
        raise ValueError("measurement trace differs")
    summary = read(out / "summary.json")
    records = records_from_rows(read(out / "requests.json"))
    recalc = analyze_online(trace, records, read(out / "steps.json"), read(out / "backlog.json"),
                            summary["end_ns"], protocol["settings"]["starvation_ms"])
    for key, value in recalc.items():
        if summary.get(key) != value:
            raise ValueError(f"raw metric audit mismatch: {key}")
    for key in ("policy", "mode", "seed", "rps"):
        if summary[key] != job[key]:
            raise ValueError("summary identity mismatch")
    if (summary["protocol_sha256"] != sha(suite / "protocol.json") or summary["job_id"] != job["id"]
            or not summary["valid_measurement"] or not summary["end_state_check_passed"]
            or not summary["source_unchanged"] or summary["diagnostic_only"]
            or summary["warmup"] != wr or (out / "error.json").exists()
            or (out / "state_error.json").exists()
            or (status == "VALID_COMPLETE") != summary["all_completed"]
            or summary["deadline_ns"] != round((trace["warmup_s"] + trace["measurement_s"]
                                                + protocol["settings"]["drain_s"]) * 1e9)):
        raise ValueError("invalid run marked valid")
    env = read(out / "environment.json")
    if (env["model_metadata_sha256"] != protocol["model_metadata_sha256"]
            or env["model_revision_declared"] != protocol["model"]["revision"]
            or env["source_sha256"] != {k: v for k, v in protocol["source_sha256"].items()
                                       if k.startswith("nanovllm/")}):
        raise ValueError("execution source/model mismatch")
    return {"status": status, "path": str(out), "valid_for_performance": True,
            "summary": summary, "records": records, "environment": env}
