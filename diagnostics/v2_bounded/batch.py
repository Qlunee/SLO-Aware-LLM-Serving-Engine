"""Append-only attempts; failures/partial runs are never silently overwritten."""
from pathlib import Path
import fcntl
import re
import subprocess
import sys
import tempfile

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, verify_artifacts
from .run import identity
from .audit import audit_attempt
from .protocol import load_suite


def resume_decision(suite, job, protocol, root=ROOT):
    suite = Path(suite)
    base = suite / "runs" / job["id"]
    attempts = []
    if base.exists():
        for path in sorted(base.iterdir()):
            if path.is_file() and re.fullmatch(r"attempt-\d{4}\.log", path.name):
                if path.is_symlink():
                    raise ValueError("symlinked attempt log")
                if not path.with_suffix("").exists():
                    attempts.append({"path": str(path.with_suffix("")), "status": "PARTIAL"})
                continue
            if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"attempt-\d{4}", path.name):
                raise ValueError("unexpected attempt entry")
            if (path / "artifact_hashes.json").exists():
                verify_artifacts(path)
                if read(path / "identity.json") != identity(suite, protocol, job):
                    raise ValueError("attempt belongs to another protocol/job")
                status = audit_attempt(suite, job, path, protocol, root=root)["status"]
            else:
                status = "PARTIAL"
            attempts.append({"path": str(path), "status": status})
    return {"action": "PRESERVE" if attempts else "RUN", "attempts": attempts}


def launch_process(command, log):
    with log.open("x", encoding="utf-8") as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def run_batch(suite, job_ids=None, launch=launch_process, root=ROOT, max_new=None,
              retry_job=None, reason=None):
    """Resume without retries; an explicit retry always creates a new attempt."""
    suite = Path(suite).resolve()
    protocol = load_suite(suite, root=root)
    if max_new is not None and (type(max_new) is not int or max_new < 1):
        raise ValueError("max-new must be positive")
    jobs = [j for j in protocol["jobs"] if job_ids is None or j["id"] in job_ids]
    if job_ids is not None and set(job_ids) != {j["id"] for j in jobs}:
        raise ValueError("unknown job ids")
    if retry_job is not None:
        if not reason or not reason.strip():
            raise ValueError("explicit retry requires a reason")
        jobs = [j for j in jobs if j["id"] == retry_job]
        if len(jobs) != 1:
            raise ValueError("unknown retry job")
    report = {"launched": 0, "preserved": 0, "jobs": [], "retry_reason": reason,
              "v1_acceptance_passed": False}
    directory = Path(tempfile.mkdtemp(prefix="batch.", dir=suite))
    with (suite / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another batch/retry holds the suite lock")
        try:
            for job in jobs:
                decision = resume_decision(suite, job, protocol, root=root)
                if decision["action"] == "PRESERVE" and retry_job is None:
                    report["preserved"] += 1
                    report["jobs"].append({"id": job["id"], **decision})
                    print("PRESERVE", job["id"], [a["status"] for a in decision["attempts"]], flush=True)
                    continue
                if max_new is not None and report["launched"] >= max_new:
                    break
                if retry_job is not None and decision["action"] == "RUN":
                    raise ValueError("retry requires an existing failed/partial attempt")
                if retry_job is not None and any(a["status"].startswith("VALID_") for a in decision["attempts"]):
                    raise ValueError("refuse to retry a valid measurement to cherry-pick outcomes")
                base = suite / "runs" / job["id"]
                base.mkdir(parents=True, exist_ok=True)
                existing = [int(p.stem.split("-")[1]) for p in base.iterdir()
                            if re.fullmatch(r"attempt-\d{4}(\.log)?", p.name)]
                number = max(existing, default=0) + 1
                if number > 9999:
                    raise ValueError("attempt limit reached")
                out = base / f"attempt-{number:04d}"
                command = [sys.executable, "-B", "-m", "diagnostics.v2_bounded", "run",
                           "--suite", str(suite), "--job", job["id"], "--out", str(out)]
                print("LAUNCH", job["id"], "log:", out.with_suffix(".log"), flush=True)
                code = launch(command, out.with_suffix(".log"))
                report["launched"] += 1
                load_suite(suite, root=root)
                after = resume_decision(suite, job, protocol, root=root)
                if after["action"] == "RUN":
                    # Process failed before creating its run directory: preserve a partial marker.
                    out.mkdir(exist_ok=False)
                    after = resume_decision(suite, job, protocol, root=root)
                report["jobs"].append({"id": job["id"], "process_exit": code, **after})
                if code == 130 or code < 0:
                    raise KeyboardInterrupt("batch interrupted; attempts retained")
        except BaseException:
            import traceback
            report["batch_exception"] = traceback.format_exc()
            raise
        finally:
            save(directory / "report.json", report)
            seal(directory)
    print("Batch ledger:", directory, flush=True)
    return report
