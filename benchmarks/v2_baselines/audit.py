"""Read-only audit from frozen inputs and raw records, not trusted summaries."""
from pathlib import Path

from benchmarks.v2_stall.metrics import records_from_rows
from benchmarks.v2_stall.protocol import ROOT, read, sha, verify_artifacts
from .metrics import analyze_online
from .protocol import load, POLICIES


def audit_run(suite, folder, root=ROOT):
    suite, folder = Path(suite), Path(folder)
    protocol = load(suite, root=root)
    verify_artifacts(folder)
    config, summary = read(folder/"run_config.json"), read(folder/"summary.json")
    if (config["protocol_sha256"] != sha(suite/"protocol.json")
            or config["policy"] not in POLICIES or config["tp"] != 1
            or config["graph"] or config["observation"] != "full"):
        raise ValueError("run configuration differs from frozen protocol")
    for name in ("policy", "mode", "seed", "diagnostic_only"):
        if config[name] != summary[name]:
            raise ValueError("run summary identity mismatch")
    trace = read(folder/"trace.json")
    relative = f"traces/{config['mode']}-seed-{config['seed']}.json"
    if relative not in protocol["trace_sha256"] or trace != read(suite/relative):
        raise ValueError("wrong Trace identity")
    records = records_from_rows(read(folder/"requests.json"))
    recalculated = analyze_online(trace, records, read(folder/"steps.json"),
        read(folder/"backlog.json"), summary["end_ns"], protocol["settings"]["starvation_ms"])
    for key, value in recalculated.items():
        if summary.get(key) != value:
            raise ValueError(f"summary audit mismatch: {key}")
    valid = (summary["source_unchanged"] and summary["end_state_check_passed"]
             and not (folder/"error.json").exists())
    if summary["valid_measurement"] != bool(valid):
        raise ValueError("invalid measurement falsely marked valid")
    env = read(folder/"environment.json")
    if (env["model_metadata_sha256"] != protocol["model_metadata_sha256"]
            or env["source_sha256"] != {k: v for k, v in protocol["source_sha256"].items()
                                       if k.startswith("nanovllm/")}):
        raise ValueError("execution model/source differs from protocol")
    return summary, records, env
