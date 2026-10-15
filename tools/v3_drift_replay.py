"""CPU-only, causal replay of a diagnostic Step-latency drift guard.

This does not change a fitted predictor or enable a Slack controller.  In a
replay, each Step is predicted before its own observed duration is added to
the guard history.  The input artifact must already be sealed and valid.
"""
import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter_ns

from benchmarks.v3_profile.protocol import read, save, seal, sha, verify_artifacts
from benchmarks.v3_profile.report import summarize_predictions


WINDOW = 128
MIN_OBSERVATIONS = 32
QUANTILE = .95


def nearest_rank(values, quantile):
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


@dataclass
class DriftGuard:
    history: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def predict(self, prediction):
        raw = prediction["predicted_ms"]
        supported = prediction["supported"]
        if not supported or len(self.history) < MIN_OBSERVATIONS:
            return raw, False, 0.0
        margin = max(0.0, nearest_rank(self.history, QUANTILE))
        return raw + margin, True, margin

    def observe(self, prediction, actual_ms):
        if prediction["supported"]:
            self.history.append(actual_ms - prediction["predicted_ms"])


def metrics(rows, key):
    shaped = [{"actual_ms": r["actual_ms"],
               "prediction": {"predicted_ms": r[key]}} for r in rows]
    return summarize_predictions(shaped)


def replay(rows, kind, clock=perf_counter_ns):
    if kind not in ("controlled", "online"):
        raise ValueError("unknown replay kind")
    guard = DriftGuard()
    previous_profile, previous_step = None, -1
    results, cpu_us = [], []
    for row in rows:
        profile = row.get("profile") if kind == "controlled" else "online"
        step = row["step_id"]
        if type(step) is not int or step < 0:
            raise ValueError("invalid Step ID")
        if profile != previous_profile:
            guard, previous_step = DriftGuard(), -1
            previous_profile = profile
        if step <= previous_step:
            raise ValueError("Steps must be in causal order within each profile")
        previous_step = step
        prediction = row["prediction"]
        raw = prediction["predicted_ms"]
        actual = row["actual_ms"]
        if (type(prediction["supported"]) is not bool or isinstance(raw, bool)
                or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0
                or isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or actual <= 0):
            raise ValueError("invalid prediction or observed duration")
        before = clock()
        adjusted, active, margin = guard.predict(prediction)
        cpu_us.append((clock() - before) / 1000)
        scored = kind == "controlled" or row["window"] == "window"
        results.append({"profile": profile, "step_id": step,
                        "window": row.get("window", "measurement"),
                        "scored": scored, "supported": prediction["supported"],
                        "calibrated": active, "actual_ms": actual,
                        "raw_ms": raw, "adjusted_ms": adjusted,
                        "margin_ms": margin,
                        "prior_supported_observations": len(guard.history)})
        guard.observe(prediction, actual)
    scored = [r for r in results if r["scored"]]
    supported = [r for r in scored if r["supported"]]
    active = [r for r in supported if r["calibrated"]]
    return results, {"total_scored_steps": len(scored),
        "supported_steps": len(supported),
        "raw_coverage": len(supported)/len(scored) if scored else None,
        "calibrated_steps": len(active),
        "calibrated_coverage": len(active)/len(scored) if scored else None,
        "cold_start_supported_steps": len(supported)-len(active),
        "unsupported_steps": len(scored)-len(supported),
        "raw_supported": metrics(supported, "raw_ms"),
        "matched_raw_calibrated": metrics(active, "raw_ms"),
        "calibrated": metrics(active, "adjusted_ms"),
        "guard_cpu_p50_us": nearest_rank(cpu_us, .5) if cpu_us else None,
        "guard_cpu_p95_us": nearest_rank(cpu_us, .95) if cpu_us else None}


def run(input_path, kind, out):
    input_path = Path(input_path).resolve()
    verify_artifacts(input_path.parent)
    report = read(input_path.parent / "report.json")
    if kind == "controlled":
        if input_path.name != "predictions.json" or report.get("schema") != "v3-step-evaluation-v1":
            raise ValueError("expected a controlled evaluation predictions.json")
    elif kind == "online":
        if input_path.name != "steps.json" or not report.get("valid_measurement"):
            raise ValueError("expected valid online steps.json")
    else:
        raise ValueError("unknown replay kind")
    rows = read(input_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty Step input")
    details, summary = replay(rows, kind)
    if not summary["total_scored_steps"]:
        raise ValueError("no measured Steps")
    summary.update({"schema": "v3-drift-replay-v1", "kind": kind,
        "input": str(input_path), "input_sha256": sha(input_path),
        "replay_source_sha256": sha(Path(__file__)),
        "predictor_sha256": report["predictor_sha256"],
        "window": WINDOW, "min_observations": MIN_OBSERVATIONS,
        "residual_quantile": QUANTILE,
        "scope": "diagnostic causal replay; only calibrated supported rows are eligible; no SLO claim"})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "summary.json", summary)
    save(out / "replay_rows.json", details)
    seal(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=("controlled", "online"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.kind, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
