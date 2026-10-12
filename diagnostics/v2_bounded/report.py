"""Coverage, paired comparisons and all generation branches, without cherry-picking."""
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import tempfile

from benchmarks.v2_stall.protocol import ROOT, read, save, seal
from benchmarks.v2_stall.metrics import compare_tokens
from .protocol import load_suite
from .batch import resume_decision
from .audit import audit_attempt, VALID


def window_backlog(trace, rows):
    times = [r["time_ns"] for r in rows]
    if times != sorted(times):
        raise ValueError("backlog is not time ordered")
    def at(seconds):
        i = bisect_right(times, round(seconds * 1e9)) - 1
        return rows[i]["backlog_total"] if i >= 0 else 0
    start, end = trace["warmup_s"], trace["warmup_s"] + trace["measurement_s"]
    xs = list(range(max(int(start), int(end) - 30), int(end)))
    ys = [at(x) for x in xs]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denominator if denominator else None
    return {"window_backlog_start": at(start), "window_backlog_end": at(end),
            "last30s_backlog_slope_rps": slope}


def paired_review(a, b, trace):
    reasons = []
    for key in ("packages", "gpus", "model_revision_declared", "torch_cuda_version"):
        if a["environment"].get(key) != b["environment"].get(key):
            reasons.append(f"environment_differs:{key}")
    for key in ("kv_blocks", "kv_payload_bytes", "protocol_sha256"):
        if a["summary"][key] != b["summary"][key]:
            reasons.append(f"runtime_differs:{key}")
    comparisons = compare_tokens(trace, {r.request_id: r.output_token_ids for r in a["records"]},
                                {r.request_id: r.output_token_ids for r in b["records"]})
    ra = {r.request_id: r for r in a["records"]}
    rb = {r.request_id: r for r in b["records"]}
    for row in comparisons:
        row["both_completed"] = ra[row["request_id"]].status == rb[row["request_id"]].status == "completed"
    differences = {}
    for key in ("window_throughput_tok_s", "ttft_p99_ms", "itl_p99_ms", "tpot_p99_ms"):
        x, y = a["summary"][key], b["summary"][key]
        differences[key] = (y / x - 1) * 100 if x is not None and y is not None and x > 0 else None
    return {"valid_for_window_comparison": not reasons, "invalid_reasons": reasons,
            "baseline": a["path"], "candidate": b["path"],
            "paired_change_pct": differences, "generation_comparisons": comparisons,
            "v1_acceptance_passed": False,
            "note": "Unfinished requests retained. Censored output lengths are not automatically numerical divergences; every comparison retained."}


def report_suite(suite, root=ROOT):
    suite = Path(suite).resolve()
    protocol = load_suite(suite, root=root)
    coverage, measurements, selected = [], [], {}
    statuses = Counter()
    for job in protocol["jobs"]:
        decision = resume_decision(suite, job, protocol, root=root)
        coverage.append({"job": job, **decision})
        if not decision["attempts"]:
            statuses["MISSING"] += 1
        for attempt in decision["attempts"]:
            statuses[attempt["status"]] += 1
        # Multiple attempts are not silently reduced to the successful one.
        if len(decision["attempts"]) != 1 or decision["attempts"][0]["status"] not in VALID:
            continue
        path = Path(decision["attempts"][0]["path"])
        data = audit_attempt(suite, job, path, protocol, root=root)
        selected[job["id"]] = data
        trace = read(suite / job["trace"])
        s = data["summary"] | window_backlog(trace, read(path / "backlog.json"))
        measurements.append({"job": job, "path": str(path), "summary": s})
    by_key = {(j["rps"], j["seed"], j["mode"], j["policy"]): j for j in protocol["jobs"]}
    pairs = []
    for job in protocol["jobs"]:
        if job["policy"] == "baseline" or job["id"] not in selected:
            continue
        baseline = by_key[(job["rps"], job["seed"], job["mode"], "baseline")]
        if baseline["id"] not in selected:
            continue
        pairs.append({"job": job, **paired_review(selected[baseline["id"]], selected[job["id"]],
                                                 read(suite / job["trace"]))})
    fields = ("window_throughput_tok_s", "window_completed_rps", "window_goodput_rps",
              "ttft_p99_ms", "itl_p99_ms", "tpot_p99_ms", "cohort_violation_rate",
              "prefill_wait_max_ms", "starvation_request_count", "preemptions", "recomputed_tokens",
              "kv_utilization_sampled_peak", "torch_peak_allocated_bytes", "torch_peak_reserved_bytes",
              "window_backlog_start", "window_backlog_end", "last30s_backlog_slope_rps")
    grouped = defaultdict(list)
    for row in measurements:
        j = row["job"]
        grouped[(j["rps"], j["mode"], j["policy"])].append(row["summary"])
    aggregate = []
    for (rate, mode, policy), rows in sorted(grouped.items()):
        stats = {}
        for key in fields:
            values = [s[key] for s in rows if s[key] is not None]
            stats[key] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                          "std": statistics.stdev(values) if len(values) > 1 else None}
        aggregate.append({"rps": rate, "mode": mode, "policy": policy, "runs": len(rows),
                          "all_configured_seeds_present": len(rows) == len(protocol["seeds"]), "metrics": stats})
    expected_pairs = len(protocol["jobs"]) - sum(j["policy"] == "baseline" for j in protocol["jobs"])
    summary = {"schema": "v2-bounded-report-v1", "expected_jobs": len(protocol["jobs"]),
        "unique_valid_jobs": len(selected), "missing_jobs": statuses["MISSING"],
        "attempt_status_counts": dict(statuses), "expected_pairs": expected_pairs,
        "available_pairs": len(pairs),
        "invalid_pairs": sum(not p["valid_for_window_comparison"] for p in pairs),
        "measurement_matrix_complete": len(selected) == len(protocol["jobs"])
            and len(pairs) == expected_pairs and all(p["valid_for_window_comparison"] for p in pairs),
        "divergent_complete_request_comparisons": sum(not r["exact_match"] and r["both_completed"]
            for p in pairs for r in p["generation_comparisons"]),
        "censored_request_comparisons": sum(not r["both_completed"] for p in pairs for r in p["generation_comparisons"]),
        "v1_acceptance_passed": False, "full_model_numerical_acceptance": "PENDING",
        "scope": "New protocol only. Means of run percentiles, not pooled percentiles. No automatic best-policy selection."}
    out = Path(tempfile.mkdtemp(prefix="report.", dir=suite))
    for name, value in (("coverage.json", coverage), ("runs.json", measurements),
                        ("pairs.json", pairs), ("aggregate.json", aggregate), ("summary.json", summary)):
        save(out / name, value)
    lines = ["# V2有界预热协议：实验报告", "", str(summary), "",
             "均值±样本标准差；P99列是各轮P99的均值，不是合并分位数。n不足冻结seed数的组不是完整评测。",
             "旧200轮未混入。Drain全部完成不代表在线稳定；有效未完成运行保留。生成分歧、删失比较均保留于pairs.json。", "",
             "| req/s | 负载 | 策略 | n | 输出Token/s | P99 TTFT(ms) | P99 ITL(ms) | P99 TPOT(ms) | Goodput(req/s) | 双SLO违约率 |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(stat, percent=False):
        if stat["mean"] is None:
            return "N/A"
        scale = 100 if percent else 1
        val = f"{stat['mean'] * scale:.2f}"
        if stat["std"] is not None:
            val += f"±{stat['std'] * scale:.2f}"
        return val + ("%" if percent else "")
    for row in aggregate:
        m = row["metrics"]
        vals = [fmt(m[k]) for k in ("window_throughput_tok_s", "ttft_p99_ms", "itl_p99_ms", "tpot_p99_ms", "window_goodput_rps")]
        lines.append(f"| {row['rps']} | {row['mode']} | {row['policy']} | {row['runs']} | "
                     + " | ".join(vals + [fmt(m["cohort_violation_rate"], True)]) + " |")
    with (out / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    seal(out)
    return out, summary
