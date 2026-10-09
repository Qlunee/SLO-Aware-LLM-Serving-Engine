"""从原始请求记录重算指标，CPU即可审计实验产物。"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

from .metrics import RequestRecord, request_metrics, summarize
from .workload import load_trace, trace_sha256


def audit(directory):
    directory = Path(directory)
    env = json.loads((directory / "environment.json").read_text())
    saved = json.loads((directory / "summary.json").read_text())
    trace = directory / "trace.jsonl"
    meta, specs = load_trace(trace)
    if trace_sha256(trace) != env["trace_sha256"]:
        raise ValueError("Trace摘要不一致")
    if not saved["valid_run"] or saved.get("empty_trace") or not specs:
        raise ValueError("无效运行或空Trace不能作为基准")
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != len(specs) or {r["request_id"] for r in rows} != {s.request_id for s in specs}:
        raise ValueError("请求记录重复、缺失或多余")
    observed = saved["observation"] != "native"
    names = {f.name for f in fields(RequestRecord)}
    spec_map = {s.request_id: s for s in specs}
    records = []
    for row in rows:
        spec = spec_map[row["request_id"]]
        for key, expected in {
            "arrival_ns": spec.arrival_ns, "prompt_tokens": len(spec.prompt_token_ids),
            "max_output_tokens": spec.max_output_tokens,
            "ttft_slo_ms": spec.ttft_slo_ms, "itl_slo_ms": spec.itl_slo_ms,
        }.items():
            if row[key] != expected:
                raise ValueError(f"请求字段被修改：{key}")
        data = {k: v for k, v in row.items() if k in names}
        data["token_times_ns"] = data["token_times_ns"] or []
        record = RequestRecord(**data)
        if record.status not in {"completed", "rejected", "unfinished", "timeout", "failed"}:
            raise ValueError("未知或未收敛的请求状态")
        if record.status == "completed" and len(record.output_token_ids) != spec.max_output_tokens:
            raise ValueError("ignore_eos完成请求输出长度不匹配")
        if observed and len(record.token_times_ns) != len(record.output_token_ids):
            raise ValueError("Token数与可见时间数不一致")
        if observed and request_metrics(record) != row["metrics"]:
            raise ValueError("请求指标与原始时间戳不一致")
        records.append(record)
    recalculated = summarize(records, round(meta["warmup_s"] * 1e9),
                             round((meta["warmup_s"] + meta["measurement_s"]) * 1e9), observed)
    for key, value in recalculated.items():
        actual = saved[key]
        equal = math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-10) if isinstance(value, float) and isinstance(actual, (int, float)) else value == actual
        if not equal:
            raise ValueError(f"汇总指标不一致：{key}: {actual!r} != {value!r}")
    return {"passed": True, "requests": len(records), "observation": saved["observation"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    print(json.dumps(audit(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
