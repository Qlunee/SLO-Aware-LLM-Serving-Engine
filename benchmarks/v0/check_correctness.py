## 读取已经完成的零间隔在线实验，再用原始generate()运行相同请求，比较输出。
from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path

from .environment import model_metadata_hashes, source_hashes
from .workload import load_trace, trace_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    observed_dir = Path(args.observed_dir)
    environment = json.loads(
        (observed_dir / "environment.json").read_text()
    )
    summary = json.loads(
        (observed_dir / "summary.json").read_text()
    )
    run_config = environment["run_config"]
    trace = observed_dir / "trace.jsonl"
    _, specs = load_trace(trace)

    if trace_sha256(trace) != environment["trace_sha256"]:
        raise ValueError("实验Trace被修改")
    root = Path(__file__).resolve().parents[2]
    if source_hashes(root) != environment["source_sha256"]:
        raise ValueError("引擎源码已变化，不能与旧结果直接比较")
    if model_metadata_hashes(Path(run_config["model"])) != environment["model_metadata_sha256"]:
        raise ValueError("模型元数据已变化")

    if not specs or any(s.arrival_ns != 0 for s in specs):
        raise ValueError("正确性对照必须使用非空零间隔Trace")
    if not summary["valid_run"]:
        raise ValueError("在线实验无效")
    if run_config["sampler"] != "greedy":
        raise ValueError("正确性对照要求greedy")

    rows = [json.loads(line) for line in (observed_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("在线记录包含重复请求")
    observed = {
        row["request_id"]: row
        for row in (
            json.loads(line)
            for line in (observed_dir / "requests.jsonl")
            .read_text().splitlines()
            if line.strip()
        )
    }
    if set(observed) != {s.request_id for s in specs}:
        raise ValueError("在线记录存在请求丢失或多余请求")
    if any(r["status"] != "completed" for r in observed.values()):
        raise ValueError("在线请求没有全部完成")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    from .preflight import check_environment
    checks = check_environment(run_config["model"], run_config["tp"])
    if checks["errors"]:
        raise SystemExit(json.dumps(checks, indent=2, ensure_ascii=False))
    import torch
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams
    from .observed_engine import GreedySampler

    torch.manual_seed(run_config["seed"])
    torch.cuda.manual_seed_all(run_config["seed"])

    engine = LLMEngine(
        run_config["model"],
        **environment["engine_kwargs"],
    )
    # 原引擎exit非幂等，这里由finally统一负责退出。
    atexit.unregister(engine.exit)
    engine.model_runner.sampler = GreedySampler()

    try:
        reference = engine.generate(
            [s.prompt_token_ids for s in specs],
            [
                SamplingParams(
                    temperature=1.0,
                    max_tokens=s.max_output_tokens,
                    ignore_eos=True,
                )
                for s in specs
            ],
            use_tqdm=False,
        )
    finally:
        engine.exit()

    cases = []
    matched_tokens = 0
    total_tokens = 0

    if len(reference) != len(specs):
        raise ValueError("离线输出请求数不匹配")

    for spec, expected in zip(specs, reference):
        actual_ids = observed[spec.request_id]["output_token_ids"]
        expected_ids = expected["token_ids"]
        matched_tokens += sum(
            a == b for a, b in zip(actual_ids, expected_ids)
        )
        total_tokens += max(len(actual_ids), len(expected_ids))
        cases.append({
            "request_id": spec.request_id,
            "match": actual_ids == expected_ids and len(expected_ids) == spec.max_output_tokens,
            "reference_tokens": expected_ids,
            "observed_tokens": actual_ids,
        })

    passed = all(case["match"] for case in cases)
    report = {
        "trace_sha256": trace_sha256(trace),
        "passed": passed,
        "matched_requests": sum(case["match"] for case in cases),
        "total_requests": len(cases),
        "token_match_rate": (
            matched_tokens / total_tokens if total_tokens else None
        ),
        "cases": cases,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({
        key: value for key, value in report.items() if key != "cases"
    }, indent=2))

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
