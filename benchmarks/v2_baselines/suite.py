"""Stage1: prepare -> regress -> smoke -> report; capacity search is NOT run."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import tempfile

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, verify_artifacts
from benchmarks.v2_stall.suite import launch
from benchmarks.v2_stall.metrics import compare_tokens
from .protocol import POLICIES, Settings, load, prepare
from .workload import Workload
from .audit import audit_run


def compare_pair(baseline, candidate, trace):
    a, ra, ea = baseline
    b, rb, eb = candidate
    reasons = []
    for label, s in (("baseline", a), ("candidate", b)):
        if not s["valid_measurement"]:
            reasons.append(f"{label}:invalid_measurement")
        if not s["all_completed"]:
            reasons.append(f"{label}:not_all_completed_in_smoke")
        if s["diagnostic_only"]:
            reasons.append("diagnostic_not_performance")
    if a["policy"] != "baseline" or b["policy"] == "baseline" or any(a[k] != b[k] for k in ("mode", "seed")):
        reasons.append("run_identity_differs")
    for key in ("packages", "gpus", "model_revision_declared", "torch_cuda_version"):
        if ea.get(key) != eb.get(key):
            reasons.append(f"environment_differs:{key}")
    if any(a[k] != b[k] for k in ("kv_blocks", "kv_payload_bytes")):
        reasons.append("KV_capacity_differs")
    branches = compare_tokens(trace, {r.request_id: r.output_token_ids for r in ra},
                              {r.request_id: r.output_token_ids for r in rb})
    return {"baseline": a, "candidate": b, "valid_for_comparison": not reasons,
            "invalid_reasons": reasons, "generation_comparisons": branches,
            "generation_review_pending": any(not r["exact_match"] for r in branches),
            "v1_acceptance_passed": False}


def reuse_or_launch(command, out, log):
    if out.exists():
        verify_artifacts(out)
        print("Preserving existing attempt:", out, flush=True)
        return
    launch(command, log)


def regression(folder, protocol):
    directory = folder/"regression"
    if (directory/"artifact_hashes.json").exists():
        verify_artifacts(directory)
        if not read(directory/"report.json")["state_attention_passed"]:
            raise SystemExit(2)
        return
    directory.mkdir(exist_ok=True)
    checks = []
    for pattern in ("test_v1_state.py", "test_v2_baselines.py", "test_v1_attention_gpu.py", "test_v1_reference_gpu.py"):
        log = directory/f"{pattern}.log"
        if log.exists():
            raise ValueError("incomplete regression exists; retain it, do not silently retry")
        code = launch([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], log)
        checks.append({"test": pattern, "exit": code})
    cases = []
    for policy in POLICIES[1:]:
        for case in ("mixed_prefix", "preempt"):
            out = directory/f"{policy}-{case}"
            command = [sys.executable, "-B", "-m", "benchmarks.v2_baselines.regress", "--suite", str(folder),
                       "--policy", policy, "--case", case, "--out", str(out)]
            reuse_or_launch(command, out, directory/f"{policy}-{case}.log")
            passed = False
            if (out/"report.json").exists():
                verify_artifacts(out)
                identity = read(out/"identity.json")
                if identity != {"protocol_sha256": sha(folder/"protocol.json"), "policy": policy,
                                "case": case, "diagnostic_only": True}:
                    raise ValueError("regression identity mismatch")
                passed = read(out/"report.json")["state_passed"] and not (out/"error.json").exists()
            cases.append({"policy": policy, "case": case, "state_passed": passed})
    report = {"protocol_sha256": sha(folder/"protocol.json"), "checks": checks, "cases": cases,
              "state_attention_passed": all(r["exit"] == 0 for r in checks) and all(r["state_passed"] for r in cases),
              "v1_acceptance_passed": False, "full_model_numerical_acceptance": "PENDING_OR_PREVIOUSLY_FAILED",
              "note": "New policy state checks and local Attention tests only. Does not supersede old full-model numerical failures."}
    save(directory/"report.json", report)
    seal(directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["state_attention_passed"]:
        raise SystemExit(2)


def report(folder, protocol):
    seed = protocol["settings"]["seed"]
    pairs, missing = [], []
    for mode in ("poisson", "burst", "long_short"):
        trace = read(folder/f"traces/{mode}-seed-{seed}.json")
        base = folder/"runs"/mode
        for policy in POLICIES[1:]:
            paths = [base/"baseline", base/policy]
            if any(not (p/"summary.json").exists() for p in paths):
                missing.append({"mode": mode, "policy": policy, "reason": "missing_or_failed_run"})
                continue
            a, b = [audit_run(folder, p) for p in paths]
            if a[0]["policy"] != "baseline" or b[0]["policy"] != policy or any(
                    data[0]["mode"] != mode or data[0]["seed"] != seed for data in (a, b)):
                raise ValueError("paired path identity mismatch")
            pairs.append({"mode": mode, "policy": policy, **compare_pair(a, b, trace)})
    regression_path = folder/"regression"
    reg_passed = False
    if (regression_path/"report.json").exists():
        verify_artifacts(regression_path)
        r = read(regression_path/"report.json")
        reg_passed = r["state_attention_passed"] and r["protocol_sha256"] == sha(folder/"protocol.json")
    pending = sum(not r["exact_match"] for p in pairs for r in p["generation_comparisons"])
    valid = not missing and len(pairs) == 18 and all(p["valid_for_comparison"] for p in pairs)
    status = {"stage": "V2-stage1-smoke", "paired_comparisons": len(pairs), "expected_pairs": 18,
              "missing": missing, "all_pairs_valid": valid, "state_attention_passed": reg_passed,
              "generation_divergent_request_comparisons": pending,
              "ready_for_exploratory_capacity_scan": valid and reg_passed and pending == 0,
              "v1_acceptance_passed": False,
              "note": "Single seed only; no policy selection or formal speedup claim. Any branches require review, not silent filtering."}
    out = Path(tempfile.mkdtemp(prefix="report.", dir=folder))
    save(out/"pairs.json", pairs)
    save(out/"summary.json", status)
    lines = ["# V2第1批：固定策略与在线负载烟雾测试", "",
             "单seed、小规模、单卡Eager；不是正式性能/容量结果，V1完整模型数值验收仍未通过。", "",
             "| 负载 | 策略 | 配对有效 | P99 TTFT(ms) | P99 ITL(ms) | 窗口Token/s | Prefill最长等待(ms) | 饥饿请求数 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    def value(x):
        return "N/A" if x is None else f"{x:.3f}"
    for pair in pairs:
        s = pair["candidate"]
        lines.append(f"| {pair['mode']} | {pair['policy']} | {pair['valid_for_comparison']} | "
                     f"{value(s['ttft_p99_ms'])} | {value(s['itl_p99_ms'])} | {value(s['window_throughput_tok_s'])} | "
                     f"{value(s['prefill_wait_max_ms'])} | {s['starvation_request_count']} |")
    lines += ["", "```json", json.dumps(status, ensure_ascii=False, indent=2), "```", "",
              "原始调度及所有失败/未完成数据保留于pairs.json/runs；TPOT等默认时延指标针对窗口到达cohort。",
              "all_requests_tpot_*另外统计所有完成请求（包括Warmup到达）的完整生成；不等于窗口平均ITL。",
              "Prefill等待/抢占/重算包括Drain；KV水位是窗口Step的postprocess采样，不是瞬时分配峰值。",
              "没有分歧只说明这些Trace上相同；分歧记录包含首次分叉与共同历史，不能自动豁免。"]
    with (out/"report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    seal(out)
    print("Stage1 report:", out, flush=True)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def smoke(folder, protocol):
    verify_artifacts(folder/"regression")
    reg = read(folder/"regression/report.json")
    if not reg["state_attention_passed"] or reg["protocol_sha256"] != sha(folder/"protocol.json"):
        raise ValueError("complete state/Attention regression before smoke")
    seed = protocol["settings"]["seed"]
    for index, mode in enumerate(("poisson", "burst", "long_short")):
        order = list(POLICIES)
        random.Random(seed+index*1000).shuffle(order)
        base = folder/"runs"/mode
        base.mkdir(parents=True, exist_ok=True)
        for policy in order:
            out = base/policy
            command = [sys.executable, "-B", "-m", "benchmarks.v2_baselines.run", "--suite", str(folder),
                       "--mode", mode, "--policy", policy, "--out", str(out)]
            reuse_or_launch(command, out, base/f"{policy}.log")
    status = report(folder, protocol)
    if not status["ready_for_exploratory_capacity_scan"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "regress", "smoke", "report"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--parent-suite", default=str(ROOT/"results/raw/v2-stall.DFlCEw"))
    from benchmarks.defaults import add_model_arguments, resolve_model_arguments
    add_model_arguments(parser)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rps", type=float, default=2)
    parser.add_argument("--warmup-s", type=float, default=2)
    parser.add_argument("--measurement-s", type=float, default=8)
    parser.add_argument("--short-tokens", type=int, default=256)
    parser.add_argument("--long-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-prefill-wait-ms", type=float, default=1000)
    parser.add_argument("--starvation-ms", type=float, default=1000)
    args = resolve_model_arguments(parser.parse_args(), parser)
    folder = Path(args.suite).resolve()
    if args.command == "prepare":
        from transformers import AutoTokenizer
        from benchmarks.v1.fixtures import language_prompt
        settings = Settings(seed=args.seed, max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len, token_budget=args.token_budget,
            max_prefill_wait_ms=args.max_prefill_wait_ms, starvation_ms=args.starvation_ms)
        workload = Workload(warmup_s=args.warmup_s, measurement_s=args.measurement_s, rps=args.rps,
            short_tokens=args.short_tokens, long_tokens=args.long_tokens, output_tokens=args.output_tokens)
        settings.validate(workload)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prepare(folder, settings, workload, args.model, args.model_revision,
                lambda n, i: language_prompt(tokenizer, n, args.seed*10000+i), parent=args.parent_suite)
        print("Frozen V2 stage1 suite:", folder)
        return
    protocol = load(folder)
    if args.command == "verify":
        print("V2 stage1 sources/model/traces verified; V1 numerical acceptance still pending.")
    elif args.command == "regress":
        regression(folder, protocol)
    elif args.command == "smoke":
        smoke(folder, protocol)
    else:
        status = report(folder, protocol)
        if not status["ready_for_exploratory_capacity_scan"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
