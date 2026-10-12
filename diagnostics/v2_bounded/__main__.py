import argparse
import json
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments
from benchmarks.v2_stall.protocol import ROOT
from benchmarks.v2_baselines.protocol import Settings, POLICIES
from benchmarks.v2_baselines.workload import Workload
from .protocol import prepare, load_suite
from .batch import run_batch, resume_decision
from .run import run_job
from .report import report_suite
from .audit import VALID


def main():
    parser = argparse.ArgumentParser(description="V2有界预热、新协议冻结与可恢复批处理")
    parser.add_argument("command", choices=("prepare", "verify", "plan", "smoke", "batch", "report", "run", "retry"))
    parser.add_argument("--suite", required=True)
    add_model_arguments(parser)
    parser.add_argument("--parent-suite", default=str(ROOT / "results/raw/v2-baselines.6NXW8a"))
    parser.add_argument("--rates", type=float, nargs="+", default=[6, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 72, 73])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["baseline", "chunk-256", "chunk-512"])
    parser.add_argument("--warmup-s", type=float, default=4)
    parser.add_argument("--measurement-s", type=float, default=60)
    parser.add_argument("--short-tokens", type=int, default=256)
    parser.add_argument("--long-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--drain-s", type=float, default=60)
    parser.add_argument("--max-prefill-wait-ms", type=float, default=1000)
    parser.add_argument("--starvation-ms", type=float, default=1000)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--job")
    parser.add_argument("--out")
    parser.add_argument("--reason")
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    if args.command == "prepare":
        args = resolve_model_arguments(args, parser)
        from transformers import AutoTokenizer
        from benchmarks.v1.fixtures import language_prompt
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        settings = Settings(max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
            token_budget=args.token_budget, drain_s=args.drain_s,
            max_prefill_wait_ms=args.max_prefill_wait_ms, starvation_ms=args.starvation_ms)
        workload = Workload(warmup_s=args.warmup_s, measurement_s=args.measurement_s,
            short_tokens=args.short_tokens, long_tokens=args.long_tokens, output_tokens=args.output_tokens)
        p = prepare(suite, args.model, args.model_revision,
            lambda n, seed, i: language_prompt(tokenizer, n, seed * 10000 + i),
            rates=args.rates, seeds=args.seeds, policies=args.policies,
            settings=settings, workload=workload, parent=args.parent_suite)
        print("Frozen bounded suite:", suite, "jobs:", len(p["jobs"]))
        return 0
    p = load_suite(suite)
    if args.command == "plan":
        print(json.dumps(p["jobs"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify":
        for job in p["jobs"]:
            resume_decision(suite, job, p)
        print("协议、工具、源码、模型元数据、Trace及已有运行审计通过；不代表矩阵完整或数值验收通过。")
        return 0
    if args.command == "run":
        if not args.job or not args.out:
            parser.error("run requires --job and --out")
        return run_job(suite, args.job, args.out)
    if args.command in ("batch", "smoke", "retry"):
        ids = None
        if args.command == "smoke":
            ids = [j["id"] for j in p["jobs"] if j["rps"] == p["rates"][0]
                   and j["seed"] == p["seeds"][0] and j["mode"] == "poisson"]
        if args.command == "retry" and (not args.job or not args.reason):
            parser.error("retry requires --job and --reason")
        ledger = run_batch(suite, job_ids=ids, max_new=args.max_new,
            retry_job=args.job if args.command == "retry" else None, reason=args.reason)
        if args.command == "smoke" or args.max_new:
            ok = all(len(j["attempts"]) == 1 and j["attempts"][0]["status"] in VALID
                     for j in ledger["jobs"])
            print("选定子集状态：", "有效" if ok else "存在失败/中断，已保留", "；不是完整矩阵验收。")
            return 0 if ok else 2
    out, summary = report_suite(suite)
    print("Report:", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("数据齐全不等于数值正确性通过；请查看pairs.json中的全部生成比较。")
    return 0 if summary["measurement_matrix_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
