"""Prepare -> smoke -> tune -> regress -> evaluate -> diagnose -> report.

All writes are append-only/new paths. Fixed-clock injection is exploratory while
the inherited V1 numerical acceptance remains unresolved.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from .metrics import aggregate_pairs, analyze_run, choose_chunk, compare_tokens, records_from_rows
from .protocol import (ROOT, Experiment, build_trace, config_dict, load_suite, read,
                       save, seal, sha, sources, verify_artifacts)


def parent_record(parent, current, root):
    if parent is None:
        return None
    parent = Path(parent)
    for name, digest in read(parent / "manifest.json").items():
        if Path(name).name != name or sha(parent / name) != digest:
            raise ValueError("parent V1 archive changed")
    inherited = read(parent / "sources.json")
    changed = [name for name, digest in inherited.items() if current.get(name) != digest]
    if changed:
        raise ValueError(f"inherited V1 sources changed; requires explicit new provenance: {changed}")
    for name, digest in read(parent / "evidence.json").items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(root / name) != digest:
            raise ValueError("parent V1 evidence changed")
    return {"path": str(parent.resolve()), "policy_sha256": sha(parent / "policy.json"),
            "manifest_sha256": sha(parent / "manifest.json"),
            "inherited_sources_unchanged": True, "added_sources": sorted(set(current)-set(inherited)),
            "note": "Old exact-source-set verifier will reject newly added V2 files. No old rule or evidence was changed; this is a NEW V2 experiment freeze, not V1 acceptance."}


def prepare_suite(folder, experiment, model, revision, token_factory, parent=None, root=ROOT):
    experiment.validate()
    model, folder = Path(model).resolve(), Path(folder).resolve()
    if not revision.strip():
        raise ValueError("model revision required")
    model_config = read(model / "config.json")
    if experiment.max_model_len > model_config["max_position_embeddings"]:
        raise ValueError("requested context exceeds model config")
    current = sources(root)
    inherited = parent_record(parent, current, root)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("prepare requires a new or empty suite; never overwrite old experiments")
    hashes = {}
    prompt_hashes = {}
    for split in ("tune", "eval"):
        (folder / split).mkdir()
        prompt_hashes[split] = set()
        for seed in getattr(experiment, f"{split}_seeds"):
            trace = build_trace(experiment, seed, split, token_factory)
            path = folder / split / f"seed-{seed}.json"
            save(path, trace)
            hashes[str(path.relative_to(folder))] = sha(path)
            for spec in trace["requests"]:
                prompt_hashes[split].add(json.dumps(spec["prompt_token_ids"]))
    if prompt_hashes["tune"] & prompt_hashes["eval"]:
        raise ValueError("actual prompt tokens overlap between tuning and evaluation")
    metadata_names = ("config.json", "tokenizer_config.json", "tokenizer.json",
                      "generation_config.json", "model.safetensors.index.json")
    protocol = {"protocol_id": "v2-eager-stall-v1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": {"path": str(model), "revision": revision},
                "model_metadata_sha256": {n: sha(model / n) for n in metadata_names if (model / n).is_file()},
                "experiment": config_dict(experiment), "source_sha256": current,
                "parent_v1": inherited, "trace_sha256": hashes,
                "v1_acceptance_passed": False, "scope": "TP=1 Eager BF16; fixed population, synthetic language fixtures; exploratory timing",
                "rules": {
                    "stall": "Maximum full adjacent-token interval over original Decode requests overlapping the common planned [inject, window_end) window; do not clip or censor intervals.",
                    "readiness": "At the PLANNED injection time every Decode request has stable_intervals completed ITLs, CV<=stable_cv_max, fresh tokens and is alive; all remain alive through window end.",
                    "timing": "CPU-visible token clocks, single-thread Step-boundary submission; report enqueue lag. Fixed-count input cohort, NOT Poisson or saturation capacity.",
                    "warmup": "16-token representative Prefill/Mixed/Decode warmup, reset cold Prefix metadata, no weight/KV tensor reallocation.",
                    "selection": "Tuning only: minimize mean per-run max Decode ITL, mean throughput loss within frozen bound, long TTFT<=SLO in every run; tie smaller chunk. No feasible policy -> retain negative result, no eval selection.",
                    "invalid": "Retain every attempt; any missing/invalid pair blocks that chunk's aggregate. No silent retries, later injection, truncation, unfinished filtering or post-hoc threshold changes.",
                    "diagnosis": "Every output branch retained. First-divergence same-history Prefill replay is a diagnostic, not reproduction of the original Decode batch or a V1 waiver.",
                    "state": "Existing mixed/prefix/preempt candidate+reference checks and a separate full-trace per-Step state scan for selected chunk. Diagnostic times excluded from performance.",
                    "order": "One fresh process per policy/seed; deterministic seed-shuffled policy order. Paired independent seeds; sample standard deviation (n-1)."}}
    save(folder / "protocol.json", protocol)
    seal(folder)
    return protocol


def launch(command, log):
    """Sequential child, full stdout/stderr retained; nonzero is never hidden."""
    log = Path(log)
    print(f"Running: {' '.join(command)}\nLog: {log}", flush=True)
    with log.open("x") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    print(f"exit={code} log={log}", flush=True)
    return code


def launch_run(folder, split, seed, chunk, base=None, verify_state=False):
    policy = "baseline" if chunk is None else f"chunk-{chunk}"
    base = Path(base) if base else folder / "runs" / split
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"seed-{seed}-{policy}"
    if out.exists():
        # Resume uses existing data, never re-runs or replaces an invalid attempt.
        verify_artifacts(out)
        print(f"Preserving existing attempt: {out}", flush=True)
        return out
    command = [sys.executable, "-B", "-m", "benchmarks.v2_stall.run", "--suite", str(folder),
               "--split", split, "--seed", str(seed), "--policy", "baseline" if chunk is None else "chunk",
               "--out", str(out)]
    if chunk is not None:
        command += ["--chunk", str(chunk)]
    if verify_state:
        command.append("--verify-state")
    launch(command, out.with_suffix(".log"))
    return out


def audit_run(path, protocol, experiment, trace, *, diagnostic=False):
    verify_artifacts(path)
    summary, config = read(path / "summary.json"), read(path / "run_config.json")
    if config["protocol_sha256"] != sha(Path(config["suite"]) / "protocol.json"):
        raise ValueError("run protocol identity changed")
    if read(path / "trace.json") != trace or summary["diagnostic_only"] != diagnostic:
        raise ValueError("trace differs or diagnostic timings used as performance")
    if config["tp"] != 1 or config["graph"] or config["observation"] != "full":
        raise ValueError("incorrect execution/observation mode")
    if any(config[key] != trace[key] or summary[key] != trace[key] for key in ("seed", "split")) or any(
            config[key] != summary[key] for key in ("policy", "chunk")):
        raise ValueError("run identity differs from frozen trace or summary")
    if config["policy"] not in ("baseline", "chunk") or (
            config["policy"] == "baseline" and config["chunk"] is not None) or (
            config["policy"] == "chunk" and config["chunk"] not in experiment.chunks):
        raise ValueError("run identity has an invalid policy/chunk")
    rows = read(path / "requests.json")
    recalculated = analyze_run(trace, records_from_rows(rows), experiment,
                               read(path / "error.json")["message"] if (path / "error.json").exists() else None)
    # End-state/source validation adds invalid reasons after timestamp analysis.
    extras = set(summary["invalid_reasons"])-set(recalculated["invalid_reasons"])
    if extras - {"source_changed_during_run", "end_state_check_failed"}:
        raise ValueError("unrecognized invalidity reason")
    if summary["source_unchanged"] is not True:
        raise ValueError("source changed during run")
    for key, value in recalculated.items():
        if key in ("valid_experiment", "invalid_reasons"):
            continue
        if summary[key] != value:
            raise ValueError(f"summary audit mismatch: {key}")
    if summary["valid_experiment"] != (recalculated["valid_experiment"] and not extras):
        raise ValueError("invalid run falsely marked valid")
    env = read(path / "environment.json")
    if env["source_sha256"] != {k: v for k, v in protocol["source_sha256"].items() if k.startswith("nanovllm/")}:
        raise ValueError("model execution source changed")
    if env["model_metadata_sha256"] != protocol["model_metadata_sha256"]:
        raise ValueError("model metadata changed")
    return summary, rows, env


def pair_reports(folder, split, chunk, seeds, protocol, experiment, base=None):
    pairs = []
    base = Path(base) if base else folder / "runs" / split
    for seed in seeds:
        trace = read(folder / split / f"seed-{seed}.json")
        paths = [base / f"seed-{seed}-baseline", base / f"seed-{seed}-chunk-{chunk}"]
        pair = {"seed": seed, "paths": list(map(str, paths)), "divergences": []}
        loaded = []
        for label, path in zip(("baseline", "candidate"), paths):
            if not (path / "summary.json").exists():
                pair[label] = {"valid_experiment": False, "invalid_reasons": ["missing_or_failed_run"]}
                loaded.append(None)
            else:
                data = audit_run(path, protocol, experiment, trace)
                pair[label] = data[0]
                loaded.append(data)
        if all(loaded):
            a, b = loaded
            if a[0]["policy"] != "baseline" or b[0]["policy"] != "chunk" or b[0]["chunk"] != chunk:
                raise ValueError("paired policy identities differ from requested comparison")
            for key in ("packages", "gpus", "model_revision_declared"):
                if a[2][key] != b[2][key]:
                    raise ValueError(f"unfair paired environment difference: {key}")
            if a[0]["kv_blocks"] != b[0]["kv_blocks"]:
                pair["candidate"] = {**pair["candidate"], "valid_experiment": False,
                                     "invalid_reasons": pair["candidate"]["invalid_reasons"]+["KV_capacity_differs"]}
            pair["divergences"] = compare_tokens(trace,
                {r["request_id"]: r["output_token_ids"] for r in a[1]},
                {r["request_id"]: r["output_token_ids"] for r in b[1]})
        pairs.append(pair)
    return pairs


def selected(folder):
    directory = folder / "selection"
    verify_artifacts(directory)
    selection = read(directory / "selection.json")
    if selection["protocol_sha256"] != sha(folder / "protocol.json") or selection["selection_split"] != "tune":
        raise ValueError("invalid or changed selection")
    if selection["chunk"] is None:
        raise ValueError("no feasible tuning candidate; retain negative result or start a NEW protocol")
    return selection["chunk"]


def tune(folder, protocol, experiment):
    if (folder / "selection").exists():
        verify_artifacts(folder / "selection")
        print("Selection already frozen; not re-tuning.")
        return
    for seed in experiment.tune_seeds:
        order = [None, *experiment.chunks]
        random.Random(seed).shuffle(order)
        for chunk in order:
            launch_run(folder, "tune", seed, chunk)
    reports, pairs = {}, {}
    for chunk in experiment.chunks:
        data = pair_reports(folder, "tune", chunk, experiment.tune_seeds, protocol, experiment)
        pairs[str(chunk)] = data
        reports[chunk] = aggregate_pairs(data, experiment.tune_seeds)
    directory = folder / "selection"
    directory.mkdir()
    save(directory / "tuning_pairs.json", pairs)
    save(directory / "tuning_aggregates.json", reports)
    save(directory / "selection.json", {**choose_chunk(reports, experiment),
                                        "protocol_sha256": sha(folder / "protocol.json")})
    seal(directory)
    print(json.dumps(read(directory / "selection.json"), ensure_ascii=False, indent=2))


def evaluate(folder, protocol, experiment):
    chunk = selected(folder)
    verify_artifacts(folder / "regression")
    regression = read(folder / "regression/report.json")
    if not regression["state_checks_passed"] or regression["chunk"] != chunk or regression["protocol_sha256"] != sha(folder / "protocol.json"):
        raise ValueError("selected configuration has not passed state regression")
    print("V1 numerical acceptance is still pending; evaluation results are exploratory.", flush=True)
    for seed in experiment.eval_seeds:
        order = [None, chunk]
        random.Random(seed).shuffle(order)
        for policy in order:
            launch_run(folder, "eval", seed, policy)
    render_report(folder, protocol, experiment)


def render_report(folder, protocol, experiment):
    chunk = selected(folder)
    pairs = pair_reports(folder, "eval", chunk, experiment.eval_seeds, protocol, experiment)
    aggregate = aggregate_pairs(pairs, experiment.eval_seeds)
    output = Path(tempfile.mkdtemp(prefix="report.", dir=folder))
    save(output / "pairs.json", pairs)
    save(output / "aggregate.json", aggregate)
    text = ["# V2单卡Eager长Prompt注入对照", "", f"固定Chunk：{chunk}；独立评测seed：{list(experiment.eval_seeds)}。",
            "", "V1数值验收未闭环；以下为探索性性能测量，不是整体V1/V2验收。", ""]
    if aggregate["valid_for_performance_comparison"]:
        text += ["| 指标 | 原始调度（均值±样本标准差） | 固定Chunk（均值±样本标准差） |",
                 "|---|---:|---:|"]
        for key in ("decode_stall_max_ms", "decode_itl_p99_ms", "long_ttft_ms", "window_throughput_tok_s"):
            a, b = aggregate["baseline"][key], aggregate["candidate"][key]
            text.append(f"| {key} | {a['mean']:.3f} ± {a['std']:.3f} | {b['mean']:.3f} ± {b['std']:.3f} |")
    else:
        text += [f"无效/缺失配对seed：{aggregate['invalid_seeds']}。不删除失败轮次，不生成收益结论。"]
    branches = sum(not r["exact_match"] for p in pairs for r in p["divergences"])
    text += ["", f"非完全一致的请求执行数：{branches}；首次分叉与同历史输入保存在pairs.json。",
             "存在分歧时必须执行diagnose并审阅；分叉后逐位置Token匹配率不是准确率。",
             "Stall统计完整的跨窗口相邻Token间隔；吞吐为共同窗口内输出Token/s，不是最大服务容量。",
             "峰值显存字段为PyTorch allocator的allocated/reserved，包含Drain，不等于NVML总显存。"]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(text)+"\n")
    seal(output)
    print(f"Report: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "smoke", "tune", "regress", "evaluate", "diagnose", "report"])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--parent-protocol", default=str(ROOT / "results/raw/v1-protocol.2_c5wbrp"))
    parser.add_argument("--decode-count", type=int, default=8)
    parser.add_argument("--inject-s", type=float, default=2)
    parser.add_argument("--window-s", type=float, default=2)
    parser.add_argument("--long-prompt-tokens", type=int, default=3072)
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    from benchmarks.defaults import add_model_arguments, resolve_model_arguments
    add_model_arguments(parser)
    args = resolve_model_arguments(parser.parse_args(), parser)
    folder = Path(args.suite).resolve()
    if args.command == "prepare":
        from transformers import AutoTokenizer
        from benchmarks.v1.fixtures import language_prompt
        experiment = Experiment(decode_count=args.decode_count, inject_s=args.inject_s, window_s=args.window_s,
            long_prompt_tokens=args.long_prompt_tokens, decode_output_tokens=args.decode_output_tokens,
            max_model_len=args.max_model_len, token_budget=args.token_budget, chunks=tuple(args.chunks))
        experiment.validate()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prepare_suite(folder, experiment, args.model, args.model_revision,
            lambda n, seed, index: language_prompt(tokenizer, n, seed*100+index),
            parent=Path(args.parent_protocol))
        print(f"Frozen V2 suite: {folder}")
        return
    protocol, experiment = load_suite(folder)
    if args.command == "verify":
        print("V2 protocol, traces and source set verified; V1 acceptance remains pending.")
    elif args.command == "smoke":
        seed, chunk = experiment.tune_seeds[0], experiment.chunks[0]
        for policy in (None, chunk):
            launch_run(folder, "tune", seed, policy, base=folder / "smoke")
        data = pair_reports(folder, "tune", chunk, [seed], protocol, experiment, base=folder / "smoke")
        print("SMOKE ONLY", json.dumps(aggregate_pairs(data, [seed]), ensure_ascii=False, indent=2))
        if not all(p[k]["valid_experiment"] for p in data for k in ("baseline", "candidate")):
            raise SystemExit(2)
    elif args.command == "tune":
        tune(folder, protocol, experiment)
    elif args.command == "evaluate":
        evaluate(folder, protocol, experiment)
    elif args.command == "report":
        render_report(folder, protocol, experiment)
    else:
        from .validation import regress, diagnose
        {"regress": regress, "diagnose": diagnose}[args.command](folder, protocol, experiment)


if __name__ == "__main__":
    main()
