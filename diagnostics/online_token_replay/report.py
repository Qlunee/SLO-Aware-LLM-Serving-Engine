"""Read-only raw-vector auditing before append-only diagnostic reports."""
from .tensors import compare_logits, validate_context
from pathlib import Path
from itertools import combinations
import tempfile
import torch

from benchmarks.v2_stall.protocol import read, save, seal, sha, verify_artifacts
from benchmarks.v2_baselines.protocol import load
from .core import tool_hashes


def compare_captured_vectors(left, right):
    if left["request_id"] != right["request_id"]:
        raise ValueError("cannot compare different requests")
    same = left["history"] == right["history"]
    return {"request_id": left["request_id"], "same_history": same,
            "status": "COMMON_HISTORY_DIAGNOSTIC_ONLY" if same else "HISTORY_DIFFERS_NOT_A_COMMON_HISTORY_COMPARISON",
            "logits": compare_logits(left["logits"], right["logits"]) if same else None,
            "v1_acceptance_passed": False}


def audit_replay(capture, replay):
    verify_artifacts(capture)
    verify_artifacts(replay)
    identity = read(capture/"identity.json")
    ri = read(replay/"identity.json")
    if (ri["source"] != str(capture.resolve()) or ri["source_manifest_sha256"] != sha(capture/"artifact_hashes.json")
            or ri["protocol_sha256"] != identity["protocol_sha256"]
            or ri["tool_sha256"] != identity["tool_sha256"] or tool_hashes() != identity["tool_sha256"]):
        raise ValueError("replay provenance differs from capture")
    protocol = load(identity["suite"])
    if sha(Path(identity["suite"])/"protocol.json") != identity["protocol_sha256"]:
        raise ValueError("capture protocol changed")
    cap_report, rep_report = read(capture/"report.json"), read(replay/"report.json")
    if any((p/"error.json").exists() for p in (capture, replay)) or cap_report["missing_targets"]:
        raise ValueError("capture/replay is failed or incomplete")
    seed = protocol["settings"]["seed"]
    trace = read(Path(identity["suite"])/f"traces/{identity['mode']}-seed-{seed}.json")
    if trace != read(capture/"trace.json"):
        raise ValueError("capture Trace differs from frozen protocol")
    from benchmarks.v2_stall.metrics import records_from_rows
    from benchmarks.v2_baselines.metrics import analyze_online
    records = records_from_rows(read(capture/"requests.json"))
    metrics = analyze_online(trace, records, read(capture/"steps.json"), read(capture/"backlog.json"),
                             cap_report["end_ns"], protocol["settings"]["starvation_ms"])
    if not metrics["all_completed"] or cap_report["driver_error"] or not cap_report["step_state_checks_passed"]:
        raise ValueError("capture driver/state/completion failed")
    prompts = {r["request_id"]: r["prompt_token_ids"] for r in trace["requests"]}
    outputs = {r.request_id: r.output_token_ids for r in records}
    captured, replayed = {}, {}
    for entry in cap_report["captures"]:
        image = torch.load(capture/entry["file"], map_location="cpu", weights_only=True)
        if image["plan"] != entry["plan"]:
            raise ValueError("snapshot plan differs from JSON index")
        validate_context(image["plan"], image["context"])
        for target in entry["targets"]:
            i = target["request_id"]
            if i in captured:
                raise ValueError("duplicate captured target")
            row = next(r for r in entry["plan"]["rows"] if r["seq_id"] == target["seq_id"])
            index = identity["output_token"]-1
            if (target["history"] != row["history"] or target["history"] != prompts[i]+outputs[i][:index]
                    or row["output_count"] != index or target["sampled_token"] != outputs[i][index]):
                raise ValueError("captured history/token differs from actual driver output")
            x = image["target_logits"][str(target["seq_id"])]
            if compare_logits(x, x)["actual"] != target["logits"] or int(x.argmax()) != target["sampled_token"]:
                raise ValueError("capture summary differs from raw vector")
            captured[i] = {"request_id": i, "history": target["history"], "logits": x,
                           "snapshot": entry["file"], "plan": entry["plan"]}
    if set(captured) != set(identity["target_ids"]):
        raise ValueError("missing/extra capture targets")
    for entry in rep_report["entries"]:
        i = entry["request_id"]
        if i not in captured or i in replayed or entry["history"] != captured[i]["history"]:
            raise ValueError("replay target identity/history differs")
        vectors = torch.load(replay/entry["vectors_file"], map_location="cpu", weights_only=True)
        n = ri["repeats"]
        full, isolated = [vectors[f"full_{j}"] for j in range(n)], [vectors[f"isolated_{j}"] for j in range(n)]
        if not torch.equal(vectors["captured"], captured[i]["logits"]):
            raise ValueError("replay captured vector was replaced")
        expected = {
            "full_vs_captured": [compare_logits(x, vectors["captured"]) for x in full],
            "full_repeats_vs_first": [compare_logits(x, full[0]) for x in full[1:]],
            "isolated_vs_full": [compare_logits(x, full[0]) for x in isolated],
            "isolated_repeats_vs_first": [compare_logits(x, isolated[0]) for x in isolated[1:]],
        }
        if any(entry[k] != v for k, v in expected.items()):
            raise ValueError("replay metric differs from raw vector")
        replayed[i] = entry
    if set(replayed) != set(captured):
        raise ValueError("missing replay targets")
    return identity, captured, replayed


def prepare(args):
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    protocol = load(suite)
    seed = protocol["settings"]["seed"]
    trace = read(suite/f"traces/{args.mode}-seed-{seed}.json")
    requests = {r["request_id"]: r for r in trace["requests"]}
    if (len(set(args.target_ids)) != len(args.target_ids)
            or any(i not in requests or requests[i]["max_output_tokens"] < args.output_token for i in args.target_ids)):
        raise ValueError("invalid target requests/output index")
    out.mkdir(parents=True, exist_ok=False)
    save(out/"plan.json", {"schema": "online-replay-plan-v1", "suite": str(suite),
        "protocol_sha256": sha(suite/"protocol.json"), "mode": args.mode,
        "target_ids": args.target_ids, "output_token": args.output_token,
        "tool_sha256": tool_hashes(), "policies": ["baseline", "chunk-512"],
        "capture_repeats": 2, "replay_repeats": args.repeats,
        "diagnostic_only": True, "v1_acceptance_passed": False})
    seal(out)
    print("Frozen independent diagnostic plan:", out)


def report(args):
    root = Path(args.root).resolve()
    verify_artifacts(root/"plan")
    plan = read(root/"plan/plan.json")
    if plan["tool_sha256"] != tool_hashes():
        raise ValueError("diagnostic tool changed after plan freeze")
    load(plan["suite"])
    if plan["protocol_sha256"] != sha(Path(plan["suite"])/"protocol.json"):
        raise ValueError("planned protocol changed")
    data, missing, details = {}, [], []
    fixed_mismatch = repeat_mismatch = isolate_mismatch = 0
    lines = ["# 在线Token捕获与固定Batch重放", "", "仅诊断，不作为性能测量或V1验收。", "",
             "| 捕获运行 | 请求 | 捕获Top1 | Top2差距 | 固定Batch复现Top1 | 固定Batch重复稳定 | 单请求Top1与完整Batch相同 |",
             "|---|---|---:|---:|---|---|---|"]
    for repeat in range(1, plan["capture_repeats"]+1):
        for policy in plan["policies"]:
            base = root/f"repeat-{repeat}"/policy
            if any(not (base/name/"report.json").exists() for name in ("capture", "replay")):
                missing.append(str(base))
                continue
            identity, captured, replayed = audit_replay(base/"capture", base/"replay")
            for key in ("suite", "protocol_sha256", "mode", "target_ids", "output_token", "tool_sha256"):
                if identity[key] != plan[key]:
                    raise ValueError(f"capture differs from planned {key}")
            if identity["policy"] != policy or read(base/"replay/identity.json")["repeats"] != plan["replay_repeats"]:
                raise ValueError("replay policy/repetition differs from plan")
            label = f"repeat-{repeat}/{policy}"
            data[label] = captured
            for i, entry in replayed.items():
                full_ok = all(x["same_top1"] for x in entry["full_vs_captured"])
                stable = all(x["exact"] for x in entry["full_repeats_vs_first"])
                isolation_ok = all(x["same_top1"] for x in entry["isolated_vs_full"])
                fixed_mismatch += not full_ok
                repeat_mismatch += not stable
                isolate_mismatch += not isolation_ok
                lines.append(f"| {label} | {i} | {entry['captured']['argmax']} | {entry['captured']['top2_margin']:.8g} | {full_ok} | {stable} | {isolation_ok} |")
                details.append({"label": label, **entry})
    crosses = []
    for (la, a), (lb, b) in combinations(data.items(), 2):
        for i in plan["target_ids"]:
            crosses.append({"left": la, "right": lb, **compare_captured_vectors(a[i], b[i])})
    summary = {"audited_capture_replay_pairs": len(data), "expected_pairs": 4,
        "missing": missing, "full_vs_capture_top1_mismatch_cases": fixed_mismatch,
        "full_repeat_not_bitexact_cases": repeat_mismatch,
        "isolated_vs_full_top1_mismatch_cases": isolate_mismatch,
        "cross_capture_history_mismatch_cases": sum(not x["same_history"] for x in crosses),
        "cross_capture_common_history_top1_mismatch_cases": sum(x["logits"] is not None and not x["logits"]["same_top1"] for x in crosses),
        "v1_acceptance_passed": False, "status": "INCOMPLETE" if missing else "MEASURED_NEEDS_REVIEW"}
    out = Path(tempfile.mkdtemp(prefix="report.", dir=root))
    save(out/"summary.json", summary)
    save(out/"replay_details.json", details)
    save(out/"cross_capture.json", crosses)
    lines += ["", "固定Batch每次恢复相同输入、Context与有效KV前缀；单请求对照还会改变Batch形状，可能改变Attention接口。",
              "不同捕获运行只有完整Token历史相同时才比较Logits；不会自动把近似并列、状态通过或重复稳定视为正确性豁免。",
              "在线复制会扰动到达时序。当前快照不是旧性能运行的历史快照，也不是独立KV数值正确性证明。"]
    with (out/"report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    seal(out)
    print("Diagnostic report:", out)
    print(summary)
    if missing:
        raise SystemExit(2)
