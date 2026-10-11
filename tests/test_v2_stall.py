"""Public experiment seams; no CUDA or model imports."""
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
import tempfile
from pathlib import Path

from benchmarks.v2_stall.protocol import Experiment, build_trace
from benchmarks.v0.metrics import RequestRecord


class TraceTests(unittest.TestCase):
    def test_tune_and_evaluation_have_disjoint_fixed_arrivals(self):
        config = Experiment()
        # Token generation is an external tokenizer boundary, not a mocked engine.
        tokens = lambda length, seed, index: [100 + seed + index] * length
        tune = build_trace(config, 141, "tune", tokens)
        evaluation = build_trace(config, 241, "eval", tokens)
        self.assertEqual(len(tune["requests"]), 9)
        self.assertEqual([r["arrival_ns"] for r in tune["requests"]], [0] * 8 + [2_000_000_000])
        self.assertEqual([r["arrival_ns"] for r in evaluation["requests"]], [0] * 8 + [2_000_000_000])
        self.assertNotEqual(tune["requests"][0]["prompt_token_ids"],
                            evaluation["requests"][0]["prompt_token_ids"])
        self.assertEqual(tune["decode_ids"], [f"decode-{i}" for i in range(8)])
        self.assertEqual(tune["requests"][-1]["max_output_tokens"], 64)

    def test_prepare_freezes_traces_and_detects_later_source_or_trace_changes(self):
        from benchmarks.v2_stall.suite import prepare_suite
        from benchmarks.v2_stall.protocol import load_suite
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "benchmarks").mkdir()
            source = root / "benchmarks/example.py"
            source.write_text("value = 1\n")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":4096}')
            out = root / "suite"
            prepare_suite(out, Experiment(), model, "test-revision",
                          lambda n, s, i: [100+s+i] * n, parent=None, root=root)
            protocol, _ = load_suite(out, root=root)
            self.assertEqual(len(protocol["trace_sha256"]), 6)
            self.assertFalse(protocol["v1_acceptance_passed"])
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "source set changed"):
                load_suite(out, root=root)
            source.write_text("value = 1\n")
            (out / "eval/seed-241.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact missing or modified"):
                load_suite(out, root=root)

    def test_rejects_overlap_or_impossible_baseline_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            replace(Experiment(), eval_seeds=(141, 242, 243)).validate()
        with self.assertRaisesRegex(ValueError, "global budget"):
            replace(Experiment(), token_budget=512).validate()

    def test_actual_prompt_overlap_is_rejected_not_just_seed_overlap(self):
        from benchmarks.v2_stall.suite import prepare_suite
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":4096}')
            with self.assertRaisesRegex(ValueError, "actual prompt tokens overlap"):
                prepare_suite(root / "suite", Experiment(), model, "r",
                              lambda n, s, i: [100] * n, parent=None, root=root)

    def test_nested_regression_evidence_is_protected(self):
        from benchmarks.v2_stall.protocol import save, seal, verify_artifacts
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "case").mkdir()
            save(folder / "case/report.json", {"passed": False})
            seal(folder)
            verify_artifacts(folder)
            (folder / "case/report.json").write_text('{"passed":true}')
            with self.assertRaisesRegex(ValueError, "artifact missing or modified"):
                verify_artifacts(folder)


class MetricTests(unittest.TestCase):
    def test_stall_includes_intervals_crossing_both_window_edges(self):
        from benchmarks.v2_stall.metrics import analyze_run
        config = replace(Experiment(), decode_count=1, inject_s=1.0, window_s=.2,
                         stable_intervals=2)
        trace = build_trace(config, 141, "tune", lambda n, s, i: [101 + i] * n)
        # The 300ms interval crosses both [1.0, 1.2) boundaries: do NOT clip it.
        short = RequestRecord("decode-0", 0, 256, 512, 1000, 100,
                              enqueue_ns=0, first_scheduled_ns=0, finish_ns=1_400_000_000,
                              status="completed", output_token_ids=[9] * 7,
                              token_times_ns=[600_000_000, 700_000_000, 800_000_000,
                                              900_000_000, 1_200_000_000, 1_300_000_000, 1_400_000_000],
                              preemptions=0, recomputed_tokens=0)
        long = RequestRecord("long", 1_000_000_000, 3072, 64, 1000, 100,
                             enqueue_ns=1_010_000_000, first_scheduled_ns=1_010_000_000,
                             finish_ns=1_300_000_000, status="completed", output_token_ids=[8, 8],
                             token_times_ns=[1_100_000_000, 1_300_000_000],
                             preemptions=0, recomputed_tokens=0)
        # A complete run must have fixed output lengths, not just plausible timestamps.
        invalid = analyze_run(trace, [short, long], config)
        self.assertFalse(invalid["valid_experiment"])
        self.assertIn("output_length_mismatch:decode-0", invalid["invalid_reasons"])
        short.max_output_tokens = 7
        long.max_output_tokens = 2
        trace["requests"][0]["max_output_tokens"] = 7
        trace["requests"][1]["max_output_tokens"] = 2
        result = analyze_run(trace, [short, long], config)
        self.assertTrue(result["valid_experiment"], result)
        self.assertEqual(result["decode_stall_max_ms"], 300.0)
        self.assertEqual(result["long_ttft_ms"], 100.0)
        self.assertEqual(result["window_throughput_tok_s"], 5.0)

    def test_unstable_or_finished_decode_is_not_silently_dropped(self):
        from benchmarks.v2_stall.metrics import analyze_run
        config = replace(Experiment(), decode_count=1)
        trace = build_trace(config, 141, "tune", lambda n, s, i: [101] * n)
        short = RequestRecord("decode-0", 0, 256, 512, 1000, 100,
                              status="unfinished", token_times_ns=[500_000_000])
        long = RequestRecord("long", 2_000_000_000, 3072, 64, 1000, 100,
                             status="rejected", error="capacity")
        result = analyze_run(trace, [short, long], config)
        self.assertFalse(result["valid_experiment"])
        self.assertIn("decode_not_stable:decode-0", result["invalid_reasons"])
        self.assertIn("decode_window_censored:decode-0", result["invalid_reasons"])
        self.assertEqual(result["completed_count"], 0)

    def test_missing_or_invalid_repeat_blocks_aggregate_instead_of_filtering(self):
        from benchmarks.v2_stall.metrics import aggregate_pairs
        with self.assertRaises(ValueError):
            aggregate_pairs([], [241, 242, 243])
        pairs = [{"seed": s, "baseline": {"valid_experiment": True},
                  "candidate": {"valid_experiment": s != 242}} for s in [241, 242, 243]]
        result = aggregate_pairs(pairs, [241, 242, 243])
        self.assertFalse(result["valid_for_performance_comparison"])
        self.assertEqual(result["invalid_seeds"], [242])
        self.assertNotIn("candidate", result)

    def test_first_branch_uses_common_history_not_divergent_outputs(self):
        from benchmarks.v2_stall.metrics import compare_tokens
        trace = {"requests": [{"request_id": "a", "prompt_token_ids": [10, 20]}]}
        row = compare_tokens(trace, {"a": [30, 40, 50]}, {"a": [30, 99, 88]})[0]
        self.assertEqual(row["first_divergence_token_1based"], 2)
        self.assertEqual(row["same_history_token_ids"], [10, 20, 30])
        self.assertEqual(row["review_status"], "NEEDS_REVIEW")

    def test_paired_variability_and_selection_keep_throughput_cost(self):
        from benchmarks.v2_stall.metrics import aggregate_pairs, choose_chunk
        keys = ["decode_stall_max_ms", "decode_itl_p99_ms", "long_ttft_ms",
                "window_throughput_tok_s", "ttft_p99_ms", "itl_p99_ms",
                "window_goodput_rps", "cohort_violation_rate"]
        pairs = []
        for seed, stall in zip([141, 142, 143], [40., 50., 60.]):
            a = {k: 100. for k in keys}
            b = {**a, "decode_stall_max_ms": stall, "window_throughput_tok_s": 98.}
            a["valid_experiment"] = b["valid_experiment"] = True
            pairs.append({"seed": seed, "baseline": a, "candidate": b})
        good = aggregate_pairs(pairs, [141, 142, 143])
        self.assertEqual(good["candidate"]["decode_stall_max_ms"],
                         {"n": 3, "mean": 50., "std": 10., "min": 40., "max": 60.})
        self.assertEqual(good["paired_stall_reduction_pct"]["mean"], 50.)
        bad = {**good, "paired_throughput_change_pct": {"mean": -10.}}
        selection = choose_chunk({256: bad, 512: good}, Experiment())
        self.assertEqual(selection["chunk"], 512)


class RunTests(unittest.TestCase):
    def test_busy_gpu_keeps_planned_injection_time_and_records_lag(self):
        from benchmarks.v2_stall.driver import execute_experiment
        config = replace(Experiment(), decode_count=1, inject_s=.25, window_s=.15,
                         stable_intervals=2, decode_output_tokens=12, long_output_tokens=2)
        trace = build_trace(config, 141, "tune", lambda n, s, i: [101] * n)
        # A hardware boundary: each GPU step lasts exactly 100ms.
        engine = ClockedDevice()
        gpu_config = SimpleNamespace(max_model_len=4096, num_kvcache_blocks=100,
                                     kvcache_block_size=256, hf_config=SimpleNamespace(vocab_size=1000))
        result, summary = execute_experiment(engine, gpu_config, trace, config, sleep=lambda _: None)
        long = next(r for r in result.records if r.request_id == "long")
        self.assertEqual(long.arrival_ns, 250_000_000)
        self.assertEqual(long.enqueue_ns, 300_000_000)
        self.assertEqual(summary["long_enqueue_lag_ms"], 50.0)
        # At the PLANNED 250ms time, only two tokens exist: not stable yet.
        self.assertIn("decode_not_stable:decode-0", summary["invalid_reasons"])

    def test_archived_run_audit_rejects_wrong_seed_even_with_valid_file_hashes(self):
        from benchmarks.v2_stall.suite import audit_run
        from benchmarks.v2_stall.metrics import analyze_run
        from benchmarks.v2_stall.protocol import save, seal, sha
        config = replace(Experiment(), decode_count=1)
        trace = build_trace(config, 241, "eval", lambda n, s, i: [101] * n)
        rows = []
        for spec in trace["requests"]:
            times = ([100_000_000+i*10_000_000 for i in range(512)] if spec["request_id"] == "decode-0"
                     else [2_200_000_000+i*10_000_000 for i in range(64)])
            record = RequestRecord(spec["request_id"], spec["arrival_ns"], len(spec["prompt_token_ids"]),
                spec["max_output_tokens"], 1000, 100, enqueue_ns=spec["arrival_ns"],
                first_scheduled_ns=spec["arrival_ns"], finish_ns=times[-1], status="completed",
                output_token_ids=[999]*len(times), token_times_ns=times, preemptions=0, recomputed_tokens=0)
            rows.append(record)
        summary = analyze_run(trace, rows, config)
        summary.update(diagnostic_only=False, source_unchanged=True, policy="baseline", chunk=None,
                       seed=241, split="eval")
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            out = folder / "run"
            out.mkdir()
            protocol = {"source_sha256": {}, "model_metadata_sha256": {}}
            save(folder / "protocol.json", protocol)
            save(out / "trace.json", trace)
            save(out / "summary.json", summary)
            save(out / "environment.json", {"source_sha256": {}, "model_metadata_sha256": {}})
            save(out / "requests.json", [asdict(r) for r in rows])
            save(out / "run_config.json", {"suite": str(folder), "protocol_sha256": sha(folder / "protocol.json"),
                "seed": 999, "split": "eval", "policy": "baseline", "chunk": None,
                "tp": 1, "graph": False, "observation": "full"})
            seal(out)
            with self.assertRaisesRegex(ValueError, "run identity"):
                audit_run(out, protocol, config, trace)

    def test_cpu_archive_to_paired_report_keeps_all_three_seeds_and_branches(self):
        from benchmarks.v2_stall.suite import prepare_suite, pair_reports, render_report
        from benchmarks.v2_stall.metrics import analyze_run, aggregate_pairs
        from benchmarks.v2_stall.protocol import save, seal, sha, read
        config = replace(Experiment(), decode_count=1, chunks=(256,))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":4096}')
            folder = root / "suite"
            protocol = prepare_suite(folder, config, model, "r", lambda n, s, i: [100+s+i] * n,
                                     parent=None, root=root)
            for seed in config.eval_seeds:
                trace = read(folder / "eval" / f"seed-{seed}.json")
                for chunk in (None, 256):
                    policy = "baseline" if chunk is None else "chunk"
                    out = folder / "runs/eval" / f"seed-{seed}-{'baseline' if chunk is None else 'chunk-256'}"
                    out.mkdir(parents=True)
                    rows = []
                    for spec in trace["requests"]:
                        long = spec["request_id"] == "long"
                        times = ([100_000_000+i*10_000_000 for i in range(512)] if not long else
                                 [2_200_000_000+i*10_000_000 for i in range(64)])
                        output = [999]*len(times)
                        if chunk is not None and not long:
                            output[10] = 998  # Keep a real branch in EVERY evaluation pair.
                        rows.append(RequestRecord(spec["request_id"], spec["arrival_ns"], len(spec["prompt_token_ids"]),
                            spec["max_output_tokens"], 1000, 100, enqueue_ns=spec["arrival_ns"],
                            first_scheduled_ns=spec["arrival_ns"], finish_ns=times[-1], status="completed",
                            output_token_ids=output, token_times_ns=times, preemptions=0, recomputed_tokens=0))
                    summary = analyze_run(trace, rows, config)
                    summary.update(policy=policy, chunk=chunk, seed=seed, split="eval", diagnostic_only=False,
                                   source_unchanged=True, kv_blocks=100)
                    save(out / "summary.json", summary)
                    save(out / "trace.json", trace)
                    save(out / "requests.json", [asdict(r) for r in rows])
                    save(out / "environment.json", {"source_sha256": {}, "model_metadata_sha256": protocol["model_metadata_sha256"],
                         "model_revision_declared": "r", "packages": {}, "gpus": []})
                    save(out / "run_config.json", {"suite": str(folder), "protocol_sha256": sha(folder / "protocol.json"),
                        "policy": policy, "chunk": chunk, "seed": seed, "split": "eval", "tp": 1,
                        "graph": False, "observation": "full"})
                    seal(out)
            pairs = pair_reports(folder, "eval", 256, config.eval_seeds, protocol, config)
            aggregate = aggregate_pairs(pairs, config.eval_seeds)
            self.assertTrue(aggregate["valid_for_performance_comparison"])
            self.assertEqual(aggregate["candidate"]["decode_stall_max_ms"]["n"], 3)
            self.assertEqual(sum(not row["exact_match"] for p in pairs for row in p["divergences"]), 3)
            selection = folder / "selection"
            selection.mkdir()
            save(selection / "selection.json", {"chunk": 256, "selection_split": "tune",
                                                "protocol_sha256": sha(folder / "protocol.json")})
            seal(selection)
            report = render_report(folder, protocol, config)
            self.assertIn("非完全一致的请求执行数：3", (report / "report.md").read_text())
            self.assertFalse(read(report / "aggregate.json")["v1_acceptance_passed"])


class ClockedDevice:
    def __init__(self):
        self.timestamp = 0
        self.scheduler = SimpleNamespace(waiting=[], running=[])
        self.records = []

    def start_clock(self):
        self.timestamp = 0

    def now(self):
        return self.timestamp

    def submit(self, spec, record):
        record.enqueue_ns = self.now()
        record.preemptions = record.recomputed_tokens = 0
        self.records.append(record)
        self.scheduler.running.append(record)

    def step(self):
        self.timestamp += 100_000_000
        for r in list(self.scheduler.running):
            if r.first_scheduled_ns is None:
                r.first_scheduled_ns = r.enqueue_ns
            r.token_times_ns.append(self.now())
            r.output_token_ids.append(999)
            if len(r.output_token_ids) == r.max_output_tokens:
                r.status, r.finish_ns = "completed", self.now()
                self.scheduler.running.remove(r)

    def is_finished(self):
        return not self.scheduler.running

    def snapshot_outputs(self):
        pass

    def close_waiting_spans(self, _):
        pass


if __name__ == "__main__":
    unittest.main()
