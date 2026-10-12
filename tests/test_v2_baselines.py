"""Public fixed-policy, trace and measurement contracts; no CUDA imports."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import test_v1_state as state


def setUpModule():
    global FixedScheduler
    state.setUpModule()
    FixedScheduler = state.load(
        "nanovllm.engine.fixed_scheduler", "nanovllm/engine/fixed_scheduler.py"
    ).FixedScheduler


def tearDownModule():
    state.tearDownModule()


class PolicyTests(unittest.TestCase):
    def make(self, policy="chunk", budget=512, slots=32, blocks=32, chunk=256):
        self.clock = [0]
        config = SimpleNamespace(
            max_num_seqs=slots, max_num_batched_tokens=budget, eos=2,
            kvcache_block_size=256, num_kvcache_blocks=blocks, max_model_len=4096,
        )
        return FixedScheduler(config, chunk, policy, max_prefill_wait_ms=10,
                              clock=lambda: self.clock[0])

    def seq(self, length, value=100, output=4):
        return state.Sequence([value] * length,
                              state.Params(max_tokens=output, ignore_eos=True))

    def commit(self, scheduler, batch):
        scheduler.postprocess(batch, {i: 1000 for i in batch.sample_seq_ids})
        scheduler.assert_consistent()

    def test_strict_decode_first_defers_prefill_until_decode_finishes(self):
        scheduler = self.make("decode_first")
        first = self.seq(16, output=2)
        scheduler.add(first)
        self.commit(scheduler, scheduler.schedule())
        second = self.seq(513, value=200)
        scheduler.add(second)
        batch = scheduler.schedule()
        self.assertEqual([(r.seq_id, r.phase) for r in batch.requests],
                         [(first.seq_id, "decode")])
        self.commit(scheduler, batch)
        self.assertEqual(scheduler.schedule().requests[0].seq_id, second.seq_id)

    def test_all_policies_drain_preemption_without_duplicate_outputs(self):
        for policy in ("chunk", "decode_first", "decode_first_wait"):
            scheduler = self.make(policy, blocks=2)
            seqs = [self.seq(256, value=100), self.seq(256, value=200)]
            for seq in seqs:
                scheduler.add(seq)
            for _ in range(100):
                if scheduler.is_finished():
                    break
                self.clock[0] += 10_000_000
                self.commit(scheduler, scheduler.schedule())
            self.assertTrue(scheduler.is_finished())
            self.assertEqual([s.num_completion_tokens for s in seqs], [4, 4])
            self.assertGreater(scheduler.preemption_count, 0)
            self.assertFalse(scheduler.block_manager.used_block_ids)
    def test_wait_protection_preempts_one_decode_slot_and_resets_on_progress(self):
        scheduler = self.make("decode_first_wait", budget=256, slots=1)
        first = self.seq(16)
        scheduler.add(first)
        self.commit(scheduler, scheduler.schedule())
        second = self.seq(513, value=200)
        scheduler.add(second)
        batch = scheduler.schedule()
        self.assertEqual(batch.requests[0].phase, "decode")
        self.commit(scheduler, batch)
        self.clock[0] = 10_000_000
        batch = scheduler.schedule()
        self.assertEqual([(r.seq_id, r.phase, r.num_scheduled_tokens) for r in batch.requests],
                         [(second.seq_id, "prefill", 256)])
        self.commit(scheduler, batch)
        self.assertEqual(scheduler.schedule().requests[0].phase, "decode")

    def test_forced_chunk_respects_global_budget_and_decode_first_metadata(self):
        scheduler = self.make("decode_first_wait", budget=257)
        first = self.seq(16)
        scheduler.add(first)
        self.commit(scheduler, scheduler.schedule())
        scheduler.add(self.seq(513, value=200))
        self.clock[0] = 10_000_000
        batch = scheduler.schedule()
        self.assertEqual([r.phase for r in batch.requests], ["decode", "prefill"])
        self.assertEqual((batch.num_decode_tokens, batch.num_prefill_tokens), (1, 256))
        self.commit(scheduler, batch)

    def test_all_chunks_commit_prefix_pages_only_after_execution(self):
        for chunk in (256, 512, 1024, 2048):
            scheduler = self.make(chunk=chunk)
            first = self.seq(513)
            scheduler.add(first)
            batch = scheduler.schedule()
            self.assertEqual(first.num_cached_tokens, 0)
            self.assertEqual(scheduler.block_manager.blocks[first.block_table[0]].hash, -1)
            self.commit(scheduler, batch)
            for _ in range(20):
                if scheduler.is_finished():
                    break
                self.commit(scheduler, scheduler.schedule())
            self.assertTrue(scheduler.is_finished())
            hot = self.seq(513, output=1)
            scheduler.add(hot)
            batch = scheduler.schedule()
            self.assertEqual((batch.requests[0].start_pos, batch.num_prefill_tokens), (512, 1))
            self.commit(scheduler, batch)
            self.assertFalse(scheduler.block_manager.used_block_ids)

    def test_wait_guard_records_kv_conflict_and_does_not_corrupt_decode(self):
        scheduler = self.make("decode_first_wait", blocks=2)
        scheduler.add(self.seq(257))
        for _ in range(2):
            self.commit(scheduler, scheduler.schedule())
        scheduler.add(self.seq(256, value=200))
        self.clock[0] = 10_000_000
        batch = scheduler.schedule()
        self.assertEqual([r.phase for r in batch.requests], ["decode"])
        self.assertEqual(scheduler.last_decision["wait_protection_conflict"], "KV_capacity")
        self.commit(scheduler, batch)


class TraceTests(unittest.TestCase):
    def test_poisson_and_burst_share_count_and_prompts_but_not_arrivals(self):
        from benchmarks.v2_baselines.workload import Workload, build_trace
        config = Workload(warmup_s=2, measurement_s=8, rps=2)
        factory = lambda n, index: [100+index] * n
        poisson = build_trace(config, "poisson", 41, factory, "model-revision")
        burst = build_trace(config, "burst", 41, factory, "model-revision")
        self.assertEqual(len(poisson["requests"]), len(burst["requests"]))
        self.assertGreater(len(poisson["requests"]), 5)
        self.assertEqual([r["prompt_token_ids"] for r in poisson["requests"]],
                         [r["prompt_token_ids"] for r in burst["requests"]])
        self.assertNotEqual([r["arrival_ns"] for r in poisson["requests"]],
                            [r["arrival_ns"] for r in burst["requests"]])

    def test_length_mixture_and_schema_reject_silent_truncation(self):
        from dataclasses import replace
        from benchmarks.v2_baselines.workload import Workload, build_trace, validate_trace
        config = Workload(rps=10, long_fraction=1)
        trace = build_trace(config, "long_short", 41, lambda n, i: [100]*n, "rev")
        self.assertTrue(all(len(r["prompt_token_ids"]) == 1024 for r in trace["requests"]))
        trace["requests"][0]["prompt_token_ids"].pop()
        with self.assertRaises(ValueError):
            validate_trace(trace)
        with self.assertRaises(ValueError):
            replace(config, burst_multiplier=4).validate()


class MetricTests(unittest.TestCase):
    def test_unfinished_cohort_remains_in_slo_and_wait_statistics(self):
        from benchmarks.v0.metrics import RequestRecord
        from benchmarks.v2_baselines.metrics import analyze_online
        from benchmarks.v2_baselines.workload import Workload
        from dataclasses import asdict
        workload = Workload(warmup_s=0, measurement_s=1, short_tokens=1, output_tokens=2)
        trace = {"schema": "v2-online-trace-v1", "mode": "poisson", "seed": 41,
                 "model_revision": "rev", "warmup_s": 0, "measurement_s": 1,
                 "workload": asdict(workload), "request_count": 2, "requests": [
                     {"request_id": name, "arrival_ns": 0, "prompt_token_ids": [100],
                      "max_output_tokens": 2, "ttft_slo_ms": 1000, "itl_slo_ms": 100}
                     for name in ("a", "b")]}
        a = RequestRecord("a", 0, 1, 2, 1000, 100, status="completed", finish_ns=200_000_000,
                          token_times_ns=[100_000_000, 200_000_000], output_token_ids=[1, 2],
                          waiting_spans=[{"start_ns": 0, "end_ns": 50_000_000}])
        b = RequestRecord("b", 0, 1, 2, 1000, 100, status="unfinished",
                          waiting_spans=[{"start_ns": 0, "end_ns": 2_000_000_000}])
        summary = analyze_online(trace, [a, b], [], [], 2_000_000_000, 1000)
        self.assertEqual(summary["cohort_violation_rate"], .5)
        self.assertEqual(summary["window_goodput_rps"], 1)
        self.assertEqual(summary["window_throughput_tok_s"], 2)
        self.assertEqual(summary["all_requests_tpot_mean_ms"], 100)
        self.assertEqual(summary["prefill_wait_max_ms"], 2000)
        self.assertEqual(summary["starvation_request_count"], 1)
        self.assertFalse(summary["all_completed"])


class ProtocolTests(unittest.TestCase):
    def test_new_suite_freezes_sources_traces_and_rejects_overwrite(self):
        from benchmarks.v2_baselines.protocol import prepare, load, Settings
        from benchmarks.v2_baselines.workload import Workload
        from benchmarks.v2_stall.protocol import save
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            save(model / "config.json", {"max_position_embeddings": 40960, "vocab_size": 20000})
            folder = root / "suite"
            prepare(folder, Settings(), Workload(), str(model), "revision",
                    lambda n, i: [100+i]*n, root=root)
            self.assertEqual(load(folder, root=root)["stage"], "V2-stage1-smoke")
            with self.assertRaises(ValueError):
                prepare(folder, Settings(), Workload(), str(model), "revision",
                        lambda n, i: [100+i]*n, root=root)
            trace = folder / "traces/poisson-seed-41.json"
            trace.write_text(trace.read_text()+" ")
            with self.assertRaises(ValueError):
                load(folder, root=root)

    def test_audit_recomputes_summary_even_when_artifact_hashes_are_valid(self):
        from benchmarks.v2_baselines.audit import audit_run
        from benchmarks.v2_baselines.protocol import prepare, Settings
        from benchmarks.v2_baselines.workload import Workload
        from benchmarks.v2_baselines.metrics import analyze_online
        from benchmarks.v2_stall.protocol import save, seal, sha, read
        from benchmarks.v0.metrics import RequestRecord
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/"model"
            model.mkdir()
            save(model/"config.json", {"max_position_embeddings": 40960})
            suite = root/"suite"
            protocol = prepare(suite, Settings(), Workload(short_tokens=1, output_tokens=2),
                               str(model), "rev", lambda n, i: [100]*n, root=root)
            trace = read(suite/"traces/poisson-seed-41.json")
            records = [RequestRecord(r["request_id"], r["arrival_ns"], 1, 2, 1000, 100,
                status="completed", finish_ns=r["arrival_ns"]+200_000_000,
                token_times_ns=[r["arrival_ns"]+100_000_000, r["arrival_ns"]+200_000_000],
                output_token_ids=[1, 2]) for r in trace["requests"]]
            out = suite/"fake-run"
            out.mkdir()
            save(out/"trace.json", trace)
            save(out/"run_config.json", {"policy": "baseline", "mode": "poisson", "seed": 41,
                "tp": 1, "graph": False, "observation": "full", "diagnostic_only": False,
                "protocol_sha256": sha(suite/"protocol.json")})
            summary = analyze_online(trace, records, [], [], 11_000_000_000, 1000)
            summary.update(valid_measurement=True, source_unchanged=True, end_state_check_passed=True,
                           end_ns=11_000_000_000, policy="baseline", mode="poisson", seed=41,
                           diagnostic_only=False, kv_blocks=302)
            summary["window_throughput_tok_s"] = 999999
            save(out/"summary.json", summary)
            save(out/"requests.json", [asdict(r) for r in records])
            save(out/"steps.json", [])
            save(out/"backlog.json", [])
            save(out/"environment.json", {"model_metadata_sha256": protocol["model_metadata_sha256"],
                                         "source_sha256": {}})
            seal(out)
            with self.assertRaisesRegex(ValueError, "summary"):
                audit_run(suite, out, root=root)


class ReportTests(unittest.TestCase):
    def test_mismatched_kv_capacity_invalidates_pair_without_dropping_it(self):
        from benchmarks.v2_baselines.suite import compare_pair
        env = {"packages": {}, "gpus": [], "model_revision_declared": "rev", "torch_cuda_version": "12.8"}
        a = {"valid_measurement": True, "all_completed": True, "diagnostic_only": False,
             "kv_blocks": 302, "kv_payload_bytes": 10872, "policy": "baseline", "mode": "poisson", "seed": 41}
        b = {**a, "policy": "chunk-256", "kv_blocks": 301}
        pair = compare_pair((a, [], env), (b, [], env), {"requests": []})
        self.assertFalse(pair["valid_for_comparison"])
        self.assertIn("KV_capacity_differs", pair["invalid_reasons"])
        self.assertEqual(pair["candidate"]["kv_blocks"], 301)

    def test_complete_archived_smoke_audits_all_21_runs_and_retains_a_branch(self):
        from dataclasses import asdict
        from benchmarks.v0.metrics import RequestRecord
        from benchmarks.v2_baselines.protocol import prepare, Settings, POLICIES
        from benchmarks.v2_baselines.workload import Workload
        from benchmarks.v2_baselines.metrics import analyze_online
        from benchmarks.v2_baselines.suite import report
        from benchmarks.v2_stall.protocol import save, seal, sha, read
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/"model"
            model.mkdir()
            save(model/"config.json", {"max_position_embeddings": 40960})
            suite = Path(tmp)/"suite"
            protocol = prepare(suite, Settings(), Workload(short_tokens=1, long_tokens=2, output_tokens=2),
                               str(model), "rev", lambda n, i: [100+i]*n)
            identity = sha(suite/"protocol.json")
            reg = suite/"regression"
            reg.mkdir()
            save(reg/"report.json", {"state_attention_passed": True, "protocol_sha256": identity,
                                    "note": "CPU archived-hardware-output fixture, not actual GPU evidence"})
            seal(reg)
            env = {"model_metadata_sha256": protocol["model_metadata_sha256"],
                   "source_sha256": {k:v for k,v in protocol["source_sha256"].items() if k.startswith("nanovllm/")},
                   "packages": {}, "gpus": [{"name": "CPU-fixture"}], "model_revision_declared": "rev",
                   "torch_cuda_version": "fixture"}
            for mode in ("poisson", "burst", "long_short"):
                trace = read(suite/f"traces/{mode}-seed-41.json")
                for policy in POLICIES:
                    out = suite/"runs"/mode/policy
                    out.mkdir(parents=True)
                    records = [RequestRecord(r["request_id"], r["arrival_ns"], len(r["prompt_token_ids"]),
                        2, 1000, 100, status="completed", finish_ns=r["arrival_ns"]+200_000_000,
                        token_times_ns=[r["arrival_ns"]+100_000_000, r["arrival_ns"]+200_000_000],
                        output_token_ids=[1, 2]) for r in trace["requests"]]
                    # A single external-hardware output branch must survive audit.
                    if mode == "burst" and policy == "chunk-256":
                        records[0].output_token_ids[1] = 3
                    config = {"policy": policy, "mode": mode, "seed": 41, "tp": 1,
                              "graph": False, "observation": "full", "diagnostic_only": False,
                              "protocol_sha256": identity}
                    summary = analyze_online(trace, records, [], [], 12_000_000_000, 1000)
                    summary.update(**config, valid_measurement=True, source_unchanged=True,
                                   end_state_check_passed=True, end_ns=12_000_000_000,
                                   kv_blocks=302, kv_payload_bytes=10872)
                    for name, data in (("trace.json", trace), ("run_config.json", config),
                        ("summary.json", summary), ("environment.json", env),
                        ("requests.json", [asdict(r) for r in records]), ("steps.json", []), ("backlog.json", [])):
                        save(out/name, data)
                    seal(out)
            status = report(suite, protocol)
            self.assertEqual(status["paired_comparisons"], 18)
            self.assertTrue(status["all_pairs_valid"])
            self.assertEqual(status["generation_divergent_request_comparisons"], 1)
            self.assertFalse(status["ready_for_exploratory_capacity_scan"])
            self.assertFalse(status["v1_acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
