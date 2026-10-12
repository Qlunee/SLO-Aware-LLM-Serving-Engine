import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from diagnostics.v2_bounded.warmup import build_warmup


class WarmupContractTests(unittest.TestCase):
    def test_warmup_is_eight_fixed_requests_not_the_measured_trace(self):
        trace = build_warmup(256, 1024, "revision", lambda n, i: [i + 1] * n)
        self.assertEqual(len(trace["requests"]), 8)
        self.assertEqual([len(r["prompt_token_ids"]) for r in trace["requests"]],
                         [256] * 4 + [1024] * 4)
        self.assertEqual({r["arrival_ns"] for r in trace["requests"]}, {0})
        self.assertEqual({r["max_output_tokens"] for r in trace["requests"]}, {16})


class FreezeTests(unittest.TestCase):
    def test_frozen_protocol_has_54_jobs_and_detects_trace_tampering(self):
        from diagnostics.v2_bounded.protocol import prepare, load_suite
        from benchmarks.v2_stall.protocol import save
        from benchmarks.v2_baselines.workload import Workload
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            save(model / "config.json", {"max_position_embeddings": 4096})
            suite = root / "suite"
            prepare(suite, model, "rev", lambda n, seed, i: [i % 10 + 1] * n,
                    workload=Workload(warmup_s=2, measurement_s=2), root=root)
            p = load_suite(suite, root=root)
            self.assertEqual(len(p["jobs"]), 54)
            self.assertEqual(len(p["trace_sha256"]), 18)
            self.assertEqual(p["warmup_rule"]["request_count"], 8)
            trace = suite / p["jobs"][0]["trace"]
            trace.write_text("{}")
            with self.assertRaises(ValueError):
                load_suite(suite, root=root)

    def test_serving_source_change_requires_a_new_protocol(self):
        from diagnostics.v2_bounded.protocol import prepare, load_suite
        from benchmarks.v2_stall.protocol import save
        from benchmarks.v2_baselines.workload import Workload
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmarks").mkdir()
            source = root / "benchmarks/fixture.py"
            source.write_text("VERSION = 1\n")
            model = root / "model"
            model.mkdir()
            save(model / "config.json", {"max_position_embeddings": 4096})
            suite = root / "suite"
            prepare(suite, model, "rev", lambda n, seed, i: [1] * n,
                    workload=Workload(warmup_s=2, measurement_s=2), root=root)
            source.write_text("VERSION = 2\n")
            with self.assertRaisesRegex(ValueError, "source/tool set changed"):
                load_suite(suite, root=root)


class CPUDeviceEngine:
    """GPU boundary fixture: real scheduler/pages/driver, constant-token device."""
    def __init__(self, fail=False):
        from nanovllm.engine.fixed_scheduler import FixedScheduler
        self.config = SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=4096,
            max_model_len=4096, num_kvcache_blocks=64, kvcache_block_size=256,
            eos=-1, hf_config=SimpleNamespace(vocab_size=1000))
        self.scheduler = FixedScheduler(self.config, 256, "chunk", clock=self.now)
        self.model_runner = SimpleNamespace(kv_cache=object())
        self.by_seq, self.history, self.steps = {}, {}, []
        self.step_count = self.total_preemptions = self.timestamp = 0
        self.last_step = None
        self.fail = fail
        self.step_ns = 1_000_000

    def start_clock(self):
        self.timestamp = 0

    def now(self):
        return self.timestamp

    def sleep(self, seconds):
        self.timestamp += max(1, round(seconds * 1e9))

    def submit(self, spec, record):
        from nanovllm.engine.mixed_sequence import MixedSequence
        from nanovllm.sampling_params import SamplingParams
        seq = MixedSequence(spec.prompt_token_ids,
            SamplingParams(max_tokens=spec.max_output_tokens, ignore_eos=True))
        self.scheduler.add(seq)
        record.seq_id, record.enqueue_ns, record.status = seq.seq_id, self.now(), "queued"
        self.by_seq[seq.seq_id] = record

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        if self.fail:
            raise RuntimeError("device boundary failure")
        start = self.now()
        batch = self.scheduler.schedule()
        self.timestamp += self.step_ns
        self.scheduler.postprocess(batch, {i: 7 for i in batch.sample_seq_ids})
        self.step_count += 1
        self.steps.append({"phase": "mixed" if batch.num_decode_tokens and batch.num_prefill_tokens
                           else "decode" if batch.pure_decode else "prefill",
                           "start_ns": start, "end_ns": self.now(),
                           "used_blocks": len(self.scheduler.block_manager.used_block_ids),
                           "total_blocks": self.config.num_kvcache_blocks})
        for row in batch.requests:
            seq, r = self.scheduler.sequences[row.seq_id], self.by_seq[row.seq_id]
            if r.first_scheduled_ns is None:
                r.first_scheduled_ns = self.now()
            if row.should_sample:
                r.output_token_ids.append(7)
                r.token_times_ns.append(self.now())
            if seq.is_finished:
                r.status, r.finish_ns = "completed", self.now()

    def snapshot_outputs(self):
        pass

    def close_waiting_spans(self, timestamp):
        pass

    def exit(self):
        pass


class WarmupExecutionTests(unittest.TestCase):
    def test_completed_warmup_resets_cold_pages_and_observations_without_reallocation(self):
        from diagnostics.v2_bounded.warmup import prewarm
        engine = CPUDeviceEngine()
        pool = engine.model_runner.kv_cache
        trace = build_warmup(256, 1024, "rev", lambda n, i: [i + 1] * n)
        with tempfile.TemporaryDirectory() as tmp:
            report = prewarm(engine, engine.config, trace, Path(tmp) / "warmup")
        self.assertTrue(report["passed"])
        self.assertIs(engine.model_runner.kv_cache, pool)
        self.assertEqual(engine.by_seq, {})
        self.assertEqual(engine.steps, [])
        self.assertEqual(engine.scheduler.sequences, {})
        self.assertEqual(engine.scheduler.wait_since, {})
        self.assertEqual(engine.scheduler.block_manager.hash_to_block_id, {})
        self.assertEqual(len(engine.scheduler.block_manager.free_block_ids), 64)
        self.assertEqual(engine.step_count, 0)

    def test_warmup_preserves_internal_device_error_and_never_proceeds(self):
        from diagnostics.v2_bounded.warmup import prewarm
        from benchmarks.v2_stall.protocol import read, verify_artifacts
        engine = CPUDeviceEngine(fail=True)
        trace = build_warmup(256, 1024, "rev", lambda n, i: [i + 1] * n)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "warmup"
            with self.assertRaises(RuntimeError):
                prewarm(engine, engine.config, trace, out)
            verify_artifacts(out)
            report = read(out / "report.json")
            self.assertFalse(report["passed"])
            self.assertIn("device boundary failure", report["driver_error"])


class ResumeTests(unittest.TestCase):
    def test_sealed_failure_is_preserved_and_not_automatically_retried(self):
        from diagnostics.v2_bounded.batch import resume_decision
        from diagnostics.v2_bounded.protocol import prepare
        from diagnostics.v2_bounded.run import identity
        from benchmarks.v2_stall.protocol import save, seal
        from benchmarks.v2_baselines.workload import Workload
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            save(model / "config.json", {"max_position_embeddings": 4096})
            suite = root / "suite"
            p = prepare(suite, model, "rev", lambda n, seed, i: [i % 10 + 1] * n,
                        workload=Workload(warmup_s=2, measurement_s=2), root=root)
            job = p["jobs"][0]
            out = suite / "runs" / job["id"] / "attempt-0001"
            out.mkdir(parents=True)
            save(out / "identity.json", identity(suite, p, job))
            save(out / "outcome.json", {"status": "SETUP_FAILED", "exception": "fixture"})
            seal(out)
            decision = resume_decision(suite, job, p, root=root)
            self.assertEqual(decision["action"], "PRESERVE")
            self.assertEqual(decision["attempts"][0]["status"], "SETUP_FAILED")
            self.assertEqual(list(out.parent.iterdir()), [out])

    def test_partial_attempt_is_retained_without_automatic_retry(self):
        from diagnostics.v2_bounded.batch import resume_decision
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            job = {"id": "job"}
            out = suite / "runs/job/attempt-0001"
            out.mkdir(parents=True)
            decision = resume_decision(suite, job, {}, root=suite)
            self.assertEqual(decision["action"], "PRESERVE")
            self.assertEqual(decision["attempts"][0]["status"], "PARTIAL")

    def test_orphan_process_log_is_partial_not_a_fresh_job(self):
        from diagnostics.v2_bounded.batch import resume_decision
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            base = suite / "runs/job"
            base.mkdir(parents=True)
            (base / "attempt-0001.log").write_text("killed before artifact creation")
            decision = resume_decision(suite, {"id": "job"}, {}, root=suite)
            self.assertEqual(decision["action"], "PRESERVE")
            self.assertEqual(decision["attempts"][0]["status"], "PARTIAL")


class CPUBackend:
    """External GPU/time boundary only; no mocking protocol, driver or metrics."""
    def check_and_seed(self, seed):
        pass

    def environment(self, protocol):
        return {"model_metadata_sha256": protocol["model_metadata_sha256"],
                "source_sha256": {k: v for k, v in protocol["source_sha256"].items()
                                  if k.startswith("nanovllm/")},
                "model_revision_declared": protocol["model"]["revision"],
                "packages": {"device": "CPU test fixture"}, "gpus": [],
                "torch_cuda_version": None}

    def create(self, protocol, policy):
        engine = CPUDeviceEngine()
        engine.scheduler.policy = "decode_first" if policy == "decode_first" else "chunk"
        engine.scheduler.chunk_size = int(policy.split("-")[1]) if policy.startswith("chunk-") else 256
        engine.model_runner.kv_cache = FakeKVPool()
        return engine, engine.config

    def synchronize(self):
        pass

    def reset_peak(self):
        pass

    def peaks(self):
        return {"torch_peak_allocated_bytes": 0, "torch_peak_reserved_bytes": 0}


class FakeKVPool:
    dtype = "fixture-bfloat16"

    def numel(self):
        return 4096

    def element_size(self):
        return 2


class SingleRunTests(unittest.TestCase):
    def setUp(self):
        from diagnostics.v2_bounded.protocol import prepare
        from benchmarks.v2_baselines.workload import Workload
        from benchmarks.v2_stall.protocol import save
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        save(model / "config.json", {"max_position_embeddings": 4096})
        self.suite = self.root / "suite"
        self.protocol = prepare(self.suite, model, "rev", lambda n, seed, i: [i % 10 + 1] * n,
            workload=Workload(warmup_s=2, measurement_s=2), root=self.root)
        self.job = self.protocol["jobs"][0]
        self.out = self.suite / "runs" / self.job["id"] / "attempt-0001"

    def test_run_audit_and_resume_use_raw_records_and_do_not_overwrite(self):
        from diagnostics.v2_bounded.protocol import prepare
        from diagnostics.v2_bounded.run import run_job
        from diagnostics.v2_bounded.audit import audit_attempt
        from diagnostics.v2_bounded.batch import resume_decision
        from benchmarks.v2_baselines.workload import Workload
        from benchmarks.v2_stall.protocol import save
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            save(model / "config.json", {"max_position_embeddings": 4096})
            suite = root / "suite"
            p = prepare(suite, model, "rev", lambda n, seed, i: [i % 10 + 1] * n,
                        workload=Workload(warmup_s=2, measurement_s=2), root=root)
            job = p["jobs"][0]
            out = suite / "runs" / job["id"] / "attempt-0001"
            self.assertEqual(run_job(suite, job["id"], out, device=CPUBackend(), root=root), 0)
            audited = audit_attempt(suite, job, out, p, root=root)
            self.assertEqual(audited["status"], "VALID_COMPLETE")
            self.assertEqual(resume_decision(suite, job, p, root=root)["action"], "PRESERVE")
            with self.assertRaises(FileExistsError):
                run_job(suite, job["id"], out, device=CPUBackend(), root=root)

    def test_valid_incomplete_is_not_a_program_failure(self):
        from diagnostics.v2_bounded.run import run_job
        from diagnostics.v2_bounded.audit import audit_attempt

        class SlowDevice(CPUBackend):
            def create(self, protocol, policy):
                engine, config = super().create(protocol, policy)
                engine.step_ns = 2_000_000_000
                return engine, config

        code = run_job(self.suite, self.job["id"], self.out, device=SlowDevice(), root=self.root)
        self.assertEqual(code, 2)
        audited = audit_attempt(self.suite, self.job, self.out, self.protocol, root=self.root)
        self.assertEqual(audited["status"], "VALID_INCOMPLETE")
        self.assertTrue(audited["valid_for_performance"])

    def test_raw_metric_tampering_is_caught_even_if_artifacts_are_resealed(self):
        import json
        from diagnostics.v2_bounded.run import run_job
        from diagnostics.v2_bounded.audit import audit_attempt
        from benchmarks.v2_stall.protocol import read, seal
        run_job(self.suite, self.job["id"], self.out, device=CPUBackend(), root=self.root)
        summary = read(self.out / "summary.json")
        summary["window_throughput_tok_s"] += 100
        (self.out / "summary.json").write_text(json.dumps(summary))
        (self.out / "artifact_hashes.json").unlink()
        seal(self.out)
        with self.assertRaisesRegex(ValueError, "raw metric audit"):
            audit_attempt(self.suite, self.job, self.out, self.protocol, root=self.root)

    def test_batch_resume_reuses_audited_success_and_leaves_failures_intact(self):
        from diagnostics.v2_bounded.batch import run_batch
        from diagnostics.v2_bounded.run import run_job

        def launch(command, log):
            job_id = command[command.index("--job") + 1]
            out = Path(command[command.index("--out") + 1])
            code = run_job(self.suite, job_id, out, device=CPUBackend(), root=self.root)
            log.write_text("CPU device/process boundary fixture")
            return code

        first = run_batch(self.suite, job_ids=[self.job["id"]], launch=launch, root=self.root)
        second = run_batch(self.suite, job_ids=[self.job["id"]], launch=launch, root=self.root)
        self.assertEqual(first["launched"], 1)
        self.assertEqual(second["launched"], 0)
        self.assertEqual(len(list(self.out.parent.iterdir())), 2)  # attempt + log

    def test_report_marks_missing_jobs_incomplete_instead_of_claiming_success(self):
        from diagnostics.v2_bounded.report import report_suite
        from benchmarks.v2_stall.protocol import verify_artifacts
        out, summary = report_suite(self.suite, root=self.root)
        verify_artifacts(out)
        self.assertEqual(summary["expected_jobs"], 54)
        self.assertEqual(summary["missing_jobs"], 54)
        self.assertFalse(summary["measurement_matrix_complete"])
        self.assertFalse(summary["v1_acceptance_passed"])

    def test_paired_report_retains_all_request_comparisons(self):
        from diagnostics.v2_bounded.report import report_suite
        from diagnostics.v2_bounded.run import run_job
        from benchmarks.v2_stall.protocol import read
        jobs = [j for j in self.protocol["jobs"] if j["rps"] == 6 and j["seed"] == 71
                and j["mode"] == "poisson" and j["policy"] in ("baseline", "chunk-256")]
        for job in jobs:
            out = self.suite / "runs" / job["id"] / "attempt-0001"
            run_job(self.suite, job["id"], out, device=CPUBackend(), root=self.root)
        out, summary = report_suite(self.suite, root=self.root)
        pairs = read(out / "pairs.json")
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["valid_for_window_comparison"])
        trace = read(self.suite / jobs[0]["trace"])
        self.assertEqual(len(pairs[0]["generation_comparisons"]), trace["request_count"])
        self.assertFalse(summary["measurement_matrix_complete"])

    def test_failed_job_does_not_block_other_jobs_or_get_automatically_retried(self):
        from diagnostics.v2_bounded.batch import run_batch
        from diagnostics.v2_bounded.run import run_job
        first, second = self.protocol["jobs"][:2]

        class BrokenDevice(CPUBackend):
            def create(self, protocol, policy):
                engine, config = super().create(protocol, policy)
                engine.fail = True
                return engine, config

        def launch(command, log):
            job_id = command[command.index("--job") + 1]
            out = Path(command[command.index("--out") + 1])
            device = BrokenDevice() if job_id == first["id"] else CPUBackend()
            code = run_job(self.suite, job_id, out, device=device, root=self.root)
            log.write_text("CPU process boundary")
            return code

        ids = [first["id"], second["id"]]
        result = run_batch(self.suite, job_ids=ids, launch=launch, root=self.root)
        self.assertEqual(result["launched"], 2)
        self.assertEqual(result["jobs"][0]["attempts"][0]["status"], "WARMUP_FAILED")
        self.assertEqual(result["jobs"][1]["attempts"][0]["status"], "VALID_COMPLETE")
        resumed = run_batch(self.suite, job_ids=ids, launch=launch, root=self.root)
        self.assertEqual(resumed["launched"], 0)


if __name__ == "__main__":
    unittest.main()
