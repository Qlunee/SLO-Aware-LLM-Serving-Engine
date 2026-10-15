"""CPU-only refinement workflow and real scheduler online adapter contracts."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.v3_profile.protocol import read, save, seal, sources, predictor_module, TIMING_CONTRACT
from benchmarks.v3_profile.report import fit, evaluate
from benchmarks.v0.environment import model_metadata_hashes
from tools import v3_refine as refine
from tools.v3_online_profile import OnlineAdapter
from test_v3_profile import fixture_profile, features


def replace_json(path, value):
    path.write_text(json.dumps(value))
    manifest = path.parent/"artifact_hashes.json"
    if manifest.exists(): manifest.unlink()
    seal(path.parent)


def parent_fixture(root):
    model = root/"model"
    model.mkdir()
    save(model/"config.json", {"max_position_embeddings": 4096, "vocab_size": 1000})
    parent = root/"parent"
    parent.mkdir()
    identity = {"timing_contract": TIMING_CONTRACT,
        "fixture": "CPU fixtures are not GPU evidence", "source_sha256": sources(),
        "model_metadata_sha256": model_metadata_hashes(model), "model_revision": "fixture-revision"}
    profile = fixture_profile(parent/"train-411", "train", 411, identity)
    protocol = read(profile/"protocol.json")
    protocol["model_path"] = str(model)
    replace_json(profile/"protocol.json", protocol)
    (parent/"evaluation").mkdir()
    save(parent/"evaluation/report.json", {"coverage": .8643, "supported": {"max_underprediction_ms": 2.03}})
    seal(parent/"evaluation")
    return parent


class FreezeTests(unittest.TestCase):
    def test_tools_have_separate_freeze_and_original_source_set_is_unchanged(self):
        execution = sources()
        workflow = refine.workflow_sources()
        self.assertEqual(len(workflow), 4)
        self.assertTrue(set(workflow).isdisjoint(execution))

    def test_prepare_freezes_new_seeds_and_independent_online_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = parent_fixture(root)
            plan = refine.prepare(parent, root/"suite")
            self.assertEqual(plan["train_seeds"], [412, 413])
            self.assertEqual(plan["eval_seed"], 512)
            self.assertTrue(set(plan["online_seeds"].values()).isdisjoint([411, 412, 413, 512]))
            self.assertEqual(refine.load_suite(root/"suite"), plan)
            self.assertEqual(sources(), plan["execution_sources"])
            with self.assertRaisesRegex(ValueError, "empty"):
                refine.prepare(parent, root/"suite")

    def test_modified_trace_and_parent_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = parent_fixture(root)
            refine.prepare(parent, root/"suite")
            file = root/"suite/traces/burst.json"
            original = file.read_text()
            file.write_text(original+" ")
            with self.assertRaises(ValueError): refine.load_suite(root/"suite")
            file.write_text(original)
            steps = parent/"train-411/steps.jsonl"
            steps.write_text(steps.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "parent"):
                refine.load_suite(root/"suite")


class ResumeTests(unittest.TestCase):
    def test_valid_completed_stage_is_reused_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, calls = Path(tmp), []
            def launch(command, log):
                calls.append(command)
                out = Path(command[0])
                out.mkdir()
                save(out/"result.json", {"ok": True})
                seal(out)
            out = refine.stage(suite, "sample", lambda out: [str(out)], refine.verify_artifacts, launch)
            again = refine.stage(suite, "sample", lambda out: [str(out)], refine.verify_artifacts, launch)
            self.assertEqual(out, again)
            self.assertEqual(len(calls), 1)
            (out/"result.json").write_text("changed")
            with self.assertRaises(ValueError):
                refine.stage(suite, "sample", lambda out: [str(out)], refine.verify_artifacts, launch)

    def test_failed_attempt_is_kept_and_next_run_uses_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            def failed(command, log):
                out = Path(command[0])
                out.mkdir()
                save(out/"error.json", {"fixture_failure": True})
                seal(out)
                raise RuntimeError("simulated subprocess failure")
            with self.assertRaises(RuntimeError):
                refine.stage(suite, "sample", lambda out: [str(out)], refine.verify_artifacts, failed)
            def successful(command, log):
                out = Path(command[0])
                out.mkdir()
                save(out/"result.json", {"ok": True})
                seal(out)
            out = refine.stage(suite, "sample", lambda out: [str(out)], refine.verify_artifacts, successful)
            self.assertEqual(out.name, "attempt-002")
            self.assertTrue((suite/"artifacts/sample/attempt-001/error.json").exists())

    def test_complete_pipeline_and_resume_with_archived_cpu_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, calls = Path(tmp), []
            parent = parent_fixture(root)
            suite = root/"suite"
            plan = refine.prepare(parent, suite)
            def fake_launch(command, log):
                calls.append(command)
                out = Path(command[command.index("--out")+1])
                out.parent.mkdir(parents=True, exist_ok=True)
                if command[3] == "tools.v3_online_profile":
                    mode = command[command.index("--mode")+1]
                    pred = Path(command[command.index("--predictor")+1])
                    out.mkdir()
                    summary = refine.prediction_summary([{"actual_ms": 10, "features": features(),
                        "prediction": {"supported": True, "predicted_ms": 19, "reason": "exact_bucket"}}]*40)
                    save(out/"report.json", {"valid_measurement": True, "all_completed": True,
                        "predictor_sha256": refine.sha(pred), "trace_sha256": refine.sha(suite/f"traces/{mode}.json"),
                        "workflow_sources": plan["workflow_sources"], "window": summary, "drain": summary,
                        "request_statuses": {"completed": 2}, "preemptions": 0,
                        "fixture": "CPU archived output, not real GPU evidence"})
                    seal(out)
                elif command[4] == "collect":
                    split = command[command.index("--split")+1]
                    seed = int(command[command.index("--seed")+1])
                    fixture_profile(out, split, seed, plan["identity"])
                    protocol = read(out/"protocol.json")
                    protocol["model_path"] = plan["model_path"]
                    replace_json(out/"protocol.json", protocol)
                elif command[4] == "fit":
                    start, end = command.index("--profiles")+1, command.index("--min-samples")
                    fit(command[start:end], out, int(command[end+1]))
                elif command[4] == "evaluate":
                    evaluate(command[command.index("--predictor")+1],
                             [command[command.index("--profiles")+1]], out)
                else:
                    self.fail(command)
            with redirect_stdout(StringIO()):
                report = refine.run(suite, fake_launch)
            self.assertTrue(report["workflow_completed"])
            self.assertTrue(report["review_targets_met"])
            self.assertFalse(report["slack_implemented"])
            self.assertEqual(len(calls), 8)
            with redirect_stdout(StringIO()):
                again = refine.run(suite, fake_launch)
            self.assertEqual(report, again)
            self.assertEqual(len(calls), 8)


class PredictionSummaryTests(unittest.TestCase):
    def test_unsupported_and_empty_samples_are_retained(self):
        rows = [{"actual_ms": 10, "features": features(),
            "prediction": {"supported": False, "predicted_ms": 89, "reason": "out_of_coverage"}}]
        summary = refine.prediction_summary(rows)
        self.assertEqual(summary["coverage"], 0)
        self.assertEqual(summary["unsupported"]["count"], 1)
        self.assertIsNone(summary["supported"]["underprediction_rate"])
        self.assertIsNone(refine.prediction_summary([])["coverage"])


class OnlineAdapterTests(unittest.TestCase):
    def test_real_scheduler_has_one_timestamp_per_token_and_uses_visibility_boundary(self):
        import test_v1_state as state
        from benchmarks.v0.metrics import RequestRecord
        from benchmarks.v0.workload import RequestSpec
        state.setUpModule()
        try:
            config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512, eos=2,
                kvcache_block_size=256, num_kvcache_blocks=32, max_model_len=4096)
            scheduler = state.Scheduler(config, 256)
            def add(tokens, params):
                seq = state.Sequence(tokens, params)
                scheduler.add(seq)
                return seq.seq_id
            def call(method, batch, logits, verify, greedy):
                return {"tokens": {i: 1000 for i in batch.sample_seq_ids}, "graph_mode": "eager"}
            engine = SimpleNamespace(scheduler=scheduler, greedy=True, add_request=add,
                is_finished=scheduler.is_finished, model_runner=SimpleNamespace(call=call))
            predictor = SimpleNamespace(predict=lambda f:
                predictor_module().Prediction(10, False, 0, "out_of_coverage"))
            ticks = iter(range(0, 100_000_000, 1_000_000))
            adapter = OnlineAdapter(engine, lambda n: state.Params(max_tokens=n, ignore_eos=True),
                                    predictor, .002, .016, lambda: next(ticks))
            adapter.start_clock()
            spec = RequestSpec("a", 0, [100]*513, 2, 1000, 100)
            record = RequestRecord("a", 0, 513, 2, 1000, 100)
            adapter.submit(spec, record)
            for _ in range(10):
                if adapter.is_finished(): break
                adapter.step()
            self.assertTrue(adapter.is_finished())
            self.assertEqual(len(record.token_times_ns), 2)
            self.assertEqual(record.status, "completed")
            self.assertEqual(len(adapter.steps), 4)
            self.assertEqual(adapter.steps[0]["features"]["sample_count"], 0)
            self.assertEqual(record.token_times_ns[0], adapter.steps[2]["visible_ns"]-adapter.origin_ns)
            self.assertEqual(record.finish_ns, adapter.steps[-1]["visible_ns"]-adapter.origin_ns)
            self.assertTrue(any(r["window"] == "drain" for r in adapter.steps))
            scheduler.assert_consistent()
            self.assertFalse(scheduler.block_manager.used_block_ids)
        finally:
            state.tearDownModule()


if __name__ == "__main__":
    unittest.main()
