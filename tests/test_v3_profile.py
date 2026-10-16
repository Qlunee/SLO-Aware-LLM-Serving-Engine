"""V3 estimator, clock, split isolation and real scheduler CPU contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.v3_profile.protocol import (Settings, TIMING_CONTRACT, planned_cases,
    predictor_module, load_profile, save, seal, read)
from benchmarks.v3_profile.report import fit, evaluate, summarize_predictions, compatible_identity
from benchmarks.v3_profile.timing import timed_step

estimator = predictor_module()


def features(**kwargs):
    return {"decode_count": 1, "prefill_tokens": 0, "prefill_request_count": 0,
        "sum_context_tokens": 257, "max_context_len": 257, "sample_count": 1,
        "prefill_has_history": False, "graph_mode": "eager", **kwargs}


def sample(duration=10, warmup=False, **kwargs):
    return {"features": features(**kwargs), "step_time_ms": duration, "warmup": warmup}


class PredictorTests(unittest.TestCase):
    def test_p95_nearest_rank_and_warmup_exclusion(self):
        rows = [sample(i) for i in range(1, 101)] + [sample(9999, warmup=True)]
        table = estimator.fit_table(rows, {"backend": "bf16"}, 30)
        result = estimator.StepPredictor(table, table["identity"]).predict(features())
        self.assertEqual(result.predicted_ms, 95)
        self.assertTrue(result.supported)
        self.assertEqual(result.sample_count, 100)
        self.assertEqual(table["training_sample_count"], 100)

    def test_sparse_and_unseen_buckets_are_unsupported(self):
        table = estimator.fit_table([sample(10)]*30+[sample(100, max_context_len=1024, sum_context_tokens=1024)], {}, 30)
        predictor = estimator.StepPredictor(table, {})
        sparse = predictor.predict(features(max_context_len=1024, sum_context_tokens=1024))
        unseen = predictor.predict(features(max_context_len=4096, sum_context_tokens=4096))
        self.assertEqual((sparse.supported, sparse.reason, sparse.sample_count), (False, "sparse_bucket", 1))
        self.assertEqual((unseen.supported, unseen.reason, unseen.sample_count), (False, "out_of_coverage", 0))
        self.assertEqual(sparse.predicted_ms, 100)

    def test_backend_identity_must_match(self):
        table = estimator.fit_table([sample()]*30, {"kv_dtype": "bf16"})
        with self.assertRaisesRegex(ValueError, "identity"):
            estimator.StepPredictor(table, {"kv_dtype": "fp8"})

    def test_runtime_tuple_metadata_matches_json_loaded_identity(self):
        identity = {"nccl_version": (2, 27, 3)}
        table = estimator.fit_table([sample()]*30, identity)
        serialized = json.loads(json.dumps(table))
        self.assertTrue(estimator.StepPredictor(serialized, identity).predict(features()).supported)

    def test_sampling_history_and_graph_do_not_share_buckets(self):
        a = features(prefill_tokens=256, prefill_request_count=1, sum_context_tokens=513,
                     max_context_len=257)
        keys = [estimator.bucket_key(a), estimator.bucket_key({**a, "sample_count": 2}),
                estimator.bucket_key({**a, "prefill_has_history": True})]
        self.assertEqual(len(set(keys)), 3)
        self.assertNotEqual(estimator.bucket_key(features()), estimator.bucket_key(features(graph_mode="cuda_graph")))

    def test_round_up_bucket_boundaries(self):
        self.assertEqual(estimator.bucket_key(features(max_context_len=300, sum_context_tokens=300)),
                         estimator.bucket_key(features(max_context_len=512, sum_context_tokens=512)))
        self.assertNotEqual(estimator.bucket_key(features(max_context_len=512, sum_context_tokens=512)),
                            estimator.bucket_key(features(max_context_len=513, sum_context_tokens=513)))

    def test_optional_decode_and_sample_widths_cover_nearby_counts(self):
        widths = {"decode_count": 4, "sample_count": 4}
        fourth = features(decode_count=4, sample_count=4,
                          sum_context_tokens=1028)
        table = estimator.fit_table([sample(10)]*12 + [
            {"features": fourth, "step_time_ms": 15, "warmup": False}]*12,
            {}, 12, widths)
        prediction = estimator.StepPredictor(table, {}).predict(
            features(decode_count=2, sample_count=2, sum_context_tokens=514))
        self.assertTrue(prediction.supported)
        self.assertEqual(prediction.sample_count, 24)
        self.assertEqual(prediction.predicted_ms, 15)

    def test_invalid_features_and_timings_rejected(self):
        for bad in (features(decode_count=-1), features(sample_count=0),
                    features(prefill_tokens=256), features(prefill_has_history=True),
                    features(sum_context_tokens=1), features(max_context_len=True),
                    features(graph_mode="unknown")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimator.bucket_key(bad)
        for duration in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                estimator.fit_table([sample(duration)], {})


class ClockTests(unittest.TestCase):
    def test_schedule_model_commit_endpoint_and_flags(self):
        calls = []
        ticks = iter([100, 200, 500, 600])
        batch = object()
        result = {"tokens": {7: 42}, "graph_mode": "eager"}
        def schedule():
            calls.append("schedule")
            return batch
        def call(*args):
            calls.append(args)
            return result
        def postprocess(b, tokens):
            self.assertIs(b, batch)
            self.assertEqual(tokens, {7: 42})
            calls.append("postprocess")
        engine = SimpleNamespace(greedy=True,
            scheduler=SimpleNamespace(schedule=schedule, postprocess=postprocess),
            model_runner=SimpleNamespace(call=call))
        _, _, times = timed_step(engine, lambda: next(ticks))
        self.assertEqual(calls, ["schedule", ("run_mixed", batch, False, False, True), "postprocess"])
        self.assertEqual(times["step_time_ms"], .0005)
        self.assertEqual(times["visible_ns"], 600)
        self.assertEqual(times["schedule_ms"]+times["run_ms"]+times["postprocess_ms"], times["step_time_ms"])


def fixture_profile(path, split, seed, identity=None):
    """Synthetic archived-clock fixture; never presented as real GPU evidence."""
    identity = identity or {"timing_contract": TIMING_CONTRACT, "fixture": "CPU-not-GPU"}
    path.mkdir()
    settings = Settings(decode_counts=(1,), prefill_budgets=(0,), contexts=(256,),
                        prefill_request_counts=(1,), repeats=40, warmup=1)
    rows = []
    for i in range(41):
        duration = 10+(i % 10)
        start = i*100_000_000+seed
        rows.append({"step_id": i, "case_id": "d1-p0-m0-c256", "episode": i,
            "episode_step": 0, "warmup": i == 0, "features": features(),
            "start_ns": start, "scheduled_ns": start+100,
            "run_end_ns": start+duration*1_000_000-100,
            "visible_ns": start+duration*1_000_000, "step_time_ms": duration})
    save(path/"protocol.json", {"schema": "v3-step-profile-v1", "split": split,
        "seed": seed, "smoke_only": False, "identity": identity, "settings": asdict(settings)})
    save(path/"summary.json", {"valid": True, "recorded_steps": 41, "measured_steps": 40,
        "completed_cases": ["d1-p0-m0-c256"], "preemptions": 0,
        "source_unchanged": True, "state_and_drain_passed": True})
    (path/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    seal(path)
    return path


class ArtifactTests(unittest.TestCase):
    def test_workflow_source_change_keeps_profile_compatibility(self):
        old = {"model_revision": "same", "source_sha256": {
            "nanovllm/engine/mixed_scheduler.py": "v1",
            "benchmarks/v3_profile/timing.py": "clock",
            "benchmarks/v3_profile/driver.py": "old"}}
        new = json.loads(json.dumps(old))
        new["source_sha256"]["benchmarks/v3_profile/driver.py"] = "new"
        self.assertTrue(compatible_identity(old, new))
        new["source_sha256"]["nanovllm/engine/mixed_scheduler.py"] = "changed"
        self.assertFalse(compatible_identity(old, new))

    def test_exact_prefill_profile_round_trips_through_sealed_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"exact"
            folder.mkdir()
            settings = replace(Settings(), decode_counts=(0,),
                prefill_budgets=(512,), contexts=(256,),
                prefill_request_counts=(2,), prefill_shape="exact",
                repeats=2, warmup=1)
            case = planned_cases(settings)[0]["case_id"]
            rows = []
            for episode in range(3):
                for local_step in range(2):
                    step_id = len(rows)
                    start = step_id*100_000_000
                    shape = ({"decode_count": 0, "prefill_tokens": 512,
                              "prefill_request_count": 2, "sum_context_tokens": 512,
                              "max_context_len": 256, "sample_count": 2,
                              "prefill_has_history": False, "graph_mode": "eager"}
                             if local_step == 0 else
                             {"decode_count": 2, "prefill_tokens": 0,
                              "prefill_request_count": 0, "sum_context_tokens": 514,
                              "max_context_len": 257, "sample_count": 2,
                              "prefill_has_history": False, "graph_mode": "eager"})
                    rows.append({"step_id": step_id, "case_id": case,
                        "episode": episode, "episode_step": local_step,
                        "warmup": episode == 0, "features": shape,
                        "start_ns": start, "scheduled_ns": start+100,
                        "run_end_ns": start+9_999_900,
                        "visible_ns": start+10_000_000, "step_time_ms": 10})
            save(folder/"protocol.json", {"schema": "v3-step-profile-v1",
                "split": "train", "seed": 621, "smoke_only": False,
                "identity": {"timing_contract": TIMING_CONTRACT},
                "settings": asdict(settings)})
            save(folder/"summary.json", {"valid": True,
                "recorded_steps": 6, "measured_steps": 4,
                "completed_cases": [case], "preemptions": 0,
                "source_unchanged": True, "state_and_drain_passed": True})
            (folder/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            seal(folder)
            _, loaded = load_profile(folder)
            self.assertEqual(len(loaded), 6)

    def test_offline_fit_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = fixture_profile(root/"train", "train", 411)
            test = fixture_profile(root/"eval", "eval", 511)
            summary = fit([train], root/"predictor")
            self.assertEqual(summary["training_samples"], 40)
            report = evaluate(root/"predictor/predictor.json", [test], root/"report")
            self.assertEqual(report["coverage"], 1)
            self.assertEqual(report["supported"]["count"], 40)
            self.assertFalse(report["slo_scheduler_implemented"])
            self.assertGreaterEqual(report["predict_cpu_p95_us"], 0)
            with self.assertRaises(FileExistsError):
                fit([train], root/"predictor")

    def test_seed_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = fixture_profile(root/"train", "train", 411)
            test = fixture_profile(root/"eval", "eval", 411)
            fit([train], root/"predictor")
            with self.assertRaisesRegex(ValueError, "seeds overlap"):
                evaluate(root/"predictor/predictor.json", [test], root/"report")

    def test_smoke_not_accepted_as_training_and_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = fixture_profile(root/"train", "train", 411)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                fit([train, train], root/"predictor")
            data = read(train/"protocol.json")
            data["smoke_only"] = True
            (train/"protocol.json").write_text(json.dumps(data))
            (train/"artifact_hashes.json").unlink()
            seal(train)
            with self.assertRaisesRegex(ValueError, "non-smoke"):
                fit([train], root/"predictor")

    def test_changed_artifact_and_wrong_clock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = fixture_profile(Path(tmp)/"train", "train", 411)
            path = train/"steps.jsonl"
            original = path.read_text()
            path.write_text(original+" ")
            with self.assertRaises(ValueError):
                load_profile(train)
            rows = [json.loads(line) for line in original.splitlines()]
            rows[1]["step_time_ms"] = 9999
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            (train/"artifact_hashes.json").unlink()
            seal(train)
            with self.assertRaisesRegex(ValueError, "timestamp"):
                load_profile(train)

    def test_different_runtime_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = fixture_profile(root/"train", "train", 411)
            test = fixture_profile(root/"eval", "eval", 511,
                {"timing_contract": TIMING_CONTRACT, "fixture": "different-GPU"})
            fit([train], root/"predictor")
            with self.assertRaisesRegex(ValueError, "identity"):
                evaluate(root/"predictor/predictor.json", [test], root/"report")

    def test_missing_episode_cannot_be_hidden_by_resealing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = fixture_profile(Path(tmp)/"train", "train", 411)
            path = profile/"steps.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
            summary = read(profile/"summary.json")
            summary.update(recorded_steps=40, measured_steps=39)
            (profile/"summary.json").write_text(json.dumps(summary))
            (profile/"artifact_hashes.json").unlink()
            seal(profile)
            with self.assertRaisesRegex(ValueError, "missing case/episode"):
                load_profile(profile)

    def test_underprediction_metrics_include_zero_and_unsupported(self):
        rows = [{"actual_ms": a, "prediction": {"predicted_ms": p}} for a, p in ((20, 10), (10, 20))]
        report = summarize_predictions(rows)
        self.assertEqual(report["underprediction_rate"], .5)
        self.assertEqual(report["max_underprediction_ms"], 10)
        self.assertIsNone(summarize_predictions([])["underprediction_rate"])


class PlanningTests(unittest.TestCase):
    def test_default_matrix_has_three_phases_and_multiple_prefills(self):
        settings = Settings()
        settings.validate()
        cases = planned_cases(settings)
        self.assertEqual(len(cases), 24)
        self.assertTrue(any(c["decode_count"] and c["prefill_budget"] for c in cases))
        self.assertTrue(any(c["decode_count"] == 0 for c in cases))
        self.assertTrue(any(c["prefill_budget"] == 0 for c in cases))
        self.assertTrue(any(c["prefill_requests"] == 2 for c in cases))

    def test_invalid_capacity_context_and_budget_rejected(self):
        for settings in (replace(Settings(), token_budget=128), replace(Settings(), kv_block_cap=1),
                         replace(Settings(), max_model_len=512), replace(Settings(), repeats=0),
                         replace(Settings(), prefill_budgets=(257,)), replace(Settings(), max_num_seqs=1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()

    def test_exact_prefill_shape_plans_first_token_samples(self):
        settings = replace(Settings(), decode_counts=(0,), prefill_budgets=(512,),
                           contexts=(256,), prefill_request_counts=(2,),
                           prefill_shape="exact", repeats=2, warmup=1)
        settings.validate()
        self.assertEqual(len(planned_cases(settings)), 1)
        self.assertEqual(settings.prefill_shape, "exact")

    def test_alternating_decode_contexts_are_named_and_capacity_checked(self):
        settings = replace(Settings(), decode_counts=(4,), prefill_budgets=(256,),
                           contexts=(256,), prefill_request_counts=(1,),
                           alternate_context_tokens=1024, repeats=2, warmup=1)
        settings.validate()
        self.assertEqual(planned_cases(settings)[0]["case_id"], "d4-p256-m1-c256-a1024")
        self.assertEqual(planned_cases(settings)[0]["alternate_context_tokens"], 1024)
        with self.assertRaisesRegex(ValueError, "KV cap"):
            replace(settings, kv_block_cap=2).validate()


class SchedulerIntegrationTests(unittest.TestCase):
    def test_driver_bootstraps_alternating_decode_contexts(self):
        import test_v1_state as state
        from benchmarks.v3_profile.driver import run_case
        state.setUpModule()
        try:
            settings = replace(Settings(), decode_counts=(4,), prefill_budgets=(512,),
                               contexts=(256,), prefill_request_counts=(2,),
                               alternate_context_tokens=1024, repeats=1, warmup=1)
            case = planned_cases(settings)[0]
            config = SimpleNamespace(max_num_seqs=settings.max_num_seqs,
                max_num_batched_tokens=settings.token_budget, eos=2, kvcache_block_size=256,
                num_kvcache_blocks=settings.kv_block_cap, max_model_len=settings.max_model_len)
            scheduler = state.Scheduler(config, 256)
            def call(method, batch, logits, verify, greedy):
                return {"tokens": {i: 1000 for i in batch.sample_seq_ids}, "graph_mode": "eager"}
            engine = SimpleNamespace(scheduler=scheduler, greedy=True,
                model_runner=SimpleNamespace(call=call), is_finished=scheduler.is_finished)
            engine.step = lambda: timed_step(engine)
            lengths = []
            def add(length, output):
                lengths.append(length)
                seq = state.Sequence([100+len(lengths)]*length,
                                     state.Params(max_tokens=output, ignore_eos=True))
                scheduler.add(seq)
                return seq.seq_id
            observed = []
            run_case(engine, case, settings, add,
                     lambda _c, _e, _s, _w, batch, result, _t:
                     observed.append(estimator.batch_features(batch, result["graph_mode"])))
            self.assertEqual(lengths[:4], [256, 1024, 256, 1024])
            self.assertEqual(len(observed), 8)
            self.assertTrue(any(f["prefill_request_count"] == 2 and f["prefill_has_history"]
                                for f in observed))
        finally:
            state.tearDownModule()

    def test_controlled_driver_uses_real_scheduler_and_drains_every_case(self):
        import test_v1_state as state
        from benchmarks.v3_profile.driver import run_case
        state.setUpModule()
        try:
            settings = replace(Settings(), decode_counts=(0, 1, 4), contexts=(256,),
                               prefill_budgets=(0, 256), repeats=2, warmup=1)
            config = SimpleNamespace(max_num_seqs=settings.max_num_seqs,
                max_num_batched_tokens=settings.token_budget, eos=2, kvcache_block_size=256,
                num_kvcache_blocks=settings.kv_block_cap, max_model_len=settings.max_model_len)
            scheduler = state.Scheduler(config, 256)
            call_count = 0
            def call(method, batch, logits, verify, greedy):
                nonlocal call_count
                call_count += 1
                self.assertEqual((method, logits, verify, greedy), ("run_mixed", False, False, True))
                return {"tokens": {i: 1000 for i in batch.sample_seq_ids}, "graph_mode": "eager"}
            engine = SimpleNamespace(scheduler=scheduler, greedy=True,
                model_runner=SimpleNamespace(call=call), is_finished=scheduler.is_finished)
            engine.step = lambda: timed_step(engine)
            serial = 0
            def add(length, output):
                nonlocal serial
                serial += 1
                seq = state.Sequence([100+serial]*length, state.Params(max_tokens=output, ignore_eos=True))
                scheduler.add(seq)
                return seq.seq_id
            measured = []
            for case in planned_cases(settings):
                observed = []
                def observe(c, episode, local_step, warmup, batch, result, times):
                    scheduler.assert_consistent()
                    observed.append((warmup, estimator.batch_features(batch, result["graph_mode"])))
                run_case(engine, case, settings, add, observe)
                expected = (settings.repeats+settings.warmup)*(4 if case["prefill_budget"] else 1)
                self.assertEqual(len(observed), expected)
                self.assertEqual(sum(not w for w, _ in observed), settings.repeats*(4 if case["prefill_budget"] else 1))
                self.assertFalse(scheduler.sequences)
                measured.extend(f for w, f in observed if not w)
            self.assertGreater(call_count, len(measured))  # Bootstrap/drain excluded.
            self.assertTrue(any(f["prefill_has_history"] for f in measured))
            self.assertTrue(any(f["prefill_tokens"] and f["sample_count"] == 0 for f in measured))
            self.assertTrue(any(f["decode_count"] and f["prefill_request_count"] == 2 for f in measured))
        finally:
            state.tearDownModule()

    def test_exact_prefill_shape_samples_multiple_first_tokens(self):
        import test_v1_state as state
        from benchmarks.v3_profile.driver import run_case
        state.setUpModule()
        try:
            settings = replace(Settings(), decode_counts=(0,),
                               prefill_budgets=(512,), contexts=(256,),
                               prefill_request_counts=(2,), prefill_shape="exact",
                               repeats=2, warmup=1)
            config = SimpleNamespace(max_num_seqs=settings.max_num_seqs,
                max_num_batched_tokens=settings.token_budget, eos=2,
                kvcache_block_size=256, num_kvcache_blocks=settings.kv_block_cap,
                max_model_len=settings.max_model_len)
            scheduler = state.Scheduler(config, 256)
            engine = SimpleNamespace(scheduler=scheduler, greedy=True,
                model_runner=SimpleNamespace(call=lambda method, batch, logits, verify, greedy:
                    {"tokens": {i: 1000 for i in batch.sample_seq_ids}, "graph_mode": "eager"}),
                is_finished=scheduler.is_finished)
            engine.step = lambda: timed_step(engine)
            serial = 0
            def add(length, output):
                nonlocal serial
                serial += 1
                seq = state.Sequence([100+serial]*length,
                                     state.Params(max_tokens=output, ignore_eos=True))
                scheduler.add(seq)
                return seq.seq_id
            observed = []
            run_case(engine, planned_cases(settings)[0], settings, add,
                lambda case, episode, local_step, warmup, batch, result, times:
                    observed.append((local_step, estimator.batch_features(batch, result["graph_mode"]))))
            self.assertEqual(len(observed), 2*(settings.warmup+settings.repeats))
            first_steps = [f for step, f in observed if step == 0]
            self.assertTrue(all(f["prefill_tokens"] == 512 and
                                f["prefill_request_count"] == 2 and
                                f["sample_count"] == 2 and
                                not f["prefill_has_history"] for f in first_steps))
        finally:
            state.tearDownModule()


if __name__ == "__main__":
    unittest.main()
