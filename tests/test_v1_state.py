"""CPU-only state tests. Load the real BlockManager without importing CUDA package init."""
import importlib.util
import pickle
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setUpModule():
    # Never execute nanovllm/__init__.py: it imports the full GPU engine.
    global saved_modules, Sequence, Params, Scheduler, build_batch
    saved_modules = dict(sys.modules)
    for name in ["nanovllm", "nanovllm.engine"]:
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / name.replace(".", "/"))]
        sys.modules[name] = module
    config = types.ModuleType("nanovllm.config")
    config.Config = SimpleNamespace
    sys.modules[config.__name__] = config
    Params = load("nanovllm.sampling_params", "nanovllm/sampling_params.py").SamplingParams
    load("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
    Sequence = load("nanovllm.engine.mixed_sequence", "nanovllm/engine/mixed_sequence.py").MixedSequence
    load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
    load("nanovllm.engine.scheduler", "nanovllm/engine/scheduler.py")
    build_batch = load("nanovllm.engine.batch_metadata", "nanovllm/engine/batch_metadata.py").build_batch
    Scheduler = load("nanovllm.engine.mixed_scheduler", "nanovllm/engine/mixed_scheduler.py").MixedScheduler


def tearDownModule():
    for name in list(sys.modules):
        if name.startswith("nanovllm"):
            if name in saved_modules:
                sys.modules[name] = saved_modules[name]
            else:
                del sys.modules[name]


class StateTests(unittest.TestCase):
    def make(self, blocks=32, budget=512, chunk=256, max_seqs=32):
        config = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
                                 eos=2, kvcache_block_size=256, num_kvcache_blocks=blocks,
                                 max_model_len=4096)
        return Scheduler(config, chunk)

    def seq(self, length, output=4, tokens=None):
        return Sequence(tokens if tokens is not None else [100 + i % 67 for i in range(length)],
                        Params(max_tokens=output, ignore_eos=True))

    def commit(self, scheduler, batch):
        scheduler.postprocess(batch, {i: 1000 for i in batch.sample_seq_ids})
        scheduler.assert_consistent()

    def drain(self, scheduler, limit=100):
        for _ in range(limit):
            if scheduler.is_finished():
                break
            self.commit(scheduler, scheduler.schedule())
        self.assertTrue(scheduler.is_finished(), "scheduler did not drain")
        self.assertFalse(scheduler.block_manager.used_block_ids)

    def test_intermediate_chunk_does_not_sample(self):
        scheduler = self.make()
        seq = self.seq(513)
        scheduler.add(seq)
        batch = scheduler.schedule()
        self.assertFalse(batch.sample_indices)
        self.assertEqual(seq.num_cached_tokens, 0)  # not committed at schedule time
        self.commit(scheduler, batch)
        self.assertEqual(seq.num_cached_tokens, 256)
        self.assertEqual(seq.num_completion_tokens, 0)
        self.drain(scheduler)
        self.assertEqual(seq.num_completion_tokens, 4)

    def test_real_mixed_batch_and_flat_offsets(self):
        scheduler = self.make()
        first = self.seq(256)
        scheduler.add(first)
        self.commit(scheduler, scheduler.schedule())
        second = self.seq(513, tokens=[201] * 513)
        scheduler.add(second)
        batch = scheduler.schedule()
        self.assertEqual([r.phase for r in batch.requests], ["decode", "prefill"])
        self.assertEqual(batch.query_offsets, (0, 1, 257))
        self.assertEqual(batch.sample_indices, (0,))
        self.assertEqual(batch.positions[:3], (256, 0, 1))
        self.commit(scheduler, batch)
        self.drain(scheduler)

    def test_global_budget_can_cut_request_chunk(self):
        scheduler = self.make(budget=257, chunk=256)
        for _ in range(2):
            scheduler.add(self.seq(513))
        batch = scheduler.schedule()
        self.assertEqual([r.num_scheduled_tokens for r in batch.requests], [256, 1])
        self.assertEqual(len(batch.input_ids), 257)
        self.commit(scheduler, batch)
        self.drain(scheduler)

    def test_prefix_hash_is_not_published_before_commit(self):
        scheduler = self.make()
        seq = self.seq(513)
        scheduler.add(seq)
        batch = scheduler.schedule()
        self.assertEqual(scheduler.block_manager.blocks[seq.block_table[0]].hash, -1)
        self.commit(scheduler, batch)
        self.assertNotEqual(scheduler.block_manager.blocks[seq.block_table[0]].hash, -1)

    def test_cold_and_hot_prefix_reuse(self):
        scheduler = self.make()
        cold = self.seq(513, output=1)
        scheduler.add(cold)
        self.drain(scheduler)
        hot = self.seq(513, output=1)
        scheduler.add(hot)
        batch = scheduler.schedule()
        self.assertEqual(batch.requests[0].start_pos, 512)
        self.assertEqual(batch.positions, (512,))
        self.commit(scheduler, batch)
        self.drain(scheduler)

    def test_preemption_recompute_without_duplicate_outputs(self):
        scheduler = self.make(blocks=2, budget=512)
        # Different token prefixes avoid a cache hit during recovery.
        a, b = self.seq(256), self.seq(256, tokens=[200] * 256)
        scheduler.add(a)
        scheduler.add(b)
        self.commit(scheduler, scheduler.schedule())
        self.assertEqual(a.num_completion_tokens, 1)
        self.assertEqual(b.num_completion_tokens, 1)
        batch = scheduler.schedule()
        self.assertGreater(scheduler.preemption_count, 0)
        self.assertEqual(b.num_completion_tokens, 1)
        self.assertEqual(b.num_cached_tokens, 0)
        self.assertEqual(b.computed_high_water, 256)
        self.commit(scheduler, batch)
        saw_recompute = False
        for _ in range(100):
            if scheduler.is_finished():
                break
            batch = scheduler.schedule()
            saw_recompute |= any(r.phase == "recompute" for r in batch.requests)
            self.commit(scheduler, batch)
        self.assertTrue(saw_recompute)
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(scheduler.block_manager.used_block_ids)
        self.assertEqual((a.num_completion_tokens, b.num_completion_tokens), (4, 4))

    def test_max_num_seqs_keeps_unselected_requests(self):
        scheduler = self.make(max_seqs=1)
        for _ in range(3):
            scheduler.add(self.seq(3))
        self.assertEqual(len(scheduler.schedule().requests), 1)
        self.commit(scheduler, scheduler.pending)
        self.drain(scheduler)

    def test_decode_budget_round_robin(self):
        scheduler = self.make(budget=2, max_seqs=2)
        for i in range(3):
            seq = self.seq(1, output=3, tokens=[100 + i])
            scheduler.add(seq)
        self.drain(scheduler)

    def test_finish_at_page_boundary_releases_all_refs(self):
        scheduler = self.make()
        scheduler.add(self.seq(255, output=3))
        self.drain(scheduler)

    def test_oversized_request_is_rejected(self):
        scheduler = self.make(blocks=1)
        with self.assertRaises(ValueError):
            scheduler.add(self.seq(256, output=2))
        self.assertTrue(scheduler.is_finished())

    def test_cannot_reschedule_or_double_commit(self):
        scheduler = self.make()
        scheduler.add(self.seq(3))
        batch = scheduler.schedule()
        with self.assertRaises(RuntimeError):
            scheduler.schedule()
        self.commit(scheduler, batch)
        with self.assertRaises(RuntimeError):
            scheduler.postprocess(batch, {})

    def test_missing_samples_fail_before_state_commit(self):
        scheduler = self.make()
        seq = self.seq(3)
        scheduler.add(seq)
        batch = scheduler.schedule()
        with self.assertRaises(ValueError):
            scheduler.postprocess(batch, {})
        self.assertEqual(seq.num_cached_tokens, 0)
        self.commit(scheduler, batch)

    def test_tp_payload_roundtrip_contains_real_tokens(self):
        scheduler = self.make()
        scheduler.add(self.seq(257))
        batch = scheduler.schedule()
        restored = pickle.loads(pickle.dumps(batch))
        self.assertEqual(restored, batch)
        restored.validate()
        self.assertLess(len(pickle.dumps(["run_mixed", batch, False, True, True])) + 4, 2**20)

    def test_bad_sampling_metadata_is_rejected(self):
        scheduler = self.make()
        scheduler.add(self.seq(513))
        batch = scheduler.schedule()
        with self.assertRaises(ValueError):
            replace(batch, sample_indices=(0,), sample_seq_ids=(batch.requests[0].seq_id,),
                    temperatures=(1.0,)).validate()

    def test_batch_chunk_grid(self):
        for batch_size in [1, 2, 8, 32]:
            for chunk in [256, 512, 1024, 2048]:
                with self.subTest(batch=batch_size, chunk=chunk):
                    scheduler = self.make(blocks=128, budget=2048, chunk=chunk)
                    for i in range(batch_size):
                        scheduler.add(self.seq(513, output=2, tokens=[100 + i] * 513))
                    self.drain(scheduler)

    def test_blocked_head_does_not_hide_allocated_partial(self):
        scheduler = self.make(blocks=3, budget=256)
        partial = self.seq(513, output=2)
        scheduler.add(partial)
        self.commit(scheduler, scheduler.schedule())
        blocked = self.seq(513, output=2, tokens=[220] * 513)
        scheduler.add(blocked)
        scheduler.waiting.remove(blocked)
        scheduler.waiting.appendleft(blocked)
        batch = scheduler.schedule()
        self.assertEqual(batch.requests[0].seq_id, partial.seq_id)
        self.commit(scheduler, batch)
        self.drain(scheduler)

    def test_active_prefix_refcounts(self):
        scheduler = self.make()
        first = self.seq(513, output=4)
        scheduler.add(first)
        for _ in range(2):
            self.commit(scheduler, scheduler.schedule())
        shared = first.block_table[:2]
        second = self.seq(513, output=1)
        scheduler.add(second)
        batch = scheduler.schedule()
        self.assertEqual(second.block_table[:2], shared)
        self.assertTrue(all(scheduler.block_manager.blocks[b].ref_count == 2 for b in shared))
        self.commit(scheduler, batch)
        self.drain(scheduler)


class NumericPolicyTests(unittest.TestCase):
    def test_prefill_overlap_uses_half_open_intervals(self):
        from benchmarks.v1.compare_runs import prefill_overlap_itls
        records = [{"token_times_ns": [0, 10, 20, 30, 40]}]
        steps = [{"start_ns": 10, "end_ns": 20, "prefill_tokens": 256}]
        # [0,10] and [20,30] only touch a boundary and do not overlap.
        self.assertEqual(prefill_overlap_itls(records, steps, 0, 40)["count"], 1)
        self.assertEqual(prefill_overlap_itls(records, steps, 0, 20)["count"], 0)

    def test_bf16_does_not_hide_elementwise_failure(self):
        from benchmarks.v1.numerics import numerical_pass
        row = dict(top1_match=True, allclose=False, relative_l2=0.01,
                   cosine=0.9999, max_abs_error=0.375)
        self.assertTrue(numerical_pass(row, "bf16"))
        self.assertFalse(numerical_pass(row, "strict"))

    def test_bf16_rejects_errors_and_nonfinite(self):
        from benchmarks.v1.numerics import numerical_pass
        row = dict(top1_match=True, allclose=False, relative_l2=0.001,
                   cosine=1.0, max_abs_error=0.01)
        for key, value in [("top1_match", False), ("relative_l2", 0.03),
                           ("cosine", 0.98), ("max_abs_error", 0.6),
                           ("relative_l2", float("nan"))]:
            with self.subTest(key=key):
                self.assertFalse(numerical_pass({**row, key: value}, "bf16"))

    def test_elementwise_pass_does_not_require_small_logit_norm(self):
        from benchmarks.v1.numerics import numerical_pass
        row = dict(top1_match=True, allclose=True, relative_l2=0.0211,
                   cosine=0.9999, max_abs_error=0.1485)
        self.assertTrue(numerical_pass(row, "bf16"))

    def test_language_fixture_lengths_and_distinct_prefixes(self):
        from benchmarks.v1.fixtures import language_prompt
        tokenizer = SimpleNamespace(encode=lambda text: list(text.encode()))
        for length in [1, 255, 256, 257, 513, 2048]:
            self.assertEqual(len(language_prompt(tokenizer, length, 1)), length)
        self.assertNotEqual(language_prompt(tokenizer, 256, 1),
                            language_prompt(tokenizer, 256, 2))


class ObservationTests(unittest.TestCase):
    """Real mixed scheduler/record adapter; replace only the GPU executor."""

    @classmethod
    def setUpClass(cls):
        class FakeBase:
            def __init__(self, model, greedy=True, **kwargs):
                self.greedy = greedy
                config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512,
                    eos=2, kvcache_block_size=256, num_kvcache_blocks=kwargs.get("blocks", 16),
                    max_model_len=4096)
                self.scheduler = Scheduler(config, 256)
                self.model_runner = SimpleNamespace(
                    call=lambda method, batch, *args: {"tokens": {i: 1000 for i in batch.sample_seq_ids},
                                                        "graph_mode": "eager"},
                    kv_cache=SimpleNamespace(dtype="bfloat16", numel=lambda: 100,
                                             element_size=lambda: 2))

            def add_request(self, prompt, params):
                seq = Sequence(prompt, params)
                self.scheduler.add(seq)
                return seq.seq_id

            def is_finished(self):
                return self.scheduler.is_finished()

            def step(self):
                batch = self.scheduler.schedule()
                result = self.model_runner.call("run_mixed", batch)
                self.scheduler.postprocess(batch, result["tokens"])
                return [(r.seq_id, self.scheduler.sequences[r.seq_id].completion_token_ids)
                        for r in batch.requests if self.scheduler.sequences[r.seq_id].is_finished], {}

        name = "nanovllm.engine.mixed_llm_engine"
        stub = types.ModuleType(name)
        stub.MixedLLMEngine = FakeBase
        previous = sys.modules.get(name)
        sys.modules[name] = stub
        try:
            cls.Engine = load("_v1_observed_test", "benchmarks/v1/observed_engine.py").ObservedMixedEngine
        finally:
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous

    def submit(self, engine, length, token=100, output=4):
        from benchmarks.v0.metrics import RequestRecord
        spec = SimpleNamespace(prompt_token_ids=[token] * length, max_output_tokens=output)
        record = RequestRecord(str(len(engine.by_seq)), 0, length, output, 1000, 100)
        engine.submit(spec, record)
        return record

    def drain(self, engine):
        for _ in range(100):
            if engine.is_finished():
                break
            engine.step()
            engine.scheduler.assert_consistent()
        self.assertTrue(engine.is_finished())
        engine.snapshot_outputs()
        engine.close_waiting_spans(engine.now())

    def test_partial_chunks_never_emit_tokens(self):
        engine = self.Engine("unused", observation="full")
        engine.start_clock()
        record = self.submit(engine, 513)
        engine.step()
        self.assertEqual(record.token_times_ns, [])
        self.assertEqual(record.output_token_ids, [])
        self.drain(engine)
        self.assertEqual(len(record.token_times_ns), 4)
        self.assertEqual(record.status, "completed")
        self.assertTrue(all(s["end_ns"] is not None for s in record.waiting_spans))

    def test_all_modes_preserve_outputs(self):
        from benchmarks.v0.metrics import request_metrics
        outputs = []
        for mode in ["native", "minimal", "full"]:
            engine = self.Engine("unused", observation=mode)
            engine.start_clock()
            first = self.submit(engine, 256)
            engine.step()
            second = self.submit(engine, 513, token=200)
            self.drain(engine)
            outputs.append([first.output_token_ids, second.output_token_ids])
            if mode == "native":
                self.assertEqual(first.token_times_ns, [])
                self.assertIsNone(first.first_scheduled_ns)
            else:
                self.assertEqual(len(first.token_times_ns), 4)
                self.assertIsNotNone(request_metrics(first)["ttft_ms"])
            self.assertEqual(bool(engine.steps), mode == "full")
            if mode == "full":
                self.assertTrue(any(s["phase"] == "mixed" for s in engine.steps))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_recompute_accounting_and_no_duplicate_token_events(self):
        engine = self.Engine("unused", observation="full", blocks=2)
        engine.start_clock()
        a = self.submit(engine, 256)
        b = self.submit(engine, 256, token=200)
        self.drain(engine)
        self.assertGreater(sum(r.preemptions for r in [a, b]), 0)
        self.assertGreater(sum(r.recomputed_tokens for r in [a, b]), 0)
        self.assertEqual([len(r.token_times_ns) for r in [a, b]], [4, 4])
        self.assertEqual(sum(s["recomputed_tokens"] for s in engine.steps),
                         sum(r.recomputed_tokens for r in [a, b]))


if __name__ == "__main__":
    unittest.main()
