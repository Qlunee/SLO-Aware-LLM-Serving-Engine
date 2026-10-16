"""CPU contracts for Slack selection, resource limits and committed clocks."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import test_v1_state as state


def setUpModule():
    global SlackScheduler, Prediction
    state.setUpModule()
    Prediction = state.load("nanovllm.engine.step_predictor",
                            "nanovllm/engine/step_predictor.py").Prediction
    SlackScheduler = state.load("nanovllm.engine.slack_scheduler",
                                "nanovllm/engine/slack_scheduler.py").SlackScheduler


def tearDownModule():
    state.tearDownModule()


class FakePredictor:
    def __init__(self, timings=None, unsupported=()):
        self.timings = timings or {}
        self.unsupported = set(unsupported)
        self.seen = []

    def predict(self, features):
        self.seen.append(features)
        p = features["prefill_tokens"]
        return Prediction(self.timings.get(p, 100.0), p not in self.unsupported,
                          40 if p not in self.unsupported else 0,
                          "exact_bucket" if p not in self.unsupported else "out_of_coverage")


class SlackTests(unittest.TestCase):
    def make(self, predictor=None, *, budget=1024, slots=32, blocks=64,
             margin=10.0, fallback=256):
        self.now = [0]
        config = SimpleNamespace(max_num_seqs=slots, max_num_batched_tokens=budget,
            eos=2, kvcache_block_size=256, num_kvcache_blocks=blocks,
            max_model_len=4096, enforce_eager=True)
        scheduler = SlackScheduler(config, predictor or FakePredictor(),
            safety_margin_ms=margin, fallback_chunk=fallback,
            min_residual_samples=2, clock=lambda: self.now[0])
        return scheduler

    def seq(self, length, value=100, output=4):
        return state.Sequence([value] * length,
                              state.Params(max_tokens=output, ignore_eos=True))

    def commit(self, scheduler, batch, visible):
        scheduler.postprocess(batch, {i: 1000 for i in batch.sample_seq_ids})
        scheduler.observe_visible(batch, visible)
        scheduler.assert_consistent()

    def ready_decode(self, scheduler, value=100, output=4):
        seq = self.seq(16, value, output=output)
        scheduler.add(seq)
        scheduler.register_request(seq.seq_id, 100)
        for _ in range(16):
            batch = scheduler.schedule()
            self.commit(scheduler, batch, 0)
            if seq.num_completion_tokens:
                break
        self.assertEqual(seq.num_completion_tokens, 1)
        return seq

    def test_no_decode_prefill_and_first_token_clock(self):
        scheduler = self.make()
        seq = self.seq(513)
        scheduler.add(seq)
        scheduler.register_request(seq.seq_id, 100)
        first = scheduler.schedule()
        self.assertEqual(first.num_prefill_tokens, 513)
        self.assertEqual(scheduler.last_decision["decision_reason"], "no_decode")
        self.assertEqual(first.sample_seq_ids, (seq.seq_id,))
        self.commit(scheduler, first, 10_000_000)
        self.assertEqual(scheduler.last_visible_ns[seq.seq_id], 10_000_000)
        self.assertEqual(seq.num_completion_tokens, 1)

    def test_one_budget_can_prefill_multiple_waiting_requests(self):
        scheduler = self.make(budget=512)
        first, second = self.seq(256, 201), self.seq(256, 202)
        for seq in (first, second):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 512)
        self.assertEqual([r.seq_id for r in batch.requests],
                         [first.seq_id, second.seq_id])
        self.assertEqual(scheduler.last_decision["selected_prefill_requests"], 2)
        self.commit(scheduler, batch, 10_000_000)

    def test_prefill_budget_is_shared_before_topping_up_long_request(self):
        scheduler = self.make(budget=512)
        long, short = self.seq(1024, 201), self.seq(256, 202)
        for seq in (long, short):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual([(r.seq_id, r.num_scheduled_tokens) for r in batch.requests],
                         [(long.seq_id, 256), (short.seq_id, 256)])
        self.commit(scheduler, batch, 10_000_000)

    def test_mixed_batch_can_choose_two_prefill_requests(self):
        predictor = FakePredictor({0: 10, 256: 30, 512: 40})
        scheduler = self.make(predictor, budget=513)
        decode = self.ready_decode(scheduler)
        scheduler.residuals["mixed"].extend((0, 0))
        first, second = self.seq(256, 201), self.seq(256, 202)
        for seq in (first, second):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual([r.phase for r in batch.requests],
                         ["decode", "prefill", "prefill"])
        self.assertEqual(batch.num_prefill_tokens, 512)
        self.assertEqual(scheduler.last_decision["selected_prefill_requests"], 2)
        self.commit(scheduler, batch, 10_000_000)

    def test_prefill_candidates_reserve_kv_for_all_selected_requests(self):
        scheduler = self.make(FakePredictor({0: 10, 256: 40}),
                              budget=512, blocks=2)
        self.ready_decode(scheduler)
        first, second = self.seq(256, 201), self.seq(256, 202)
        for seq in (first, second):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 256)
        self.assertEqual(scheduler.last_decision["selected_prefill_requests"], 1)
        self.commit(scheduler, batch, 10_000_000)

    def test_largest_feasible_budget_and_decode_first_rows(self):
        predictor = FakePredictor({0: 10, 256: 40, 512: 80, 513: 100})
        scheduler = self.make(predictor)
        decode = self.ready_decode(scheduler)
        waiting = self.seq(513, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 100)
        self.now[0] = 20_000_000
        batch = scheduler.schedule()
        self.assertEqual([(r.phase, r.num_scheduled_tokens) for r in batch.requests],
                         [("decode", 1), ("prefill", 256)])
        self.assertEqual(scheduler.last_decision["selected_prefill_budget"], 256)
        self.assertEqual(scheduler.last_decision["decision_reason"], "cold_start_fixed_fallback")
        self.assertAlmostEqual(scheduler.last_decision["min_slack_ms"], 30)
        self.assertEqual(waiting.num_cached_tokens, 0)
        self.commit(scheduler, batch, 60_000_000)
        self.assertEqual(waiting.num_cached_tokens, 256)
        self.assertEqual(decode.num_completion_tokens, 2)

    def test_tightest_decode_request_limits_budget(self):
        scheduler = self.make(FakePredictor({0: 10, 16: 10, 256: 40, 512: 80, 513: 100}))
        a = self.ready_decode(scheduler, 100)
        b = self.ready_decode(scheduler, 200)
        scheduler.last_visible_ns[a.seq_id] = 0
        scheduler.last_visible_ns[b.seq_id] = 20_000_000
        waiting = self.seq(513, 300)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 100)
        self.now[0] = 51_000_000
        batch = scheduler.schedule()
        self.assertEqual(batch.num_decode_tokens, 2)
        self.assertEqual(batch.num_prefill_tokens, 0)
        self.assertEqual(scheduler.last_decision["decision_reason"], "cold_start_decode_only")
        self.commit(scheduler, batch, 60_000_000)

    def test_unsupported_mixed_shape_uses_explicit_fixed_fallback(self):
        scheduler = self.make(FakePredictor({0: 10}, unsupported=(256, 512, 513)))
        self.ready_decode(scheduler)
        waiting = self.seq(513, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 256)
        self.assertEqual(scheduler.last_decision["decision_reason"], "unsupported_fixed_fallback")
        self.assertFalse(scheduler.last_decision["prediction_supported"])
        self.commit(scheduler, batch, 40_000_000)

    def test_unsupported_fallback_packs_multiple_waiting_requests(self):
        predictor = FakePredictor({0: 10}, unsupported=(256, 512))
        scheduler = self.make(predictor, budget=513)
        self.ready_decode(scheduler)
        first, second = self.seq(256, 201), self.seq(256, 202)
        for seq in (first, second):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 512)
        self.assertEqual([r.phase for r in batch.requests],
                         ["decode", "prefill", "prefill"])
        self.assertEqual(scheduler.last_decision["decision_reason"],
                         "unsupported_fixed_fallback")
        self.assertEqual(scheduler.last_decision["selected_prefill_requests"], 2)
        self.commit(scheduler, batch, 40_000_000)

    def test_cold_fallback_packs_multiple_waiting_requests(self):
        scheduler = self.make(FakePredictor({0: 10, 256: 40, 512: 50}), budget=513)
        self.ready_decode(scheduler)
        first, second = self.seq(256, 201), self.seq(256, 202)
        for seq in (first, second):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 512)
        self.assertEqual(scheduler.last_decision["decision_reason"],
                         "cold_start_fixed_fallback")
        self.assertEqual(scheduler.last_decision["selected_prefill_requests"], 2)
        self.commit(scheduler, batch, 40_000_000)

    def test_deadline_already_unreachable_selects_pure_decode(self):
        scheduler = self.make(FakePredictor({0: 15, 256: 40, 512: 80}))
        self.ready_decode(scheduler)
        waiting = self.seq(512, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 100)
        self.now[0] = 90_000_000
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 0)
        self.assertTrue(scheduler.last_decision["unavoidable_risk"])
        self.commit(scheduler, batch, 110_000_000)

    def test_calibrated_outlier_cannot_starve_supported_prefill_forever(self):
        scheduler = self.make(FakePredictor({0: 120, 256: 147, 512: 180}),
                              margin=15)
        decode = self.ready_decode(scheduler, output=12)
        scheduler.itl_slo_ms[decode.seq_id] = 180
        scheduler.residuals["mixed"].extend((0, 18))
        waiting = self.seq(512, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 180)
        visible = 0
        for attempt in range(3):
            self.now[0] = visible + 1_000_000
            batch = scheduler.schedule()
            if attempt < 2:
                self.assertEqual(batch.num_prefill_tokens, 0)
            else:
                self.assertEqual(batch.num_prefill_tokens, 256)
                self.assertEqual(scheduler.last_decision["decision_reason"],
                                 "prefill_progress_guard")
                self.assertLess(scheduler.last_decision["min_slack_ms"], 0)
            visible = self.now[0] + 105_000_000
            self.commit(scheduler, batch, visible)
        self.now[0] = visible + 1_000_000
        followup = scheduler.schedule()
        self.assertEqual(followup.num_prefill_tokens, 256)
        self.assertEqual(scheduler.last_decision["decision_reason"],
                         "prefill_progress_guard")
        self.commit(scheduler, followup, self.now[0] + 105_000_000)

    def test_prefill_progress_guard_keeps_fixed_safety_margin(self):
        scheduler = self.make(FakePredictor({0: 40, 256: 95}), margin=10)
        decode = self.ready_decode(scheduler, output=12)
        scheduler.itl_slo_ms[decode.seq_id] = 100
        scheduler.residuals["mixed"].extend((0, 20))
        scheduler.prefill_skip_steps = 2
        waiting = self.seq(256, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 100)
        self.now[0] = 1_000_000
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 0)
        self.assertNotEqual(scheduler.last_decision["decision_reason"],
                            "prefill_progress_guard")
        self.commit(scheduler, batch, 40_000_000)

    def test_slow_guard_step_restores_cooldown(self):
        scheduler = self.make(FakePredictor({0: 120, 256: 147, 512: 180}),
                              margin=15)
        decode = self.ready_decode(scheduler, output=12)
        scheduler.itl_slo_ms[decode.seq_id] = 180
        scheduler.residuals["mixed"].extend((0, 18))
        scheduler.prefill_skip_steps = 2
        waiting = self.seq(512, 200)
        scheduler.add(waiting)
        scheduler.register_request(waiting.seq_id, 180)
        self.now[0] = 1_000_000
        guarded = scheduler.schedule()
        self.assertEqual(guarded.num_prefill_tokens, 256)
        self.commit(scheduler, guarded, 175_000_000)
        self.assertFalse(scheduler.last_decision["progress_guard_rearmed"])
        self.now[0] = 176_000_000
        cooldown = scheduler.schedule()
        self.assertEqual(cooldown.num_prefill_tokens, 0)
        self.commit(scheduler, cooldown, 280_000_000)

    def test_ttft_urgency_uses_bounded_empirical_fallback_for_critical_request(self):
        scheduler = self.make(FakePredictor({0: 10, 256: 90, 512: 95},
                                            unsupported=(256,)), margin=10)
        decode = self.ready_decode(scheduler, output=12)
        scheduler.last_visible_ns[decode.seq_id] = 149_000_000
        first, critical = self.seq(256, 201), self.seq(512, 202)
        for seq, ttft in ((first, 1000), (critical, 300)):
            scheduler.add(seq)
            scheduler.register_request(seq.seq_id, 100, ttft, 0)
        scheduler.small_prefill_ms.extend((70, 75, 80))
        self.now[0] = 150_000_000
        batch = scheduler.schedule()
        self.assertEqual(batch.num_prefill_tokens, 256)
        self.assertEqual(batch.requests[-1].seq_id, critical.seq_id)
        self.assertEqual(scheduler.last_decision["decision_reason"],
                         "ttft_urgent_fallback")
        self.assertFalse(scheduler.last_decision["prediction_supported"])
        self.commit(scheduler, batch, 230_000_000)

    def test_global_token_and_sequence_limits_prevent_prefill(self):
        for kwargs in ({"budget": 1}, {"slots": 1}):
            with self.subTest(kwargs=kwargs):
                scheduler = self.make(**kwargs)
                self.ready_decode(scheduler)
                waiting = self.seq(512, 200)
                scheduler.add(waiting)
                scheduler.register_request(waiting.seq_id, 100)
                batch = scheduler.schedule()
                self.assertEqual(batch.num_prefill_tokens, 0)
                self.commit(scheduler, batch, 20_000_000)

    def test_residual_margin_uses_only_previous_visible_steps(self):
        scheduler = self.make(FakePredictor({0: 10}), margin=2)
        seq = self.ready_decode(scheduler)
        for visible in (20_000_000, 40_000_000):
            self.now[0] = visible - 20_000_000
            batch = scheduler.schedule()
            was_calibrated = scheduler.last_decision["residual_calibrated"]
            self.commit(scheduler, batch, visible)
            self.assertFalse(was_calibrated)
        self.now[0] = 40_000_000
        batch = scheduler.schedule()
        self.assertTrue(scheduler.last_decision["residual_calibrated"])
        self.assertGreater(scheduler.last_decision["safety_margin_ms"], 2)
        self.commit(scheduler, batch, 60_000_000)

    def test_unsupported_then_supported_step_writes_one_csv_schema(self):
        from benchmarks.v0.run import write_csv
        scheduler = self.make(FakePredictor({0: 10}, unsupported=(16,)))
        seq = self.seq(16)
        scheduler.add(seq)
        scheduler.register_request(seq.seq_id, 100)
        first = scheduler.schedule()
        self.commit(scheduler, first, 10_000_000)
        rows = [dict(scheduler.last_decision)]
        self.now[0] = 10_000_000
        second = scheduler.schedule()
        self.commit(scheduler, second, 20_000_000)
        rows.append(dict(scheduler.last_decision))
        with tempfile.TemporaryDirectory() as temp:
            write_csv(Path(temp)/"steps.csv", rows)


if __name__ == "__main__":
    unittest.main()
