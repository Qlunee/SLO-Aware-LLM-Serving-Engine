"""CPU回归：不导入nanovllm、torch或transformers。"""
import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from benchmarks.v0.accounting import merge_interval
from benchmarks.v0.driver import execute_trace, rejection_reason
from benchmarks.v0.metrics import RequestRecord, quantile, request_metrics, summarize
from benchmarks.v0.workload import RequestSpec, load_trace, validate_trace
from benchmarks.v0.validate_results import audit


def spec(name="r0", arrival=0, output=2, prompt=None):
    return RequestSpec(name, arrival, [1, 2] if prompt is None else prompt, output, 1000., 100.)


def metadata(count=1, mode="poisson", warmup=0., measurement=1.):
    return dict(kind="metadata", schema_version=1, mode=mode, model_revision="test-revision",
                request_count=count, rps=2. if mode == "poisson" else None,
                warmup_s=warmup, measurement_s=measurement)


def completed(name="r0", arrival=0, times=(10_000_000, 20_000_000)):
    return RequestRecord(name, arrival, 2, len(times), 1000., 100., enqueue_ns=arrival,
                         first_scheduled_ns=arrival, finish_ns=times[-1], status="completed",
                         output_token_ids=list(range(len(times))), token_times_ns=list(times))


CONFIG = SimpleNamespace(max_model_len=4096, kvcache_block_size=256,
                         num_kvcache_blocks=16, hf_config=SimpleNamespace(vocab_size=100))


class FakeEngine:
    def __init__(self, step_ns=50_000_000, fail=False):
        self.timestamp = 0
        self.step_ns = step_ns
        self.fail = fail
        self.scheduler = SimpleNamespace(waiting=[], running=[])

    def start_clock(self):
        self.timestamp = 0

    def now(self):
        return self.timestamp

    def sleep(self, seconds):
        self.timestamp += max(1, round(seconds * 1e9))

    def submit(self, request, record):
        record.enqueue_ns = self.now()
        record.status = "queued"
        self.scheduler.waiting.append(record)

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running

    def step(self):
        if self.fail:
            raise RuntimeError("injected failure")
        self.scheduler.running.extend(self.scheduler.waiting)
        self.scheduler.waiting.clear()
        for r in self.scheduler.running:
            if r.first_scheduled_ns is None:
                r.first_scheduled_ns = self.now()
        self.timestamp += self.step_ns
        for r in self.scheduler.running:
            r.output_token_ids.append(1)
            r.token_times_ns.append(self.now())
            if len(r.output_token_ids) == r.max_output_tokens:
                r.finish_ns, r.status = self.now(), "completed"
        self.scheduler.running[:] = [r for r in self.scheduler.running if r.status != "completed"]

    def snapshot_outputs(self):
        pass

    def close_waiting_spans(self, timestamp):
        pass


class MetricTests(unittest.TestCase):
    def test_quantile_interpolation(self):
        self.assertEqual(quantile([0, 10], .95), 9.5)
        self.assertIsNone(quantile([], .5))
        for values, q in [([math.nan], .5), ([1], -1), ([math.inf], .5)]:
            with self.assertRaises(ValueError):
                quantile(values, q)

    def test_ttft_itl_tpot(self):
        values = request_metrics(completed(times=(10_000_000, 30_000_000, 60_000_000)))
        self.assertEqual(values["ttft_ms"], 10)
        self.assertEqual(values["tpot_ms"], 25)
        self.assertAlmostEqual(values["itl_p99_ms"], 29.9)
        self.assertTrue(values["slo_pass"])

    def test_unfinished_rejected_in_denominator(self):
        records = [completed(), RequestRecord("r1", 0, 2, 2, 1000., 100., status="unfinished"),
                   RequestRecord("r2", 0, 2, 2, 1000., 100., status="rejected")]
        values = summarize(records, 0, 1_000_000_000, True)
        self.assertAlmostEqual(values["cohort_violation_rate"], 2 / 3)
        self.assertEqual(values["completed_cohort_violation_rate"], 0)

    def test_single_token_excluded_from_dual_slo(self):
        values = summarize([completed(times=(10_000_000,))], 0, 1_000_000_000, True)
        self.assertEqual(values["single_token_cohort_arrivals"], 1)
        self.assertIsNone(values["cohort_violation_rate"])
        self.assertEqual(values["window_goodput_rps"], 0)

    def test_drain_is_not_window_goodput(self):
        r = completed(arrival=900_000_000, times=(1_010_000_000, 1_020_000_000))
        values = summarize([r], 0, 1_000_000_000, True)
        self.assertEqual(values["cohort_success_rate"], 1)
        self.assertEqual(values["window_goodput_rps"], 0)
        self.assertEqual(values["window_throughput_tok_s"], 0)

    def test_warmup_and_half_open_window(self):
        records = [completed("warm", 0, (90_000_000, 110_000_000)),
                   completed("edge", 100_000_000, (150_000_000, 200_000_000))]
        values = summarize(records, 100_000_000, 200_000_000, True)
        self.assertEqual(values["cohort_arrivals"], 1)
        self.assertEqual(values["window_completed_count"], 1)
        self.assertEqual(values["window_throughput_tok_s"], 20)

    def test_native_metrics_are_unknown(self):
        r = completed()
        r.token_times_ns = []
        values = summarize([r], 0, 1_000_000_000, False)
        self.assertIsNone(values["window_goodput_rps"])
        self.assertIsNone(values["ttft_p99_ms"])

    def test_invalid_timestamps(self):
        for changes in [dict(enqueue_ns=-1), dict(first_scheduled_ns=-1),
                        dict(token_times_ns=[20, 10]), dict(finish_ns=0), dict(finish_ns=None)]:
            r = completed()
            for k, v in changes.items():
                setattr(r, k, v)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                request_metrics(r)

    def test_duplicate_records_rejected(self):
        with self.assertRaises(ValueError):
            summarize([completed(), completed()], 0, 1_000_000_000, True)


class TraceTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(json.dumps(metadata()) + "\n" + json.dumps(asdict(spec())) + "\n")
            meta, requests = load_trace(path)
            self.assertEqual(meta, metadata())
            self.assertEqual(requests, [spec()])

    def test_invalid_metadata(self):
        for changes in [dict(schema_version=99), dict(model_revision=""), dict(warmup_s=-1),
                        dict(measurement_s=0), dict(measurement_s=math.nan), dict(rps=math.inf),
                        dict(request_count=2), dict(mode="closed_loop")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_trace(metadata() | changes, [spec()])

    def test_invalid_requests(self):
        for changes in [dict(arrival_ns=-1), dict(arrival_ns=True), dict(arrival_ns=1_000_000_000),
                        dict(prompt_token_ids=[]), dict(prompt_token_ids=[True]),
                        dict(max_output_tokens=0), dict(ttft_slo_ms=math.nan), dict(request_id="")]:
            r = spec()
            for k, v in changes.items():
                setattr(r, k, v)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_trace(metadata(), [r])

    def test_duplicate_and_unsorted(self):
        for requests in [[spec(), spec()], [spec("a", 2), spec("b", 1)]]:
            with self.assertRaises(ValueError):
                validate_trace(metadata(2), requests)

    def test_simultaneous_contract(self):
        validate_trace(metadata(mode="simultaneous"), [spec()])
        with self.assertRaises(ValueError):
            validate_trace(metadata(mode="simultaneous"), [spec(arrival=1)])

    def test_empty_poisson_trace_valid_but_not_benchmark(self):
        validate_trace(metadata(0), [])


class AccountingTests(unittest.TestCase):
    def test_prefix_cached_tokens_are_not_computed(self):
        overlap, history = merge_interval([], 256, 512)
        self.assertEqual(overlap, 0)
        overlap, history = merge_interval(history, 0, 768)
        self.assertEqual(overlap, 256)
        self.assertEqual(history, [(0, 768)])

    def test_union_and_repeated_recompute(self):
        history = [(0, 5), (10, 20)]
        overlap, history = merge_interval(history, 3, 12)
        self.assertEqual(overlap, 4)
        self.assertEqual(history, [(0, 20)])
        self.assertEqual(merge_interval(history, 3, 12)[0], 9)

    def test_zero_and_invalid_spans(self):
        self.assertEqual(merge_interval([], 1, 1), (0, []))
        for a, b in [(-1, 1), (2, 1)]:
            with self.assertRaises(ValueError):
                merge_interval([], a, b)


class DriverTests(unittest.TestCase):
    def drive(self, requests, engine=None, measurement=1., drain=0.):
        engine = engine or FakeEngine()
        return execute_trace(engine, CONFIG, metadata(len(requests), measurement=measurement),
                             requests, drain, "full", sleep=engine.sleep)

    def test_open_loop_arrival_is_not_shifted_by_busy_step(self):
        result = self.drive([spec("a"), spec("b", 10_000_000)])
        second = result.records[1]
        self.assertEqual(second.arrival_ns, 10_000_000)
        self.assertEqual(second.enqueue_ns, 50_000_000)
        self.assertTrue(any(row["arrived_not_submitted"] == 1 for row in result.backlog))
        self.assertEqual([r.status for r in result.records], ["completed", "completed"])

    def test_idle_does_not_submit_future_request(self):
        r = self.drive([spec(arrival=250_000_000)]).records[0]
        self.assertEqual(r.enqueue_ns, 250_000_000)

    def test_deadline_overrun_and_unsubmitted_remain_in_cohort(self):
        result = self.drive([spec("a", output=1), spec("b", 500_000_000)], FakeEngine(2_000_000_000))
        self.assertEqual([r.status for r in result.records], ["timeout", "unfinished"])
        self.assertIsNone(result.records[1].enqueue_ns)
        self.assertEqual(result.end_ns, 2_000_000_000)

    def test_finite_drain(self):
        result = self.drive([spec(arrival=900_000_000)], measurement=1., drain=.2)
        self.assertEqual(result.records[0].status, "completed")

    def test_exception_marks_all_uncompleted(self):
        result = self.drive([spec("a"), spec("b", 500_000_000)], FakeEngine(fail=True))
        self.assertIn("injected failure", result.error)
        self.assertEqual([r.status for r in result.records], ["failed", "failed"])

    def test_rejections(self):
        self.assertEqual(rejection_reason(spec(prompt=[100]), CONFIG), "invalid_prompt_token_id")
        self.assertEqual(rejection_reason(spec(output=4096), CONFIG), "request_exceeds_max_model_len")
        small = copy.copy(CONFIG)
        small.num_kvcache_blocks = 1
        self.assertEqual(rejection_reason(spec(output=300), small), "request_exceeds_kv_capacity")
        result = self.drive([spec(prompt=[-1])])
        self.assertEqual(result.records[0].status, "rejected")


class AuditTests(unittest.TestCase):
    def test_recompute_and_detect_tampering(self):
        from benchmarks.v0.workload import trace_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps(metadata()) + "\n" + json.dumps(asdict(spec())) + "\n")
            r = completed()
            summary = summarize([r], 0, 1_000_000_000, True)
            summary.update(valid_run=True, observation="full", empty_trace=False)
            (root / "environment.json").write_text(json.dumps(dict(trace_sha256=trace_sha256(trace))))
            (root / "summary.json").write_text(json.dumps(summary))
            (root / "requests.jsonl").write_text(json.dumps(asdict(r) | dict(metrics=request_metrics(r))) + "\n")
            self.assertTrue(audit(root)["passed"])
            summary["cohort_violation_rate"] = .5
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                audit(root)


if __name__ == "__main__":
    unittest.main()
