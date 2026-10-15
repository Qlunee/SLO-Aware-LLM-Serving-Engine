"""CPU regression for same-named CPU/GPU profiler annotations."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.v3_latency_probe import summarize_profile


class FakeAverages(list):
    def table(self, **_):
        return "operator table\n"


class FakeProfiler:
    def export_chrome_trace(self, path):
        Path(path).write_text("{}")

    def key_averages(self):
        return FakeAverages([
            SimpleNamespace(key="V3_DIAGNOSTIC_STEP", count=8,
                            cpu_time_total=800000, self_device_time_total=0,
                            device_type="DeviceType.CPU"),
            SimpleNamespace(key="V3_DIAGNOSTIC_STEP", count=8,
                            cpu_time_total=0, self_device_time_total=800000,
                            device_type="DeviceType.CUDA"),
            SimpleNamespace(key="aten::mm", count=100,
                            cpu_time_total=20000, self_device_time_total=90000,
                            device_type="DeviceType.CPU"),
        ])

    def events(self):
        raise AssertionError("raw events must not decide Step count")


class LatencyProbeTest(unittest.TestCase):
    def test_uses_aggregated_step_count(self):
        with tempfile.TemporaryDirectory() as folder:
            result = summarize_profile(FakeProfiler(), Path(folder), "d6-p0-m0-c1024")
            self.assertEqual(result["profiled_steps"], 8)
            self.assertEqual(result["cuda_timed_op_calls"], 100)
            self.assertEqual(result["cuda_self_ms_sum"], 90)
            self.assertTrue((Path(folder) / result["trace"]).exists())


if __name__ == "__main__":
    unittest.main()
