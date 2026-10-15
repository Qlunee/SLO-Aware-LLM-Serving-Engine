"""Causality and cold-start checks for the diagnostic drift replay."""
import unittest

from tools.v3_drift_replay import replay


def row(step, actual, predicted=10.0, supported=True, profile="one", window="window"):
    return {"profile": profile, "step_id": step, "actual_ms": actual,
            "prediction": {"predicted_ms": predicted, "supported": supported},
            "window": window}


class DriftReplayTest(unittest.TestCase):
    def test_current_step_cannot_calibrate_itself(self):
        rows = [row(i, 12.0) for i in range(32)] + [row(32, 30.0), row(33, 12.0)]
        details, report = replay(rows, "controlled")
        self.assertFalse(details[31]["calibrated"])
        self.assertEqual(details[32]["adjusted_ms"], 12.0)
        self.assertEqual(details[33]["adjusted_ms"], 12.0)
        self.assertEqual(report["calibrated_steps"], 2)

    def test_unsupported_does_not_train_guard(self):
        rows = [row(i, 100.0, supported=False) for i in range(40)]
        rows += [row(i, 12.0) for i in range(40, 73)]
        details, report = replay(rows, "controlled")
        self.assertEqual(details[72]["prior_supported_observations"], 32)
        self.assertEqual(details[72]["adjusted_ms"], 12.0)
        self.assertEqual(report["unsupported_steps"], 40)

    def test_profile_resets_and_warmup_updates_without_scoring(self):
        rows = [row(i, 12.0, window="warmup") for i in range(32)]
        rows += [row(32, 12.0)]
        details, report = replay(rows, "online")
        self.assertTrue(details[32]["calibrated"])
        self.assertEqual(report["total_scored_steps"], 1)
        controlled = [row(i, 12.0) for i in range(32)]
        controlled += [row(0, 12.0, profile="two")]
        details, report = replay(controlled, "controlled")
        self.assertFalse(details[-1]["calibrated"])
        self.assertEqual(report["calibrated_steps"], 0)

    def test_rejects_out_of_order_steps(self):
        with self.assertRaisesRegex(ValueError, "causal order"):
            replay([row(1, 12.0), row(1, 13.0)], "controlled")


if __name__ == "__main__":
    unittest.main()
