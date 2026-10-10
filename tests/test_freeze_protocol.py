"""Standard-library-only tests; never freeze or modify the real project."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/freeze_v1_protocol.py"
spec = importlib.util.spec_from_file_location("freeze_v1_protocol", SCRIPT)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class FreezeProtocolTests(unittest.TestCase):
    def fixture(self, root):
        (root/"configs").mkdir()
        protocol.save(root/"configs/qwen3_4b.json", {"model_id": "test-only"})
        (root/"nanovllm").mkdir()
        (root/"nanovllm/test.py").write_text("# test fixture\n")
        for name in protocol.EVIDENCE_DIRS:
            folder = root/"results/raw"/name
            folder.mkdir(parents=True)
            protocol.save(folder/"report.json", {"status": "MEASURED_NEEDS_REVIEW"})
            (folder/"payload.bin").write_bytes(b"evidence")
            protocol.save(folder/"artifact_hashes.json", {
                name: protocol.sha(folder/name) for name in ("report.json", "payload.bin")})

    def test_freeze_verify_skips_empty_directory_and_preserves_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            folder = protocol.freeze(root)
            (root/"results/raw/v1-protocol.empty").mkdir()
            protocol.verify(root, folder)
            self.assertEqual(protocol.latest(root), folder)
            rules = protocol.load(folder/"policy.json")
            self.assertFalse("acceptance_passed" in rules)
            track = rules["tracks"]["legacy_numerical_compatibility"]
            self.assertTrue(track["near_tie_is_not_automatic_pass"])
            self.assertEqual(track["bf16_policy"]["relative_l2_max"], .02)
            self.assertEqual(track["strict_allclose"], {"atol": .15, "rtol": .03})

    def test_detects_source_change_and_added_file(self):
        for added in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                folder = protocol.freeze(root)
                name = "added.py" if added else "test.py"
                (root/"nanovllm"/name).write_text("# changed fixture\n")
                with self.assertRaises(RuntimeError):
                    protocol.verify(root, folder)

    def test_detects_payload_and_policy_tampering(self):
        for payload in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                folder = protocol.freeze(root)
                target = (root/"results/raw"/protocol.EVIDENCE_DIRS[0]/"payload.bin"
                          if payload else folder/"policy.json")
                target.write_bytes(b"changed")
                with self.assertRaises(RuntimeError):
                    protocol.verify(root, folder)

    def test_rejects_escape_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            with self.assertRaises(RuntimeError):
                protocol.check(root, {"../escape": "bad"})
            (root/"results/raw"/protocol.EVIDENCE_DIRS[0]/"payload.bin").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                protocol.freeze(root)
            self.assertFalse(any((root/"results/raw").glob("v1-protocol.*")))

    def test_early_inline_probe_has_no_fabricated_historical_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            early = root/"results/raw/v1-first-divergence.005ytr8a"
            (early/"artifact_hashes.json").unlink()
            (early/"step11.pt").write_bytes(b"saved-inline-probe")
            folder = protocol.freeze(root)
            protocol.verify(root, folder)
            self.assertFalse((early/"artifact_hashes.json").exists())
            evidence = protocol.load(folder/"evidence.json")
            self.assertIn("results/raw/v1-first-divergence.005ytr8a/step11.pt", evidence)
            (early/"step11.pt").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                protocol.verify(root, folder)


if __name__ == "__main__":
    unittest.main()
