import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.defaults import (
    DEFAULT_MODEL_PATH, DEFAULT_MODEL_REVISION, MODEL_PROFILE,
    add_model_arguments, resolve_model_arguments, revision_for,
)


class ModelDefaultsTests(unittest.TestCase):
    def parse(self, arguments):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        return resolve_model_arguments(parser.parse_args(arguments), parser)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_model_and_frozen_revision(self):
        args = self.parse([])
        self.assertEqual(args.model, str(Path(DEFAULT_MODEL_PATH).resolve()))
        self.assertEqual(args.model_revision, DEFAULT_MODEL_REVISION)
        self.assertEqual(MODEL_PROFILE["model_id"], "Qwen/Qwen3-4B")

    @patch.dict(os.environ, {"V0_MODEL": "/tmp/another-model", "V0_MODEL_REVISION": "custom-sha"}, clear=True)
    def test_environment_override(self):
        args = self.parse([])
        self.assertEqual(args.model, str(Path("/tmp/another-model").resolve()))
        self.assertEqual(args.model_revision, "custom-sha")

    @patch.dict(os.environ, {"V0_MODEL": "/tmp/environment-model", "V0_MODEL_REVISION": "environment-sha"}, clear=True)
    def test_explicit_arguments_take_precedence(self):
        args = self.parse(["--model", "/tmp/explicit-model", "--model-revision", "explicit-sha"])
        self.assertEqual(args.model, str(Path("/tmp/explicit-model").resolve()))
        self.assertEqual(args.model_revision, "explicit-sha")

    @patch.dict(os.environ, {}, clear=True)
    def test_custom_path_needs_revision(self):
        with self.assertRaises(ValueError):
            revision_for("/tmp/another-model")

    @patch.dict(os.environ, {}, clear=True)
    def test_whitespace_revision_rejected(self):
        with self.assertRaises(ValueError):
            revision_for(DEFAULT_MODEL_PATH, "   ")

    def test_qwen4b_gqa_and_kv_geometry(self):
        profile = MODEL_PROFILE
        self.assertEqual(profile["num_attention_heads"] // profile["num_key_value_heads"], 4)
        # Qwen3-4B的head_dim不能误算为hidden_size/num_heads=80。
        self.assertEqual(profile["head_dim"], 128)
        payload = 2 * profile["num_hidden_layers"] * profile["num_key_value_heads"] * profile["head_dim"] * 2
        self.assertEqual(payload, 147456)
        self.assertEqual(payload // 2, 73728)

    @patch.dict(os.environ, {}, clear=True)
    def test_run_parser_uses_shared_defaults(self):
        from benchmarks.v0.run import parse_args
        with patch.object(sys, "argv", ["run", "--trace", "trace.jsonl", "--out", "output"]):
            args = parse_args()
        self.assertEqual(args.model, str(Path(DEFAULT_MODEL_PATH).resolve()))
        self.assertEqual(args.model_revision, DEFAULT_MODEL_REVISION)

    def test_offline_entry_help_does_not_import_cuda(self):
        root = Path(__file__).resolve().parents[1]
        for script in ["example.py", "bench.py"]:
            result = subprocess.run([sys.executable, str(root / script), "--help"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Qwen3-4B", result.stdout)

    def test_environment_script_exports_defaults_without_installing(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("V0_MODEL", None)
        env.pop("V0_MODEL_REVISION", None)
        env["V0_PYTHON"] = sys.executable
        result = subprocess.run(["bash", "-c",
                                 'source scripts/env_qwen3_4b.sh && "$V0_PYTHON" -c "import os; print(os.environ[\'V0_MODEL\']); print(os.environ[\'V0_MODEL_REVISION\'])"'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [DEFAULT_MODEL_PATH, DEFAULT_MODEL_REVISION])


if __name__ == "__main__":
    unittest.main()
