"""Frozen experiment inputs, provenance and integrity (standard library only)."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ("nanovllm/**/*.py", "benchmarks/**/*.py", "tests/*.py",
            "scripts/**/*.py", "scripts/**/*.sh", "configs/**/*.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sources(root=ROOT):
    return {str(p.relative_to(root)): sha(p)
            for p in sorted({p for pattern in PATTERNS for p in root.glob(pattern) if p.is_file()})}


def seal(folder):
    folder = Path(folder)
    save(folder / "artifact_hashes.json", {str(p.relative_to(folder)): sha(p)
         for p in sorted(folder.rglob("*"))
         if p.is_file() and p != folder / "artifact_hashes.json"})


def verify_artifacts(folder):
    folder = Path(folder)
    manifest = read(folder / "artifact_hashes.json")
    if not manifest:
        raise ValueError("empty artifact manifest")
    for name, digest in manifest.items():
        path = (folder / name).resolve()
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_relative_to(folder.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"artifact missing or modified: {folder / name}")


@dataclass(frozen=True)
class Experiment:
    decode_count: int = 8
    decode_prompt_tokens: int = 256
    decode_output_tokens: int = 512
    long_prompt_tokens: int = 3072
    long_output_tokens: int = 64
    inject_s: float = 2.0
    window_s: float = 2.0
    drain_s: float = 120.0
    stable_intervals: int = 8
    stable_cv_max: float = 0.5
    max_enqueue_lag_ms: float = 250.0
    max_model_len: int = 4096
    token_budget: int = 4096
    memory: float = 0.8
    ttft_slo_ms: float = 1000.0
    itl_slo_ms: float = 100.0
    throughput_loss_limit: float = 0.05
    chunks: tuple = (256, 512, 1024, 2048)
    tune_seeds: tuple = (141, 142, 143)
    eval_seeds: tuple = (241, 242, 243)

    def validate(self):
        positive_ints = (self.decode_count, self.decode_prompt_tokens, self.decode_output_tokens,
                         self.long_prompt_tokens, self.long_output_tokens, self.stable_intervals,
                         self.max_model_len, self.token_budget)
        if any(type(v) is not int or v <= 0 for v in positive_ints):
            raise ValueError("counts and lengths must be positive integers")
        values = (self.inject_s, self.window_s, self.drain_s, self.stable_cv_max,
                  self.max_enqueue_lag_ms, self.ttft_slo_ms, self.itl_slo_ms)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("times, stability and SLO parameters must be positive and finite")
        if not 0 < self.memory < 1 or not 0 <= self.throughput_loss_limit < 1:
            raise ValueError("invalid memory or throughput-loss bound")
        if self.decode_output_tokens <= self.stable_intervals + 1:
            raise ValueError("Decode output too short for stability check")
        if max(self.decode_prompt_tokens + self.decode_output_tokens,
               self.long_prompt_tokens + self.long_output_tokens) > self.max_model_len:
            raise ValueError("input plus output exceeds model length")
        if self.token_budget < max(self.long_prompt_tokens, self.decode_count):
            raise ValueError("global budget must admit the full original long Prefill and Decode batch")
        if not self.chunks or len(set(self.chunks)) != len(self.chunks) or any(
                c not in (256, 512, 1024, 2048) for c in self.chunks):
            raise ValueError("chunks must be distinct supported V1 regression sizes")
        if self.decode_count not in (1, 2, 8, 32):
            raise ValueError("Decode count must match the existing V1 regression entry")
        for seeds in (self.tune_seeds, self.eval_seeds):
            if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError("at least three distinct nonnegative seeds per split required")
        if set(self.tune_seeds) & set(self.eval_seeds):
            raise ValueError("tuning and evaluation seeds overlap")


def build_trace(config, seed, split, token_factory):
    config.validate()
    if split not in ("tune", "eval") or seed not in getattr(config, f"{split}_seeds"):
        raise ValueError("seed does not belong to the requested split")
    rows = []
    for i in range(config.decode_count + 1):
        long = i == config.decode_count
        length = config.long_prompt_tokens if long else config.decode_prompt_tokens
        tokens = token_factory(length, seed, i)
        if len(tokens) != length or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("tokenizer fixture did not produce the requested length")
        rows.append({"request_id": "long" if long else f"decode-{i}",
                     "arrival_ns": round(config.inject_s * 1e9) if long else 0,
                     "prompt_token_ids": list(tokens),
                     "max_output_tokens": config.long_output_tokens if long else config.decode_output_tokens,
                     "ttft_slo_ms": config.ttft_slo_ms, "itl_slo_ms": config.itl_slo_ms})
    return {"schema": "controlled-injection-v1", "split": split, "seed": seed,
            "synthetic_language_fixture": True, "not_real_user_data": True,
            "decode_ids": [r["request_id"] for r in rows[:-1]], "long_id": "long",
            "inject_ns": round(config.inject_s * 1e9),
            "window_end_ns": round((config.inject_s + config.window_s) * 1e9), "requests": rows}


def config_dict(config):
    return json.loads(json.dumps(asdict(config)))


def load_suite(folder, root=ROOT):
    folder = Path(folder)
    verify_artifacts(folder)
    protocol = read(folder / "protocol.json")
    config = Experiment(**protocol["experiment"])
    config.validate()
    if sources(root) != protocol["source_sha256"]:
        raise ValueError("source set changed since V2 protocol freeze; prepare a NEW suite")
    for relative, digest in protocol["trace_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(folder / path) != digest:
            raise ValueError("frozen trace changed")
    return protocol, config
