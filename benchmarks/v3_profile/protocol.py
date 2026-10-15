"""Standard-library experiment planning, provenance and artifact validation."""
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources, verify_artifacts

TIMING_CONTRACT = "decision-to-cpu-visible-postprocess-v1"


def predictor_module():
    # nanovllm/__init__.py imports CUDA libraries. Offline fit/evaluate and CPU
    # tests must work without torch; load the stdlib-only estimator directly.
    name = "_slo_v3_step_predictor"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/"nanovllm/engine/step_predictor.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class Settings:
    decode_counts: tuple = (0, 1, 8)
    prefill_budgets: tuple = (0, 256, 512)
    contexts: tuple = (256, 1024)
    prefill_request_counts: tuple = (1, 2)
    repeats: int = 50
    warmup: int = 3
    max_model_len: int = 4096
    token_budget: int = 4096
    max_num_seqs: int = 32
    memory: float = .8
    kv_block_cap: int = 256

    def validate(self):
        for name in ("decode_counts", "prefill_budgets", "contexts", "prefill_request_counts"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"invalid matrix: {name}")
        if min(self.contexts) < 1 or min(self.prefill_request_counts) < 1:
            raise ValueError("contexts/Prefill request counts must be positive")
        for name in ("repeats", "warmup", "max_model_len", "token_budget", "max_num_seqs", "kv_block_cap"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.memory) or not 0 < self.memory < 1:
            raise ValueError("invalid memory fraction")
        planned_cases(self)


def decode_output_limit(settings, case):
    # Bootstrap may let early requests Decode while later ones still Prefill.
    bootstrap_bound = math.ceil(case["context_tokens"]/256)*case["decode_count"]
    return 4*(settings.repeats+settings.warmup) + bootstrap_bound + 8


def planned_cases(settings):
    cases = {}
    for d in settings.decode_counts:
        for p in settings.prefill_budgets:
            if not d and not p:
                continue
            for context in settings.contexts if d else (0,):
                for m in settings.prefill_request_counts if p else (0,):
                    if p and (p < m or p % m):
                        raise ValueError("Prefill budget must divide evenly across requests")
                    case = {"decode_count": d, "prefill_budget": p,
                            "prefill_requests": m, "context_tokens": context}
                    if d+p > settings.token_budget or d+m > settings.max_num_seqs:
                        raise ValueError("case exceeds global token/request limit")
                    output = decode_output_limit(settings, case)
                    prompt = 2*(p//m)+1 if p else 0
                    if max(context+output if d else 0, prompt+1) > settings.max_model_len:
                        raise ValueError("case may exceed model context; reduce repeats/length")
                    # Independent random prompts deliberately have no prefix sharing.
                    blocks = d*math.ceil((context+output-1)/256) + m*math.ceil((prompt+1)/256)
                    if blocks > settings.kv_block_cap:
                        raise ValueError("case may exceed test KV cap; increase cap or reduce matrix")
                    name = f"d{d}-p{p}-m{m}-c{context}"
                    cases[name] = {"case_id": name, **case}
    if not cases:
        raise ValueError("empty case matrix")
    return list(cases.values())


def runtime_identity(environment, config, kv_cache):
    """Only stable execution facts; exclude seed, timestamps and run paths."""
    return {"model_revision": environment["model_revision_declared"],
        "model_metadata_sha256": environment["model_metadata_sha256"],
        "weight_manifest": environment["weight_manifest"],
        "packages": environment["packages"], "torch_cuda_version": environment["torch_cuda_version"],
        "gpus": environment["gpus"], "nccl_version": environment["nccl_version"],
        "driver": environment["v3_driver"],
        "source_sha256": sources(), "tp_size": config.tensor_parallel_size,
        "enforce_eager": config.enforce_eager, "kv_dtype": str(kv_cache.dtype),
        "attention_backend": "flash-attn-v1-mixed", "sampler": "greedy",
        "block_size": config.kvcache_block_size, "kv_blocks": config.num_kvcache_blocks,
        "max_model_len": config.max_model_len, "max_num_seqs": config.max_num_seqs,
        "global_token_budget": config.max_num_batched_tokens,
        "timing_contract": TIMING_CONTRACT}


def load_profile(folder):
    folder = Path(folder)
    verify_artifacts(folder)
    protocol, summary = read(folder/"protocol.json"), read(folder/"summary.json")
    if protocol.get("schema") != "v3-step-profile-v1" or not summary.get("valid"):
        raise ValueError("not a complete valid V3 profile")
    settings = Settings(**protocol["settings"])
    settings.validate()
    cases = {c["case_id"]: c for c in planned_cases(settings)}
    if (set(summary["completed_cases"]) != set(cases)
            or len(summary["completed_cases"]) != len(cases)
            or summary["preemptions"] != 0 or not summary["source_unchanged"]
            or not summary["state_and_drain_passed"]):
        raise ValueError("incomplete case matrix or invalid state/provenance")
    if protocol["identity"]["timing_contract"] != TIMING_CONTRACT:
        raise ValueError("wrong timing endpoint")
    import json
    rows = [json.loads(line) for line in (folder/"steps.jsonl").read_text().splitlines()]
    if not rows or len(rows) != summary["recorded_steps"]:
        raise ValueError("missing Step records")
    estimator = predictor_module()
    observed = set()
    for i, row in enumerate(rows):
        if row["step_id"] != i or row["case_id"] not in cases:
            raise ValueError("invalid Step/case identity")
        case = cases[row["case_id"]]
        if (type(row["episode"]) is not int or not 0 <= row["episode"] < settings.warmup+settings.repeats
                or type(row["episode_step"]) is not int
                or not 0 <= row["episode_step"] < (4 if case["prefill_budget"] else 1)
                or row["warmup"] != (row["episode"] < settings.warmup)):
            raise ValueError("invalid episode or warmup membership")
        signature = (row["case_id"], row["episode"], row["episode_step"])
        if signature in observed:
            raise ValueError("duplicate Step sample")
        observed.add(signature)
        estimator.validate_features(row["features"])
        timestamps = [row[k] for k in ("start_ns", "scheduled_ns", "run_end_ns", "visible_ns")]
        if any(type(t) is not int for t in timestamps) or timestamps != sorted(timestamps):
            raise ValueError("invalid Step clock")
        expected = (row["visible_ns"]-row["start_ns"])/1e6
        if expected <= 0 or not math.isclose(expected, row["step_time_ms"], abs_tol=1e-9):
            raise ValueError("Step duration differs from timestamp endpoints")
        if type(row.get("warmup")) is not bool:
            raise ValueError("missing warmup marker")
    expected = sum((4 if c["prefill_budget"] else 1)*(settings.warmup+settings.repeats)
                   for c in cases.values())
    if len(observed) != expected:
        raise ValueError("missing case/episode samples")
    if sum(not r["warmup"] for r in rows) != summary["measured_steps"]:
        raise ValueError("measured sample count differs")
    return protocol, rows

