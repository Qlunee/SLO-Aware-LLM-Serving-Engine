"""New source-set freeze; old V1/V2 evidence is retained, not rewritten."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha, sources, verify_artifacts
from .workload import Workload, build_trace, validate_trace

POLICIES = ("baseline", "chunk-256", "chunk-512", "chunk-1024", "chunk-2048",
            "decode_first", "decode_first_wait")


@dataclass(frozen=True)
class Settings:
    seed: int = 41
    max_model_len: int = 4096
    token_budget: int = 4096
    max_num_seqs: int = 32
    memory: float = .8
    drain_s: float = 60
    max_prefill_wait_ms: float = 1000
    starvation_ms: float = 1000

    def validate(self, workload):
        workload.validate()
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid seed")
        for name in ("max_model_len", "token_budget", "max_num_seqs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.memory) or not 0 < self.memory < 1:
            raise ValueError("invalid memory fraction")
        for name in ("drain_s", "max_prefill_wait_ms", "starvation_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if workload.long_tokens+workload.output_tokens > self.max_model_len:
            raise ValueError("input plus output exceeds model length; no silent truncation")
        if self.token_budget < workload.long_tokens:
            raise ValueError("reference budget must admit the full longest Prompt")
        if self.max_num_seqs < 2:
            raise ValueError("stage1 needs at least two sequence slots to test mixed execution")
        # Unit-average Burst is exact over complete cycles, including each window.
        for seconds in (workload.warmup_s, workload.measurement_s):
            ratio = seconds/workload.burst_period_s
            if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
                raise ValueError("warmup/measurement must contain complete Burst cycles")


def metadata(model):
    model = Path(model)
    names = ("config.json", "tokenizer_config.json", "tokenizer.json",
             "generation_config.json", "model.safetensors.index.json")
    return {name: sha(model/name) for name in names if (model/name).is_file()}


def prepare(folder, settings, workload, model, revision, token_factory, parent=None, root=ROOT):
    settings.validate(workload)
    folder, model = Path(folder).resolve(), Path(model).resolve()
    if not revision.strip() or settings.max_model_len > read(model/"config.json")["max_position_embeddings"]:
        raise ValueError("model revision/context invalid")
    current = sources(root)
    inherited = None
    if parent is not None:
        parent = Path(parent).resolve()
        verify_artifacts(parent)
        previous = read(parent/"protocol.json")
        if any(current.get(k) != v for k, v in previous["source_sha256"].items()):
            raise ValueError("inherited source was modified, not merely extended")
        if previous["model"] != {"path": str(model), "revision": revision}:
            raise ValueError("parent model differs")
        inherited = {"path": str(parent), "protocol_sha256": sha(parent/"protocol.json"),
                     "inherited_source_unchanged": True,
                     "added_sources": sorted(set(current)-set(previous["source_sha256"])),
                     "v1_acceptance_passed": False}
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("prepare needs a new/empty directory; never overwrite evidence")
    (folder/"traces").mkdir()
    hashes = {}
    for mode in ("poisson", "burst", "long_short"):
        trace = build_trace(workload, mode, settings.seed, token_factory, revision)
        start, end = round(workload.warmup_s*1e9), round((workload.warmup_s+workload.measurement_s)*1e9)
        if not any(start <= row["arrival_ns"] < end for row in trace["requests"]):
            raise ValueError("empty measured cohort; choose a prospective new smoke configuration")
        name = f"traces/{mode}-seed-{settings.seed}.json"
        save(folder/name, trace)
        hashes[name] = sha(folder/name)
    protocol = {"schema": "v2-online-protocol-v1", "stage": "V2-stage1-smoke",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": {"path": str(model), "revision": revision},
                "model_metadata_sha256": metadata(model), "settings": asdict(settings),
                "workload": asdict(workload), "policies": list(POLICIES),
                "source_sha256": current, "trace_sha256": hashes, "parent_v2": inherited,
                "scope": "TP1/Eager/BF16/full; single-seed smoke, NOT capacity/selection/formal performance",
                "rules": {"priority_wait": "one forced Chunk after fixed no-progress threshold; conflicts recorded, not a hard latency guarantee",
                          "numerics": "retain every generation divergence; state/Attention success is not full-model numerical acceptance",
                          "incomplete": "retain rejected/unfinished/timeout/failed requests and finite drain",
                          "fairness": "same Trace, seed, model, settings, actual KV blocks and software; fresh processes, shuffled policy order"},
                "v1_acceptance_passed": False}
    save(folder/"protocol.json", protocol)
    seal(folder)
    return protocol


def load(folder, root=ROOT):
    folder = Path(folder)
    verify_artifacts(folder)
    protocol = read(folder/"protocol.json")
    if protocol["schema"] != "v2-online-protocol-v1" or protocol["stage"] != "V2-stage1-smoke":
        raise ValueError("wrong protocol")
    Settings(**protocol["settings"]).validate(Workload(**protocol["workload"]))
    if sources(root) != protocol["source_sha256"]:
        raise ValueError("source set changed; freeze a NEW suite")
    if metadata(protocol["model"]["path"]) != protocol["model_metadata_sha256"]:
        raise ValueError("model metadata changed")
    if protocol["policies"] != list(POLICIES):
        raise ValueError("policy matrix changed")
    for name, digest in protocol["trace_sha256"].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError("trace changed")
        trace = read(path)
        validate_trace(trace)
        if (trace["model_revision"] != protocol["model"]["revision"]
                or trace["workload"] != protocol["workload"]
                or trace["seed"] != protocol["settings"]["seed"]):
            raise ValueError("trace metadata differs from protocol")
    return protocol
