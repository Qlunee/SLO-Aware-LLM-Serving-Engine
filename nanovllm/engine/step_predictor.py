"""CPU-only empirical Step estimator. Unsupported estimates are NOT safe budgets.

Use via the existing nanovllm package on the server; CPU tests load this file
directly to avoid executing nanovllm's CUDA-dependent package initializer.
请求批次
  -> 提取特征 batch_features()
  -> 校验特征 validate_features()
  -> 特征分桶 bucket_key()
  -> 用历史样本 fit_table() 建表
  -> StepPredictor.predict() 查询预测结果
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path


SCHEMA = "v3-step-p95-v1"
FEATURE_FIELDS = ("decode_count", "prefill_tokens", "prefill_request_count",
                  "sum_context_tokens", "max_context_len", "sample_count",
                  "prefill_has_history")
DEFAULT_WIDTHS = {"prefill_tokens": 256, "sum_context_tokens": 2048,
                  "max_context_len": 256}


def percentile(values, q):
    """Nearest-rank quantile; no NumPy dependency or interpolation ambiguity."""
    # 计算 P95 延迟
    values = sorted(values)
    if not values or not 0 < q <= 1 or any(
            not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid percentile input")
    return values[max(0, math.ceil(q * len(values)) - 1)]


def batch_features(batch, graph_mode):
    # 从实际 batch 中提取预测所需的特征
    contexts = [row.end_pos for row in batch.requests]
    result = {
        "decode_count": batch.num_decode_tokens,
        "prefill_tokens": batch.num_prefill_tokens,
        "prefill_request_count": sum(r.phase != "decode" for r in batch.requests),
        "sum_context_tokens": sum(contexts), "max_context_len": max(contexts),
        "sample_count": len(batch.sample_indices),
        "prefill_has_history": any(r.phase != "decode" and r.start_pos > 0
                                   for r in batch.requests),
        "graph_mode": graph_mode,
    }
    validate_features(result)
    return result

# 校验提取的特征是否有效
def validate_features(features):
    for name in FEATURE_FIELDS[:-1]:
        if type(features.get(name)) is not int or features[name] < 0:
            raise ValueError(f"invalid feature: {name}")
    if type(features.get("prefill_has_history")) is not bool:
        raise ValueError("invalid prefill_has_history")
    if features.get("graph_mode") not in ("eager", "cuda_graph"):
        raise ValueError("unknown graph mode")
    d, p, n = (features[k] for k in FEATURE_FIELDS[:3])
    if d + p == 0 or bool(p) != bool(n) or n > p:
        raise ValueError("empty or inconsistent batch")
    if features["sample_count"] < d or features["sample_count"] > d + n:
        raise ValueError("inconsistent sample count")
    if features["max_context_len"] < 1 or features["sum_context_tokens"] < max(
            features["max_context_len"], d + p):
        raise ValueError("inconsistent context lengths")
    if features["sum_context_tokens"] > (d+n)*features["max_context_len"]:
        raise ValueError("context sum exceeds request count times maximum")
    if not p and features["prefill_has_history"]:
        raise ValueError("pure Decode has no Prefill history")
    if p and features["graph_mode"] != "eager":
        raise ValueError("V1 mixed/Prefill execution is eager")

# 将特征映射到离散桶中
def bucket_key(features, widths=DEFAULT_WIDTHS):
    validate_features(features)
    if set(widths) != set(DEFAULT_WIDTHS) or any(
            type(v) is not int or v <= 0 for v in widths.values()):
        raise ValueError("invalid bucket widths")
    values = []
    for name in FEATURE_FIELDS:
        value = features[name]
        if name in widths:
            value = ((value + widths[name] - 1) // widths[name]) * widths[name]
        values.append(value)
    return json.dumps([*values, features["graph_mode"]], separators=(",", ":"))


def fit_table(rows, identity, min_samples=30, widths=None):
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be positive")
    widths = dict(DEFAULT_WIDTHS if widths is None else widths)
    groups = {}
    for row in rows:
        if row.get("warmup"):
            continue
        value = row["step_time_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Step duration must be positive and finite")
        key = bucket_key(row["features"], widths)
        groups.setdefault(key, []).append(value)
    if not groups:
        raise ValueError("no measured samples")
    buckets = {key: {"count": len(values), "p95_ms": percentile(values, .95),
                     "max_ms": max(values)} for key, values in sorted(groups.items())}
    identity = json.loads(json.dumps(identity, allow_nan=False))
    return {"schema": SCHEMA, "identity": identity, "min_samples": min_samples,
            "bucket_widths": widths, "quantile": .95, "quantile_method": "nearest_rank",
            "buckets": buckets, "training_sample_count": sum(b["count"] for b in buckets.values()),
            "unsupported_fallback_ms": max(b["p95_ms"] for b in buckets.values()),
            "scope": "empirical estimate; supported does not guarantee an SLO"}


@dataclass(frozen=True)
class Prediction:
    predicted_ms: float
    supported: bool
    sample_count: int
    reason: str


class StepPredictor:
    def __init__(self, table, runtime_identity):
        runtime_identity = json.loads(json.dumps(runtime_identity, allow_nan=False))
        if table.get("schema") != SCHEMA or table.get("identity") != runtime_identity:
            raise ValueError("predictor schema or runtime identity differs")
        if table.get("quantile") != .95 or table.get("quantile_method") != "nearest_rank":
            raise ValueError("unsupported estimator")
        self.table = table
        if type(table.get("min_samples")) is not int or table["min_samples"] < 1:
            raise ValueError("invalid minimum samples")
        if not table.get("buckets"):
            raise ValueError("empty predictor")
        for key, bucket in table["buckets"].items():
            if type(bucket["count"]) is not int or bucket["count"] < 1:
                raise ValueError("invalid bucket count")
            if any(not math.isfinite(bucket[k]) or bucket[k] <= 0
                   for k in ("p95_ms", "max_ms")) or bucket["max_ms"] < bucket["p95_ms"]:
                raise ValueError("invalid bucket duration")
        if table["unsupported_fallback_ms"] != max(b["p95_ms"] for b in table["buckets"].values()):
            raise ValueError("invalid unsupported fallback")

    @classmethod
    def load(cls, path, runtime_identity):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), runtime_identity)

    def predict(self, features):
        key = bucket_key(features, self.table["bucket_widths"])
        bucket = self.table["buckets"].get(key)
        count = bucket["count"] if bucket else 0
        if bucket and count >= self.table["min_samples"]:
            return Prediction(bucket["p95_ms"], True, count, "exact_bucket")
        # A global P95 maximum is merely a diagnostic fallback. A future Slack
        # controller must reject/handle unsupported candidates explicitly.
        return Prediction(self.table["unsupported_fallback_ms"], False, count,
                          "sparse_bucket" if bucket else "out_of_coverage")

