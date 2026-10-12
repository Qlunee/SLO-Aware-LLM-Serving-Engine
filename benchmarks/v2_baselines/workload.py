"""Independent open-loop arrivals and synthetic text fixtures, not quality data."""
from dataclasses import asdict, dataclass
import math
import random

from benchmarks.v0.workload import RequestSpec


@dataclass(frozen=True)
class Workload:
    warmup_s: float = 2
    measurement_s: float = 8
    rps: float = 2
    short_tokens: int = 256
    long_tokens: int = 1024
    long_fraction: float = .2
    output_tokens: int = 32
    ttft_slo_ms: float = 1000
    itl_slo_ms: float = 100
    burst_period_s: float = 2
    burst_fraction: float = .25
    burst_multiplier: float = 3

    def validate(self):
        for name in ("measurement_s", "rps", "ttft_slo_ms", "itl_slo_ms", "burst_period_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0:
            raise ValueError("invalid warmup")
        if not math.isfinite(self.warmup_s+self.measurement_s):
            raise ValueError("invalid total duration")
        for name in ("short_tokens", "long_tokens", "output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.long_tokens < self.short_tokens or not 0 <= self.long_fraction <= 1:
            raise ValueError("invalid length mixture")
        if not (0 < self.burst_fraction < 1 and math.isfinite(self.burst_multiplier)
                and 1 < self.burst_multiplier < 1/self.burst_fraction):
            raise ValueError("burst must have positive low rate and unit cycle-average multiplier")


def _burst_time(unit_time, config):
    period = config.burst_period_s
    cycle = math.floor(unit_time/period)
    remainder = unit_time-cycle*period
    high_duration = period*config.burst_fraction
    high_hazard = high_duration*config.burst_multiplier
    low = (1-config.burst_fraction*config.burst_multiplier)/(1-config.burst_fraction)
    offset = remainder/config.burst_multiplier if remainder < high_hazard else (
        high_duration+(remainder-high_hazard)/low)
    return cycle*period+offset


def build_trace(config, mode, seed, token_factory, revision):
    config.validate()
    if mode not in ("poisson", "burst", "long_short"):
        raise ValueError("unknown workload")
    if type(seed) is not int or seed < 0 or not revision.strip():
        raise ValueError("nonnegative seed and model revision required")
    arrivals, lengths = random.Random(seed), random.Random(seed+1_000_003)
    stop = config.warmup_s+config.measurement_s
    rows, unit_time = [], 0.0
    while True:
        unit_time += arrivals.expovariate(config.rps)
        timestamp = _burst_time(unit_time, config) if mode == "burst" else unit_time
        if timestamp >= stop:
            break
        ns = round(timestamp*1e9)
        if ns >= round(stop*1e9):
            break
        index = len(rows)
        long = mode == "long_short" and lengths.random() < config.long_fraction
        count = config.long_tokens if long else config.short_tokens
        tokens = token_factory(count, index)
        if len(tokens) != count:
            raise ValueError("token factory returned wrong input length")
        rows.append(asdict(RequestSpec(
            f"r{index:08d}", ns, list(tokens), config.output_tokens,
            config.ttft_slo_ms, config.itl_slo_ms,
        )))
    trace = {"schema": "v2-online-trace-v1", "mode": mode, "seed": seed,
             "model_revision": revision, "warmup_s": config.warmup_s,
             "measurement_s": config.measurement_s, "workload": asdict(config),
             "synthetic_language_fixture": True, "not_real_user_data": True,
             "request_count": len(rows), "requests": rows}
    validate_trace(trace)
    return trace


def validate_trace(trace):
    if trace.get("schema") != "v2-online-trace-v1" or trace.get("mode") not in (
            "poisson", "burst", "long_short"):
        raise ValueError("invalid trace schema/mode")
    config = Workload(**trace["workload"])
    config.validate()
    if (trace["warmup_s"] != config.warmup_s or trace["measurement_s"] != config.measurement_s
            or type(trace["seed"]) is not int or trace["seed"] < 0
            or not isinstance(trace["model_revision"], str) or not trace["model_revision"].strip()):
        raise ValueError("trace metadata mismatch")
    rows = trace["requests"]
    if trace["request_count"] != len(rows):
        raise ValueError("request count mismatch")
    ids, previous = set(), -1
    stop = round((config.warmup_s+config.measurement_s)*1e9)
    for row in rows:
        spec = RequestSpec(**row)
        if not isinstance(spec.request_id, str) or not spec.request_id or spec.request_id in ids:
            raise ValueError("invalid/duplicate request id")
        ids.add(spec.request_id)
        if type(spec.arrival_ns) is not int or not 0 <= spec.arrival_ns < stop or spec.arrival_ns < previous:
            raise ValueError("invalid/unsorted arrival")
        previous = spec.arrival_ns
        allowed_lengths = (config.short_tokens, config.long_tokens) if trace["mode"] == "long_short" else (config.short_tokens,)
        if (not isinstance(spec.prompt_token_ids, list) or len(spec.prompt_token_ids) not in allowed_lengths
                or any(type(t) is not int or t < 0 for t in spec.prompt_token_ids)
                or spec.max_output_tokens != config.output_tokens
                or spec.ttft_slo_ms != config.ttft_slo_ms or spec.itl_slo_ms != config.itl_slo_ms):
            raise ValueError("request does not match frozen workload")
    return [RequestSpec(**row) for row in rows]
