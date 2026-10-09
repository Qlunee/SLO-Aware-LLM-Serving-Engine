## 负责生成和读取Trace
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments


SCHEMA_VERSION = 1


@dataclass
class RequestSpec:
    request_id: str
    arrival_ns: int
    prompt_token_ids: list[int]
    max_output_tokens: int
    ttft_slo_ms: float
    itl_slo_ms: float


def validate_trace(meta: dict, requests: list[RequestSpec]) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的Trace schema_version")
    if meta.get("mode") not in {"poisson", "simultaneous"}:
        raise ValueError("不支持的到达模式")
    if not isinstance(meta.get("model_revision"), str) or not meta["model_revision"].strip():
        raise ValueError("模型revision不能为空")
    warmup, measurement = meta.get("warmup_s"), meta.get("measurement_s")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [warmup, measurement]):
        raise ValueError("实验时长必须是有限数值")
    if warmup < 0 or measurement <= 0 or not math.isfinite(warmup + measurement):
        raise ValueError("Warmup必须非负，测量窗口必须为正")
    if meta.get("request_count") != len(requests):
        raise ValueError("metadata中的请求数与Trace不一致")
    if meta["mode"] == "poisson":
        rate = meta.get("rps")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Poisson rps必须有限且为正")
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("request_id重复")
    stop_ns = round((warmup + measurement) * 1e9)
    previous = -1
    for request in requests:
        if not isinstance(request.request_id, str) or not request.request_id:
            raise ValueError("request_id必须为非空字符串")
        if type(request.arrival_ns) is not int or not 0 <= request.arrival_ns < stop_ns:
            raise ValueError("到达时间必须为窗口内的整数纳秒")
        if request.arrival_ns < previous:
            raise ValueError("Trace必须按到达时间排序")
        if meta["mode"] == "simultaneous" and (warmup != 0 or request.arrival_ns != 0):
            raise ValueError("simultaneous要求Warmup和到达时间均为0")
        if not isinstance(request.prompt_token_ids, list) or not request.prompt_token_ids:
            raise ValueError("Prompt必须为非空Token列表")
        if any(type(t) is not int for t in request.prompt_token_ids):
            raise ValueError("Token ID必须为整数")
        if type(request.max_output_tokens) is not int or request.max_output_tokens < 1:
            raise ValueError("输出长度必须为正整数")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                   for v in [request.ttft_slo_ms, request.itl_slo_ms]):
            raise ValueError("SLO必须有限且为正")
        previous = request.arrival_ns


def load_trace(path: str | Path) -> tuple[dict, list[RequestSpec]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "metadata":
        raise ValueError("Trace首行必须是metadata")

    meta = rows[0]
    requests = [RequestSpec(**row) for row in rows[1:]]

    validate_trace(meta, requests)
    return meta, requests


def trace_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(args: argparse.Namespace) -> tuple[dict, list[RequestSpec]]:
    from transformers import AutoConfig, AutoTokenizer

    if not all(math.isfinite(v) for v in [args.warmup_s, args.measurement_s, args.rps]):
        raise ValueError("时长和到达率必须为有限值")
    if args.warmup_s < 0 or args.measurement_s <= 0:
        raise ValueError("Warmup必须非负，测量窗口必须为正")
    if args.output_tokens < 1 or min(args.input_lengths) < 1:
        raise ValueError("输入输出长度必须为正")
    if args.ttft_slo_ms <= 0 or args.itl_slo_ms <= 0:
        raise ValueError("SLO必须为正")
    if args.mode == "simultaneous" and args.warmup_s != 0:
        raise ValueError("零间隔正确性Trace要求warmup_s=0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    # 限制在模型和Tokenizer共同支持的ID范围内。
    vocabulary_size = min(len(tokenizer), config.vocab_size)
    special_ids = set(tokenizer.all_special_ids)
    allowed_ids = [
        token_id
        for token_id in range(vocabulary_size)
        if token_id not in special_ids
    ]
    if not allowed_ids:
        raise ValueError("没有可用于合成输入的Token")

    arrival_rng = random.Random(args.seed)
    prompt_rng = random.Random(args.seed + 1)
    stop_s = args.warmup_s + args.measurement_s

    if args.mode == "simultaneous":
        if args.requests < 1:
            raise ValueError("requests必须为正")
        arrivals = [0] * args.requests
    else:
        if args.rps <= 0:
            raise ValueError("rps必须为正")
        arrivals = []
        current_s = 0.0
        while True:
            current_s += arrival_rng.expovariate(args.rps)
            if current_s >= stop_s:
                break
            arrival_ns = round(current_s * 1e9)
            if arrival_ns < round(stop_s * 1e9):
                arrivals.append(arrival_ns)

    requests = []
    for index, arrival_ns in enumerate(arrivals):
        input_length = prompt_rng.choice(args.input_lengths)
        prompt = prompt_rng.choices(allowed_ids, k=input_length)
        requests.append(
            RequestSpec(
                request_id=f"r{index:08d}",
                arrival_ns=arrival_ns,
                prompt_token_ids=prompt,
                max_output_tokens=args.output_tokens,
                ttft_slo_ms=args.ttft_slo_ms,
                itl_slo_ms=args.itl_slo_ms,
            )
        )

    meta = {
        "kind": "metadata",
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "rps": args.rps if args.mode == "poisson" else None,
        "warmup_s": args.warmup_s,
        "measurement_s": args.measurement_s,
        "model_revision": args.model_revision,
        "vocabulary_size": vocabulary_size,
        "input_lengths": args.input_lengths,
        "output_tokens": args.output_tokens,
        "request_count": len(requests),
        "synthetic_tokens": True,
        "model_config_sha256": hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest(),
    }
    validate_trace(meta, requests)
    return meta, requests


def main() -> None:
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument(
        "--mode", choices=["poisson", "simultaneous"], default="poisson"
    )
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmup-s", type=float, default=10)
    parser.add_argument("--measurement-s", type=float, default=60)
    parser.add_argument("--input-lengths", type=int, nargs="+",
                        default=[256, 1024, 2048])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--itl-slo-ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = resolve_model_arguments(parser.parse_args(), parser)

    meta, requests = generate(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 避免覆盖已有实验Trace。
    with path.open("x") as file:
        file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for request in requests:
            file.write(json.dumps(asdict(request)) + "\n")

    print(json.dumps({
        "trace": str(path),
        "requests": len(requests),
        "sha256": trace_sha256(path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
