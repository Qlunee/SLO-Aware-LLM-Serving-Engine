"""
这个程序分别测试：
- INT8存储与转换；
- PyTorch FP8转换；
- Triton FP8转换；
- 软件E4M3FN编码和查表解码；
- 转换路径的Graph能力与时延。
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from .metrics import quantile


def decode_e4m3fn(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7

    if exponent == 15 and mantissa == 7:
        return math.nan
    if exponent == 0:
        return sign * mantissa * (2.0 ** -9)
    return sign * (1 + mantissa / 8) * (2.0 ** (exponent - 7))


POSITIVE_VALUES = [decode_e4m3fn(i) for i in range(127)]


def encode_e4m3fn(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("CPU参考对非有限输入明确拒绝")

    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), 448.0)
    index = bisect.bisect_left(POSITIVE_VALUES, magnitude)

    candidates = {
        max(0, min(126, index)),
        max(0, min(126, index - 1)),
    }
    # 距离相同优先最低有效位为0，即ties-to-even。
    code = min(
        candidates,
        key=lambda i: (abs(POSITIVE_VALUES[i] - magnitude), i & 1),
    )
    return sign | code


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def fp8_roundtrip_kernel(X, Q, Y, N: tl.constexpr, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        values = tl.load(X + offsets, offsets < N, other=0)
        encoded = values.to(tl.float8e4nv)
        tl.store(Q + offsets, encoded, offsets < N)
        tl.store(Y + offsets, encoded.to(tl.float32), offsets < N)

    @triton.jit
    def lut_decode_kernel(Q, LUT, Y, N: tl.constexpr, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        codes = tl.load(Q + offsets, offsets < N, other=0).to(tl.int32)
        values = tl.load(LUT + codes)
        tl.store(Y + offsets, values, offsets < N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--path", choices=["int8", "torch_fp8", "software_fp8", "triton_fp8", "triton_software_lut_decode"], help=argparse.SUPPRESS)
    parser.add_argument("--timeout-s", type=float, default=120)
    args = parser.parse_args()

    if min(args.rows, args.head_dim, args.repeats) < 1 or args.rows * args.head_dim < 256 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("至少测试256个元素，repeats必须为正")

    import torch

    report = {
        "schema_version": 1,
        "shape": [args.rows, args.head_dim],
        "format": "E4M3FN",
        "cpu_reference": {},
        "paths": {},
        "native_fp8_tensor_core_test": "not_tested",
        "note": "转换成功不表示原生FP8矩阵乘或Attention成功",
    }

    finite_codes = [
        code for code in range(256)
        if (code & 0x7F) != 0x7F
    ]
    bitwise_ok = all(
        encode_e4m3fn(decode_e4m3fn(code)) == code
        for code in finite_codes
    )
    report["cpu_reference"] = {
        "status": "pass" if bitwise_ok else "fail",
        "finite_code_roundtrip": bitwise_ok,
        "max_finite": decode_e4m3fn(126),
        "nonfinite_policy": "reject",
        "rounding": "nearest_ties_to_even",
        "overflow_policy": "saturate",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)

    if not torch.cuda.is_available():
        report["cuda"] = "unavailable"
        out.write_text(json.dumps(report, indent=2) + "\n")
        return

    torch.cuda.set_device(args.device)
    report["device"] = torch.cuda.get_device_name(args.device)
    report["compute_capability"] = list(
        torch.cuda.get_device_capability(args.device)
    )
    report["torch"] = torch.__version__
    report["triton"] = getattr(triton, "__version__", None)
    report["torch_cuda"] = torch.version.cuda
    report["timing_scope"] = "CUDA-event conversion path, including its allocations; not attention latency"
    if report["compute_capability"] == [8, 6]:
        report["native_fp8_tensor_core_test"] = "not_supported_on_SM86"

    # 编译器可能直接abort；每条GPU路线独立进程，避免整份报告丢失。
    if args.path is None:
        with tempfile.TemporaryDirectory(prefix="v0-quant-") as tmp:
            for name in ["int8", "torch_fp8", "software_fp8", "triton_fp8", "triton_software_lut_decode"]:
                child_out = Path(tmp) / f"{name}.json"
                command = [sys.executable, "-m", "benchmarks.v0.probe_quant_support",
                           "--path", name, "--device", str(args.device), "--rows", str(args.rows),
                           "--head-dim", str(args.head_dim), "--repeats", str(args.repeats),
                           "--out", str(child_out)]
                try:
                    child = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout_s)
                    if child.returncode == 0 and child_out.exists():
                        result = json.loads(child_out.read_text())["paths"][name]
                    else:
                        result = {"status": "unavailable", "returncode": child.returncode,
                                  "error": child.stderr[-6000:]}
                except subprocess.TimeoutExpired:
                    result = {"status": "unavailable", "error": "probe_timeout"}
                report["paths"][name] = result
                out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    torch.manual_seed(42)
    x_cpu = torch.randn(args.rows, args.head_dim).to(torch.bfloat16)
    edge_values = torch.tensor(
        [decode_e4m3fn(code) for code in finite_codes],
        dtype=torch.bfloat16,
    )
    x_cpu.flatten()[:len(edge_values)] = edge_values
    x = x_cpu.cuda()
    n = x.numel()

    expected_bits = torch.tensor(
        [encode_e4m3fn(float(v)) for v in x_cpu.flatten().tolist()],
        dtype=torch.uint8,
    ).reshape_as(x_cpu).cuda()

    lut = torch.tensor(
        [decode_e4m3fn(code) for code in range(256)],
        dtype=torch.float32,
        device="cuda",
    )
    midpoints = torch.tensor(
        [
            (a + b) / 2
            for a, b in zip(POSITIVE_VALUES, POSITIVE_VALUES[1:])
        ],
        dtype=torch.float32,
        device="cuda",
    )

    def software_fp8():
        values = x.float()
        magnitude = values.abs().clamp(max=448)
        indices = torch.bucketize(magnitude, midpoints)

        midpoint = midpoints[indices.clamp(max=125)]
        ties_up = (
            (indices < 126)
            & (magnitude == midpoint)
            & ((indices & 1) == 1)
        )
        indices = indices + ties_up.long()
        sign = torch.signbit(values).long() << 7
        codes = (indices | sign).to(torch.uint8)
        return codes, lut[codes.long()]

    def int8_path():
        codes = x.float().round().clamp(-127, 127).to(torch.int8)
        return codes, codes.float()

    def torch_fp8_path():
        dtype = getattr(torch, "float8_e4m3fn")
        codes = x.float().clamp(-448, 448).to(dtype)
        return codes, codes.float()

    fp8_buffer = None
    if hasattr(torch, "float8_e4m3fn"):
        fp8_buffer = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    decoded_buffer = torch.empty_like(x, dtype=torch.float32)

    def triton_fp8_path():
        if triton is None or fp8_buffer is None:
            raise RuntimeError("Triton或torch.float8_e4m3fn不可用")
        fp8_roundtrip_kernel[(triton.cdiv(n, 256),)](
            x, fp8_buffer, decoded_buffer, n, 256
        )
        return fp8_buffer, decoded_buffer

    def triton_lut_path():
        if triton is None:
            raise RuntimeError("Triton不可用")
        lut_decode_kernel[(triton.cdiv(n, 256),)](
            expected_bits, lut, decoded_buffer, n, 256
        )
        return expected_bits, decoded_buffer

    def measure(function):
        for _ in range(5):
            function()
        torch.cuda.synchronize()

        durations = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            function()
            end.record()
            end.synchronize()
            durations.append(begin.elapsed_time(end))
        return {
            "p50_ms": quantile(durations, 0.5),
            "p95_ms": quantile(durations, 0.95),
        }

    def graph_probe(function):
        input_buffer = expected_bits if args.path == "triton_software_lut_decode" else x
        original_input = input_buffer.clone()
        try:
            for _ in range(3):
                function()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = function()
            # 原地址更新内容，验证Graph不是仅能重放旧输入。
            input_buffer.copy_(original_input.flip(0))
            graph.replay()
            torch.cuda.synchronize()
            captured_codes = captured[0].view(torch.uint8).clone()
            captured_value = captured[1].clone()

            reference_codes, reference_value = function()
            torch.cuda.synchronize()
            matched = torch.equal(captured_codes, reference_codes.view(torch.uint8)) and torch.allclose(
                captured_value, reference_value, atol=0, rtol=0
            )
            return {"status": "pass" if matched else "fail", "dynamic_input_replay": True}
        except Exception as exc:
            return {"status": "fail", "error": str(exc)}
        finally:
            input_buffer.copy_(original_input)
            torch.cuda.synchronize()

    paths = {
        "int8": int8_path,
        "torch_fp8": torch_fp8_path,
        "software_fp8": software_fp8,
        "triton_fp8": triton_fp8_path,
        "triton_software_lut_decode": triton_lut_path,
    }

    for name, function in paths.items():
        if name != args.path:
            continue
        try:
            codes, decoded = function()
            torch.cuda.synchronize()

            if name == "int8":
                reference = x.float().round().clamp(-127, 127)
                correctness = torch.equal(decoded, reference)
            else:
                correctness = torch.equal(
                    codes.view(torch.uint8), expected_bits
                )
                correctness = correctness and torch.equal(
                    decoded, lut[expected_bits.long()]
                )

            error = decoded.float() - x.float()
            result = {
                "status": "pass" if correctness else "fail",
                "reference_match": bool(correctness),
                "storage_element_bytes": codes.element_size(),
                "storage_bytes": codes.numel() * codes.element_size(),
                "mse": error.square().mean().item(),
                "max_abs_error": error.abs().max().item(),
                "latency": measure(function),
                "graph": graph_probe(function),
            }
        except Exception as exc:
            result = {"status": "fail", "error": str(exc)}

        report["paths"][name] = result

    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
