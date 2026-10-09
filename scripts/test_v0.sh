#!/usr/bin/env bash
# CPU默认不依赖GPU；GPU测试逐个进程执行，不能并行占用上游NCCL端口。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
V0_PYTHON="${V0_PYTHON:-python3}"
mode="${1:---cpu}"
if [[ "$mode" != "--cpu" && "$mode" != "--gpu" ]]; then
    echo "Usage: V0_PYTHON=/path/python bash scripts/test_v0.sh --cpu|--gpu [--model DIR --model-revision SHA --tp 1 --repeats 3 --graph]" >&2
    exit 2
fi
shift || true
model="${V0_MODEL:-}"; revision="${V0_MODEL_REVISION:-}"; tp=1; repeats=3; graph=()
while (( $# )); do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --model-revision) revision="$2"; shift 2 ;;
        --tp) tp="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --graph) graph=(--graph); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
"$V0_PYTHON" -m unittest discover -s tests -p 'test_v0*.py' -v
for program in workload run sweep check_correctness preflight probe_quant_support validate_results; do
    "$V0_PYTHON" -m "benchmarks.v0.$program" --help >/dev/null
done
if [[ "$mode" == "--cpu" ]]; then exit 0; fi
if [[ -z "$model" ]]; then
    model="$("$V0_PYTHON" -m benchmarks.defaults --field model_path)"
fi
if [[ -z "$revision" ]]; then
    revision="$("$V0_PYTHON" -m benchmarks.defaults --revision-for "$model")"
fi
[[ -n "$model" && -n "$revision" && "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "GPU测试要求model、真实model-revision和正整数repeats" >&2; exit 2; }
"$V0_PYTHON" -m benchmarks.v0.preflight --model "$model" --tp "$tp"
mkdir -p results/raw
result_dir="$(mktemp -d results/raw/v0-test.XXXXXX)"
echo "Results: $PWD/$result_dir"
"$V0_PYTHON" -m benchmarks.v0.probe_quant_support --rows 32 --head-dim 128 --repeats 10 --out "$result_dir/quant_support.json"
"$V0_PYTHON" -m benchmarks.v0.workload --model "$model" --model-revision "$revision" \
    --mode simultaneous --requests 8 --warmup-s 0 --measurement-s .01 \
    --input-lengths 32 256 512 --output-tokens 8 --out "$result_dir/simultaneous.jsonl"
for (( repeat=1; repeat<=repeats; repeat++ )); do
    for observation in native minimal full; do
        run_dir="$result_dir/${observation}_${repeat}"
        "$V0_PYTHON" -m benchmarks.v0.run --model "$model" --model-revision "$revision" \
            --trace "$result_dir/simultaneous.jsonl" --out "$run_dir" --tp "$tp" \
            --max-model-len 1024 --max-num-seqs 8 --token-budget 256 --drain-s 60 \
            --gpu-memory-utilization .5 --observation "$observation" "${graph[@]}" >"${run_dir}.log" 2>&1
        "$V0_PYTHON" -m benchmarks.v0.validate_results --run-dir "$run_dir"
        "$V0_PYTHON" -m benchmarks.v0.check_correctness --observed-dir "$run_dir" --out "$run_dir/correctness.json" >"${run_dir}_correctness.log" 2>&1
    done
done
# 一个开放环烟雾测试，不用此短窗口估计饱和点或性能收益。
"$V0_PYTHON" -m benchmarks.v0.workload --model "$model" --model-revision "$revision" \
    --rps 10 --warmup-s .2 --measurement-s 2 --input-lengths 32 256 512 \
    --output-tokens 8 --out "$result_dir/poisson.jsonl"
"$V0_PYTHON" -m benchmarks.v0.run --model "$model" --model-revision "$revision" \
    --trace "$result_dir/poisson.jsonl" --out "$result_dir/poisson" --tp "$tp" \
    --max-model-len 1024 --max-num-seqs 8 --token-budget 256 --drain-s 60 \
    --gpu-memory-utilization .5 "${graph[@]}" >"$result_dir/poisson.log" 2>&1
"$V0_PYTHON" -m benchmarks.v0.validate_results --run-dir "$result_dir/poisson"
echo "GPU smoke/regression passed. Artifacts: $PWD/$result_dir"
