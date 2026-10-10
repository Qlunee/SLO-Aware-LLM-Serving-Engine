#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
tp=1
graph=()
while (($#)); do
  case "$1" in
    --tp) tp="$2"; shift 2 ;;
    --graph) graph=(--graph); shift ;;
    *) echo "Usage: bash scripts/test_v1_online.sh [--tp 1|2] [--graph]" >&2; exit 2 ;;
  esac
done
mkdir -p results/raw
online_root="$(mktemp -d "$PWD/results/raw/v1-online.XXXXXX")"
echo "V1 online artifacts: $online_root"
common=(--tp "$tp" "${graph[@]}" --sampler greedy --drain-s 30
        --max-model-len 4096 --max-num-seqs 64 --token-budget 4096
        --gpu-memory-utilization 0.8 --seed 41)
"$V0_PYTHON" -m benchmarks.v0.workload --mode simultaneous --requests 4 \
  --warmup-s 0 --measurement-s 10 --input-lengths 255 257 513 \
  --output-tokens 8 --seed 41 --out "$online_root/random-simultaneous.jsonl"
"$V0_PYTHON" -m benchmarks.v1.prepare_trace --trace "$online_root/random-simultaneous.jsonl" \
  --out "$online_root/simultaneous.jsonl"
for mode in native minimal full; do
  "$V0_PYTHON" -m benchmarks.v1.run "${common[@]}" --chunk-size 256 \
    --trace "$online_root/simultaneous.jsonl" --observation "$mode" --out "$online_root/$mode"
  "$V0_PYTHON" -m benchmarks.v0.validate_results --run-dir "$online_root/$mode"
  "$V0_PYTHON" -m benchmarks.v0.check_correctness --observed-dir "$online_root/$mode" \
    --out "$online_root/$mode-greedy.json"
done
"$V0_PYTHON" -m benchmarks.v0.workload --mode poisson --rps 2 \
  --warmup-s 2 --measurement-s 8 --input-lengths 256 1024 2048 \
  --output-tokens 32 --seed 41 --out "$online_root/poisson.jsonl"
"$V0_PYTHON" -m benchmarks.v0.run "${common[@]}" --observation full \
  --trace "$online_root/poisson.jsonl" --out "$online_root/legacy"
"$V0_PYTHON" -m benchmarks.v1.run "${common[@]}" --observation full --chunk-size 256 \
  --trace "$online_root/poisson.jsonl" --out "$online_root/mixed"
for engine in legacy mixed; do
  "$V0_PYTHON" -m benchmarks.v0.validate_results --run-dir "$online_root/$engine"
done
"$V0_PYTHON" -m benchmarks.v1.compare_runs --baseline "$online_root/legacy" \
  --candidate "$online_root/mixed" --out "$online_root/comparison.json"
echo "V1 online smoke passed: $online_root (NOT a repeated performance experiment)"
