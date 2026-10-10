#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
"$V0_PYTHON" -B -m unittest discover -s tests -p 'test_v1_state.py' -v
if [[ "${1:---cpu}" == "--cpu" ]]; then
  exit 0
fi
if [[ "$1" != "--gpu" ]]; then
  echo 'Usage: bash scripts/test_v1.sh --cpu | --gpu [--tp 1|2] [--graph] [--batch N] [--chunk N] [--token-budget N] [--repeats N]' >&2
  exit 2
fi
shift
tp=1
batch=8
chunk=256
budget=512
repeats=1
seed=41
prompt_kind=natural
graph=()
while (($#)); do
  case "$1" in
    --tp) tp="$2"; shift 2 ;;
    --graph) graph=(--graph); shift ;;
    --batch) batch="$2"; shift 2 ;;
    --chunk) chunk="$2"; shift 2 ;;
    --token-budget) budget="$2"; shift 2 ;;
    --repeats) repeats="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --prompt-kind) prompt_kind="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo '--repeats must be a positive integer' >&2
  exit 2
fi
"$V0_PYTHON" -m benchmarks.v0.preflight --tp "$tp"
"$V0_PYTHON" -m unittest discover -s tests -p 'test_v1_attention_gpu.py' -v
mkdir -p results/raw
test_root="$(mktemp -d "$PWD/results/raw/v1-check.XXXXXX")"
echo "V1 artifacts: $test_root"
failures=()
for ((repeat=1; repeat<=repeats; repeat++)); do
mkdir -p "$test_root/repeat_$repeat"
for case_name in mixed prefix preempt; do
  extra=()
  if [[ "$case_name" == preempt ]]; then extra=(--kv-block-cap 2); fi
  if ! "$V0_PYTHON" -m benchmarks.v1.check_v1 candidate \
    --case "$case_name" --tp "$tp" "${graph[@]}" "${extra[@]}" \
    --batch "$batch" --chunk "$chunk" --token-budget "$budget" --seed "$seed" \
    --prompt-kind "$prompt_kind" \
    --out "$test_root/repeat_$repeat/$case_name-candidate" 2>&1 | tee "$test_root/repeat_$repeat/$case_name-candidate.log"; then
    failures+=("repeat_$repeat/$case_name-candidate")
    continue
  fi
  if ! "$V0_PYTHON" -m benchmarks.v1.check_v1 reference \
    --numerical-policy bf16 \
    --candidate "$test_root/repeat_$repeat/$case_name-candidate" --out "$test_root/repeat_$repeat/$case_name-reference" \
    2>&1 | tee "$test_root/repeat_$repeat/$case_name-reference.log"; then
    failures+=("repeat_$repeat/$case_name-reference")
  fi
done
done
if ((${#failures[@]})); then
  printf '%s\n' "${failures[@]}" > "$test_root/failed_cases.txt"
  echo "V1 correctness checks FAILED: $test_root/failed_cases.txt" >&2
  exit 1
fi
echo "V1 correctness checks passed: $test_root"
