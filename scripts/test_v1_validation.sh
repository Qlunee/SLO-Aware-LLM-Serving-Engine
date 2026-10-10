#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s tests -p test_v1_validation.py -v ;;
  --attention)
    exec "$V0_PYTHON" -B -m unittest discover -s tests -p test_v1_reference_gpu.py -v ;;
  --semantic|--calibrate|--quality)
    exec "$V0_PYTHON" -B -m benchmarks.v1_validation.run "${mode#--}" "$@" ;;
  *)
    echo 'Usage: bash scripts/test_v1_validation.sh --cpu | --attention | --semantic [args] | --calibrate [args] | --quality [args]' >&2
    exit 2 ;;
esac
