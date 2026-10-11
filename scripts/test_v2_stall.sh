#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s tests -p test_v2_stall.py -v ;;
  --prepare|--verify|--smoke|--tune|--regress|--evaluate|--diagnose|--report)
    exec "$V0_PYTHON" -B -m benchmarks.v2_stall.suite "${mode#--}" "$@" ;;
  *)
    echo 'Usage: bash scripts/test_v2_stall.sh --cpu | --prepare/--verify/--smoke/--tune/--regress/--evaluate/--diagnose/--report --suite PATH [prepare options]' >&2
    exit 2 ;;
esac
