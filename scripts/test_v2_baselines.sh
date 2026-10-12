#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s tests -p test_v2_baselines.py -v ;;
  --prepare|--verify|--regress|--smoke|--report)
    exec "$V0_PYTHON" -B -m benchmarks.v2_baselines.suite "${mode#--}" "$@" ;;
  *)
    echo 'Usage: bash scripts/test_v2_baselines.sh --cpu | --prepare/--verify/--regress/--smoke/--report --suite PATH [prepare options]' >&2
    exit 2 ;;
esac
