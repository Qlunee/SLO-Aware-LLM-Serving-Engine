#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s diagnostics/v2_bounded/tests -v ;;
  --prepare|--verify|--plan|--smoke|--batch|--report|--run|--retry)
    exec "$V0_PYTHON" -B -m diagnostics.v2_bounded "${mode#--}" "$@" ;;
  *)
    echo 'Usage: bash diagnostics/v2_bounded/test.sh --cpu | --prepare/--verify/--plan/--smoke/--batch/--report/--run/--retry --suite PATH' >&2
    exit 2 ;;
esac
