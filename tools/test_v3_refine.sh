#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    export PYTHONPATH="$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
    exec "$V0_PYTHON" -B -m unittest discover -s tools/tests -p test_v3_refine.py -v ;;
  --prepare|--run|--verify)
    exec "$V0_PYTHON" -B -m tools.v3_refine "${mode#--}" "$@" ;;
  *)
    echo 'Usage: --cpu | --prepare --parent OLD_SUITE --suite NEW_SUITE | --run/--verify --suite NEW_SUITE' >&2
    exit 2 ;;
esac
