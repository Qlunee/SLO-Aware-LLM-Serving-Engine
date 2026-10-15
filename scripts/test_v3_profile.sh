#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s tests -p test_v3_profile.py -v ;;
  --smoke)
    exec "$V0_PYTHON" -B -m benchmarks.profile_step_time collect --smoke --split smoke --seed 41 "$@" ;;
  --collect|--fit|--evaluate|--inspect)
    exec "$V0_PYTHON" -B -m benchmarks.profile_step_time "${mode#--}" "$@" ;;
  *)
    echo 'Usage: --cpu | --smoke --out PATH | --collect --split train/eval --seed N --out PATH | --fit --profiles PATH... --out PATH | --evaluate --predictor FILE --profiles PATH... --out PATH | --inspect --profiles PATH...' >&2
    exit 2 ;;
esac
