#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env_qwen3_4b.sh
mode="${1:---cpu}"
if (($#)); then shift; fi
case "$mode" in
  --cpu)
    exec "$V0_PYTHON" -B -m unittest discover -s diagnostics/online_token_replay/tests -v ;;
  --prepare|--capture|--replay|--report)
    exec "$V0_PYTHON" -B -m diagnostics.online_token_replay "${mode#--}" "$@" ;;
  *) echo 'Usage: test.sh --cpu | --prepare/--capture/--replay/--report [options]' >&2; exit 2 ;;
esac
