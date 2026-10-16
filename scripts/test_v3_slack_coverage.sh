#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to the one GPU used for this profile" >&2
    exit 2
fi

mkdir -p results/raw
profile_root="$(mktemp -d "$PWD/results/raw/v3-slack-coverage.XXXXXX")"
echo "V3 Slack coverage suite: $profile_root"

common_matrix=(
    --decode-counts 0,1,4,8,12,16
    --prefill-budgets 0,256,512,1024
    --contexts 256,1024
    --repeats 20
    --warmup 2
)

bash scripts/test_v3_profile.sh --smoke \
    --prefill-shape exact --out "$profile_root/smoke"

bash scripts/test_v3_profile.sh --collect \
    --split train --seed 621 --out "$profile_root/train-exact-621" \
    "${common_matrix[@]}" --prefill-request-counts 1,2 --prefill-shape exact

bash scripts/test_v3_profile.sh --collect \
    --split train --seed 622 --out "$profile_root/train-long-622" \
    "${common_matrix[@]}" --prefill-request-counts 1 --prefill-shape long

bash scripts/test_v3_profile.sh --collect \
    --split eval --seed 721 --out "$profile_root/eval-exact-721" \
    "${common_matrix[@]}" --prefill-request-counts 1,2 --prefill-shape exact

bash scripts/test_v3_profile.sh --collect \
    --split eval --seed 722 --out "$profile_root/eval-long-722" \
    "${common_matrix[@]}" --prefill-request-counts 1 --prefill-shape long

bash scripts/test_v3_profile.sh --fit \
    --profiles "$profile_root/train-exact-621" "$profile_root/train-long-622" \
    --min-samples 20 --decode-count-width 4 --sample-count-width 4 \
    --out "$profile_root/predictor"

bash scripts/test_v3_profile.sh --evaluate \
    --predictor "$profile_root/predictor/predictor.json" \
    --profiles "$profile_root/eval-exact-721" "$profile_root/eval-long-722" \
    --out "$profile_root/evaluation"

echo "V3 Slack coverage suite complete: $profile_root"
