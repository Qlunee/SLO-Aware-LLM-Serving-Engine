#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to the GPU used for the existing V3 profiles" >&2
    exit 2
fi

old=results/raw/v3-slack-coverage.Nrms3Z
if [[ ! -f "$old/train-exact-621/artifact_hashes.json" ||
      ! -f "$old/train-long-622/artifact_hashes.json" ]]; then
    echo "Existing sealed V3 training profiles are missing: $old" >&2
    exit 2
fi

mkdir -p results/raw
root="$(mktemp -d "$PWD/results/raw/v3-slack-hetero.XXXXXX")"
echo "V3 mixed-context profile suite: $root"

matrix=(
    --decode-counts 5,6,7,8,9,10,11,12
    --prefill-budgets 0,256,512,1024
    --contexts 256
    --alternate-context-tokens 1024
    --prefill-request-counts 1,2
    --repeats 12 --warmup 2
)

bash scripts/test_v3_profile.sh --smoke \
    --prefill-shape exact --alternate-context-tokens 1024 --out "$root/smoke"

bash scripts/test_v3_profile.sh --collect --split train --seed 731 \
    --prefill-shape exact "${matrix[@]}" --out "$root/train-exact-731"
bash scripts/test_v3_profile.sh --collect --split train --seed 732 \
    --prefill-shape long "${matrix[@]}" --out "$root/train-long-732"

bash scripts/test_v3_profile.sh --fit \
    --profiles "$old/train-exact-621" "$old/train-long-622" \
               "$root/train-exact-731" "$root/train-long-732" \
    --min-samples 10 --decode-count-width 4 --sample-count-width 4 \
    --out "$root/predictor"

bash scripts/test_v3_profile.sh --collect --split eval --seed 831 \
    --prefill-shape exact "${matrix[@]}" --out "$root/eval-exact-831"
bash scripts/test_v3_profile.sh --collect --split eval --seed 832 \
    --prefill-shape long "${matrix[@]}" --out "$root/eval-long-832"

bash scripts/test_v3_profile.sh --evaluate \
    --predictor "$root/predictor/predictor.json" \
    --profiles "$root/eval-exact-831" "$root/eval-long-832" \
    --out "$root/evaluation"

echo "V3 mixed-context profile suite complete: $root"
