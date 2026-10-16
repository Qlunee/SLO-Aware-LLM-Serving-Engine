#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_qwen3_4b.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to the GPU used for the existing V3 profiles" >&2
    exit 2
fi

base=results/raw/v3-slack-hetero.038cUU
old=results/raw/v3-slack-coverage.Nrms3Z
for profile in "$old/train-exact-621" "$old/train-long-622" \
               "$base/train-exact-731" "$base/train-long-732"; do
    if [[ ! -f "$profile/artifact_hashes.json" ]]; then
        echo "Missing sealed training profile: $profile" >&2
        exit 2
    fi
done

root="$(mktemp -d "$PWD/results/raw/v3-slack-uneven.XXXXXX")"
echo "V3 uneven-context profile suite: $root"

# Alternating 128/1024 Decode contexts fills the sum-context buckets missed
# by the earlier 256/1024 50:50 profile near D9/D10 online candidates.
matrix=(
    --decode-counts 8,9,10,11
    --prefill-budgets 0,256,512,1024
    --contexts 128
    --alternate-context-tokens 1024
    --prefill-request-counts 1,2
    --repeats 12 --warmup 2
)

bash scripts/test_v3_profile.sh --smoke \
    --prefill-shape exact --alternate-context-tokens 1024 --out "$root/smoke"

bash scripts/test_v3_profile.sh --collect --split train --seed 741 \
    --prefill-shape exact "${matrix[@]}" --out "$root/train-exact-741"
bash scripts/test_v3_profile.sh --collect --split train --seed 742 \
    --prefill-shape long "${matrix[@]}" --out "$root/train-long-742"

bash scripts/test_v3_profile.sh --fit \
    --profiles "$old/train-exact-621" "$old/train-long-622" \
               "$base/train-exact-731" "$base/train-long-732" \
               "$root/train-exact-741" "$root/train-long-742" \
    --min-samples 10 --decode-count-width 4 --sample-count-width 4 \
    --out "$root/predictor"

bash scripts/test_v3_profile.sh --collect --split eval --seed 841 \
    --prefill-shape exact "${matrix[@]}" --out "$root/eval-exact-841"
bash scripts/test_v3_profile.sh --collect --split eval --seed 842 \
    --prefill-shape long "${matrix[@]}" --out "$root/eval-long-842"

bash scripts/test_v3_profile.sh --evaluate \
    --predictor "$root/predictor/predictor.json" \
    --profiles "$root/eval-exact-841" "$root/eval-long-842" \
    --out "$root/evaluation"

echo "V3 uneven-context profile suite complete: $root"
