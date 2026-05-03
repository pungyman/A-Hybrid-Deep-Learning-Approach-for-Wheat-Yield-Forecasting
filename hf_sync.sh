#!/usr/bin/env bash
# Sync data and model artifacts with Hugging Face dataset repo.
#
# Usage:
#   ./hf_sync.sh push          # Upload all data + models to HF
#   ./hf_sync.sh pull          # Download all data + models from HF
#   ./hf_sync.sh push-models   # Upload only saved models
#   ./hf_sync.sh pull-data     # Download only the datasets (rf_flat + final)
#
# Prereqs:
#   brew install hf
#   hf auth login

set -euo pipefail

HF_REPO="pungyman/winter-wheat-yield-forecasting-india"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

RF_FLAT_DIR="data/datasets/rf_flat"
FINAL_DIR="data/datasets/final"
MODELS_DIR="src/rnn/saved_models"
HPO_DIR="experiments/hpo"

usage() {
    echo "Usage: $0 {push|pull|push-models|pull-data}"
    exit 1
}

push_all() {
    echo "==> Uploading data, models, and HPO artifacts to HF..."
    cd "$REPO_ROOT"
    hf upload "$HF_REPO" "$RF_FLAT_DIR" "$RF_FLAT_DIR" --repo-type=dataset
    hf upload "$HF_REPO" "$FINAL_DIR"   "$FINAL_DIR"   --repo-type=dataset
    hf upload "$HF_REPO" "$MODELS_DIR"  "$MODELS_DIR"  --repo-type=dataset
    if [ -d "$HPO_DIR" ]; then
        hf upload "$HF_REPO" "$HPO_DIR" "$HPO_DIR"    --repo-type=dataset
    fi
    echo "==> Done."
}

pull_all() {
    echo "==> Downloading data, models, and HPO artifacts from HF..."
    cd "$REPO_ROOT"
    hf download "$HF_REPO" --include "$RF_FLAT_DIR/**" --repo-type=dataset --local-dir .
    hf download "$HF_REPO" --include "$FINAL_DIR/**"   --repo-type=dataset --local-dir .
    hf download "$HF_REPO" --include "$MODELS_DIR/**"  --repo-type=dataset --local-dir .
    hf download "$HF_REPO" --include "$HPO_DIR/**"     --repo-type=dataset --local-dir . 2>/dev/null || true
    echo "==> Done."
}

push_models() {
    echo "==> Uploading saved models to HF..."
    cd "$REPO_ROOT"
    hf upload "$HF_REPO" "$MODELS_DIR" "$MODELS_DIR" --repo-type=dataset
    if [ -d "$HPO_DIR" ]; then
        hf upload "$HF_REPO" "$HPO_DIR" "$HPO_DIR"   --repo-type=dataset
    fi
    echo "==> Done."
}

pull_data() {
    echo "==> Downloading datasets (rf_flat + final) from HF..."
    cd "$REPO_ROOT"
    hf download "$HF_REPO" --include "$RF_FLAT_DIR/**" --repo-type=dataset --local-dir .
    hf download "$HF_REPO" --include "$FINAL_DIR/**"   --repo-type=dataset --local-dir .
    echo "==> Done."
}

case "${1:-}" in
    push)        push_all ;;
    pull)        pull_all ;;
    push-models) push_models ;;
    pull-data)   pull_data ;;
    *)           usage ;;
esac
