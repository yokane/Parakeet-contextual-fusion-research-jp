#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

PHONE_HEAD="${PHONE_HEAD:-artifacts/phone_head.pt}"
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-artifacts/encoder}"
INPUT="${INPUT:-${RESULTS_DIR}/E04_phone_ready.jsonl}"
OUT="${RESULTS_DIR}/E05_phone_rerank.jsonl"

require_file "${INPUT}"
require_file "${PHONE_HEAD}"
if [[ ! -d "${ENCODER_FEATURE_DIR}" ]]; then
  echo "Encoder feature directory not found: ${ENCODER_FEATURE_DIR}" >&2
  exit 2
fi

uv run --locked python scripts/rerank_phone.py \
  --input "${INPUT}" \
  --output "${OUT}" \
  --checkpoint "${PHONE_HEAD}" \
  --feature-dir "${ENCODER_FEATURE_DIR}" \
  --alpha "${PHONE_ALPHA}"

echo "E05: ${OUT}"
