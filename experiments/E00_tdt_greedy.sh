#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"

OUT="${RESULTS_DIR}/E00_tdt_greedy.jsonl"
decode_tdt \
  --manifest "${MANIFEST}" \
  --output "${OUT}" \
  --strategy greedy_batch \
  --batch-size "${BATCH_SIZE}"

echo "E00: ${OUT}"
