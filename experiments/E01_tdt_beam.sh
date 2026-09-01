#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"

OUT="${RESULTS_DIR}/E01_tdt_beam.jsonl"
decode_tdt \
  --manifest "${MANIFEST}" \
  --output "${OUT}" \
  --strategy malsd_batch \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}"

echo "E01: ${OUT}"
