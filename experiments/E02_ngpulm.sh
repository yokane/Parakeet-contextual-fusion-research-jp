#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"
require_file "${NGPU_LM}"

OUT="${RESULTS_DIR}/E02_ngpulm.jsonl"
decode_tdt \
  --manifest "${MANIFEST}" \
  --output "${OUT}" \
  --strategy malsd_batch \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ngram-lm-model "${NGPU_LM}" \
  --ngram-lm-alpha "${LM_ALPHA}"

echo "E02: ${OUT}"
