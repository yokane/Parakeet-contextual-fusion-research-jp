#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"
require_file "${NGPU_LM}"
require_file "${CONTEXT_PHRASES}"

OUT="${RESULTS_DIR}/E03_gpu_pb.jsonl"
decode_tdt \
  --manifest "${MANIFEST}" \
  --output "${OUT}" \
  --strategy malsd_batch \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ngram-lm-model "${NGPU_LM}" \
  --ngram-lm-alpha "${LM_ALPHA}" \
  --context-phrases "${CONTEXT_PHRASES}" \
  --boosting-tree-alpha "${PB_ALPHA}"

echo "E03: ${OUT}"
