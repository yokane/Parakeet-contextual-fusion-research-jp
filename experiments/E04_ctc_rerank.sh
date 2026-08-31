#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"
require_file "${NGPU_LM}"
require_file "${CONTEXT_PHRASES}"

NBEST="${RESULTS_DIR}/E04_nbest.jsonl"
OUT="${RESULTS_DIR}/E04_ctc_rerank.jsonl"

python scripts/decode_nbest.py \
  --manifest "${MANIFEST}" \
  --output "${NBEST}" \
  --model "${MODEL_NAME}" \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ngram-lm-model "${NGPU_LM}" \
  --ngram-lm-alpha "${LM_ALPHA}" \
  --context-phrases "${CONTEXT_PHRASES}" \
  --boosting-tree-alpha "${PB_ALPHA}"

python scripts/rerank_ctc.py \
  --input "${NBEST}" \
  --output "${OUT}" \
  --model "${MODEL_NAME}" \
  --alpha "${CTC_ALPHA}"

echo "E04: ${OUT}"
