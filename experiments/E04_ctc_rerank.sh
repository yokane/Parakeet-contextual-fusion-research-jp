#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"
require_file "${NGPU_LM}"
require_file "${CONTEXT_PHRASES}"

NBEST="${RESULTS_DIR}/E04_nbest.jsonl"
OUT="${RESULTS_DIR}/E04_ctc_rerank.jsonl"

decode_tdt \
  --manifest "${MANIFEST}" \
  --output "${NBEST}" \
  --strategy malsd_batch \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ngram-lm-model "${NGPU_LM}" \
  --ngram-lm-alpha "${LM_ALPHA}" \
  --context-phrases "${CONTEXT_PHRASES}" \
  --boosting-tree-alpha "${PB_ALPHA}"

uv run --locked python scripts/rerank_ctc.py \
  --input "${NBEST}" \
  --output "${OUT}" \
  --model-revision "${MODEL_REVISION}" \
  --alpha "${CTC_ALPHA}"

echo "E04: ${OUT}"
