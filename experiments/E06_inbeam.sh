#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

: "${E06_DRIVER:?Set E06_DRIVER to a NeMo-version-specific in-beam fusion driver}"

if [[ ! -f "${E06_DRIVER}" ]]; then
  echo "E06 driver not found: ${E06_DRIVER}" >&2
  exit 2
fi

python "${E06_DRIVER}" \
  --manifest "${MANIFEST}" \
  --model "${MODEL_NAME}" \
  --beam-size "${BEAM_SIZE}" \
  --ngram-lm-model "${NGPU_LM}" \
  --ngram-lm-alpha "${LM_ALPHA}" \
  --context-phrases "${CONTEXT_PHRASES}" \
  --boosting-tree-alpha "${PB_ALPHA}" \
  --ctc-alpha "${CTC_ALPHA}" \
  --phone-alpha "${PHONE_ALPHA}" \
  --output "${RESULTS_DIR}/E06_inbeam.jsonl"
