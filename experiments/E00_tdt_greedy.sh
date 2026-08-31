#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"

nemo_eval \
  pretrained_name="${MODEL_NAME}" \
  dataset_manifest="${MANIFEST}" \
  output_filename="${RESULTS_DIR}/E00_tdt_greedy.json" \
  batch_size="${BATCH_SIZE}" \
  use_cer=true \
  decoder_type=rnnt \
  rnnt_decoding.strategy=greedy_batch
