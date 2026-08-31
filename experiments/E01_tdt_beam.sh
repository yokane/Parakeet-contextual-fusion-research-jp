#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"

nemo_eval \
  pretrained_name="${MODEL_NAME}" \
  dataset_manifest="${MANIFEST}" \
  output_filename="${RESULTS_DIR}/E01_tdt_beam.json" \
  batch_size="${BATCH_SIZE}" \
  use_cer=true \
  decoder_type=rnnt \
  rnnt_decoding.strategy=malsd_batch \
  rnnt_decoding.beam.beam_size="${BEAM_SIZE}" \
  rnnt_decoding.beam.pruning_mode=late \
  rnnt_decoding.beam.blank_lm_score_mode=lm_weighted_full \
  rnnt_decoding.beam.allow_cuda_graphs=true
