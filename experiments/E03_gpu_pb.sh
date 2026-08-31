#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
require_file "${MANIFEST}"
require_file "${NGPU_LM}"
require_file "${CONTEXT_PHRASES}"

nemo_eval \
  pretrained_name="${MODEL_NAME}" \
  dataset_manifest="${MANIFEST}" \
  output_filename="${RESULTS_DIR}/E03_gpu_pb.json" \
  batch_size="${BATCH_SIZE}" \
  use_cer=true \
  decoder_type=rnnt \
  rnnt_decoding.strategy=malsd_batch \
  rnnt_decoding.beam.beam_size="${BEAM_SIZE}" \
  rnnt_decoding.beam.pruning_mode=late \
  rnnt_decoding.beam.blank_lm_score_mode=lm_weighted_full \
  rnnt_decoding.beam.allow_cuda_graphs=true \
  rnnt_decoding.beam.ngram_lm_model="${NGPU_LM}" \
  rnnt_decoding.beam.ngram_lm_alpha="${LM_ALPHA}" \
  rnnt_decoding.beam.boosting_tree.key_phrases_file="${CONTEXT_PHRASES}" \
  rnnt_decoding.beam.boosting_tree.context_score=1.0 \
  rnnt_decoding.beam.boosting_tree.depth_scaling=2.0 \
  rnnt_decoding.beam.boosting_tree_alpha="${PB_ALPHA}"
