#!/usr/bin/env bash
set -euo pipefail

: "${NEMO_ROOT:?Set NEMO_ROOT to a NeMo Speech checkout}"
: "${MODEL_NEMO:?Set MODEL_NEMO to a local .nemo file for the Japanese Parakeet model}"

LM_CORPUS="${LM_CORPUS:-data/generated/lm_corpus.txt}"
KENLM_BIN_DIR="${KENLM_BIN_DIR:-/opt/kenlm/build/bin}"
OUT_DIR="${OUT_DIR:-artifacts/lm}"
NGRAM_ORDER="${NGRAM_ORDER:-6}"

mkdir -p "${OUT_DIR}"
TRAIN_SCRIPT="${NEMO_ROOT}/scripts/asr_language_modeling/ngram_lm/train_kenlm.py"
MODEL_BASE="${OUT_DIR}/ja-${NGRAM_ORDER}gram"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "NeMo train_kenlm.py was not found at ${TRAIN_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${LM_CORPUS}" ]]; then
  echo "LM corpus was not found at ${LM_CORPUS}" >&2
  exit 2
fi
if [[ ! -x "${KENLM_BIN_DIR}/lmplz" || ! -x "${KENLM_BIN_DIR}/build_binary" ]]; then
  echo "KenLM lmplz/build_binary were not found under ${KENLM_BIN_DIR}" >&2
  exit 2
fi

uv run --locked --no-sync python "${TRAIN_SCRIPT}" \
  nemo_model_file="${MODEL_NEMO}" \
  train_paths="[${LM_CORPUS}]" \
  kenlm_bin_path="${KENLM_BIN_DIR}" \
  kenlm_model_file="${MODEL_BASE}" \
  ngram_length="${NGRAM_ORDER}" \
  preserve_arpa=true \
  save_nemo=true

test -f "${MODEL_BASE}.nemo" || { echo "NGPU-LM .nemo artifact was not produced" >&2; exit 2; }
echo "KenLM binary: ${MODEL_BASE}"
echo "NGPU-LM: ${MODEL_BASE}.nemo"
