#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export MODEL_NAME="${MODEL_NAME:-nvidia/parakeet-tdt_ctc-0.6b-ja}"
export NEMO_ROOT="${NEMO_ROOT:-/opt/NeMo}"
export MANIFEST="${MANIFEST:-data/generated/nemo_eval.jsonl}"
export CONTEXT_PHRASES="${CONTEXT_PHRASES:-data/generated/context_phrases.txt}"
export NGPU_LM="${NGPU_LM:-artifacts/lm/ja-6gram.nemo}"
export RESULTS_DIR="${RESULTS_DIR:-results}"
export BEAM_SIZE="${BEAM_SIZE:-8}"
export LM_ALPHA="${LM_ALPHA:-0.20}"
export PB_ALPHA="${PB_ALPHA:-1.0}"
export CTC_ALPHA="${CTC_ALPHA:-0.20}"
export PHONE_ALPHA="${PHONE_ALPHA:-0.20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
mkdir -p "${RESULTS_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 2
  fi
}

nemo_eval() {
  local eval_script="${NEMO_ROOT}/examples/asr/speech_to_text_eval.py"
  require_file "${eval_script}"
  python "${eval_script}" "$@"
}
