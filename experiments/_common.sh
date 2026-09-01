#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

read_hf_lock() {
  local field="$1"
  uv run --locked --no-sync python - "$field" <<'PY'
import json
import sys
from pathlib import Path
entry = json.loads(Path("locks/hf-revisions.lock.json").read_text(encoding="utf-8"))["repositories"]["base_model"]
print(entry[sys.argv[1]])
PY
}

export MODEL_NAME="${MODEL_NAME:-$(read_hf_lock repo_id)}"
LOCKED_MODEL_REVISION="$(read_hf_lock revision)"
export MODEL_REVISION="${MODEL_REVISION:-${LOCKED_MODEL_REVISION}}"
if [[ "${MODEL_REVISION}" != "${LOCKED_MODEL_REVISION}" ]]; then
  echo "MODEL_REVISION must match locks/hf-revisions.lock.json" >&2
  exit 2
fi

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

decode_tdt() {
  uv run --locked --no-sync python scripts/decode_nbest.py \
    --model-revision "${MODEL_REVISION}" \
    "$@"
}
