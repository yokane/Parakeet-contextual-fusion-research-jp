#!/usr/bin/env bash
set -euo pipefail

ROOT="${JPA_CF_IMAGE_HOME:-/opt/jpacf}"
cd "$ROOT"

phase="${1:-${JPA_CF_PHASE:-}}"
phase="${phase^^}"

case "$phase" in
  E00|E01|E02|E03|E04|E05|E06) ;;
  *)
    echo "usage: $0 <E00|E01|E02|E03|E04|E05|E06>" >&2
    exit 2
    ;;
esac

export EVAL_DIR="${EVAL_DIR:-${JPA_CF_STATE_ROOT:-/workspace/state}/generated/eval}"
export MANIFEST="${MANIFEST:-${EVAL_DIR}/nemo_eval.jsonl}"
export CONTEXT_PHRASES="${CONTEXT_PHRASES:-${EVAL_DIR}/context_phrases.txt}"
export NGPU_LM="${NGPU_LM:-${JPA_CF_STATE_ROOT:-/workspace/state}/artifacts/lm/ja-6gram.nemo}"
export RESULTS_DIR="${RESULTS_DIR:-${JPA_CF_STATE_ROOT:-/workspace/state}/results/${JPA_CF_RESULTS_NAME:-staged}}"
BENCHMARK_INDEX="${BENCHMARK_INDEX:-${EVAL_DIR}/bench_index.jsonl}"
PHONE_HEAD="${PHONE_HEAD:-${JPA_CF_STATE_ROOT:-/workspace/state}/artifacts/phone_head.pt}"
PHONE_VOCAB="${PHONE_VOCAB:-${JPA_CF_STATE_ROOT:-/workspace/state}/artifacts/phone_vocab.json}"
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-${JPA_CF_STATE_ROOT:-/workspace/state}/artifacts/encoder}"
PHONE_TRAIN_MANIFEST="${PHONE_TRAIN_MANIFEST:-${JPA_CF_STATE_ROOT:-/workspace/state}/generated/phone_train.jsonl}"

mkdir -p "$RESULTS_DIR"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "Required file not found: $path" >&2
    exit 2
  }
}

run_e05_full_phase() {
  local e04_input="${INPUT:-${RESULTS_DIR}/E04_ctc_rerank.jsonl}"
  local annotated_e04="${RESULTS_DIR}/E04_phone_ready.jsonl"
  local model_revision phone_vocab_size

  require_file "$MANIFEST"
  require_file "$BENCHMARK_INDEX"
  require_file "$e04_input"

  model_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))['repositories']['base_model']['revision'])
PY
)"

  if [[ "${E05_PREPARE:-1}" == "1" || "${E05_PREPARE:-1}" == "true" ]]; then
    uv run --locked --no-sync python scripts/extract_encoder_features.py \
      --manifest "$MANIFEST" \
      --output-dir "$ENCODER_FEATURE_DIR" \
      --model-revision "$model_revision"

    uv run --locked --no-sync python scripts/prepare_phone_head_data.py \
      --benchmark "$BENCHMARK_INDEX" \
      --e04 "$e04_input" \
      --feature-dir "$ENCODER_FEATURE_DIR" \
      --train-manifest "$PHONE_TRAIN_MANIFEST" \
      --vocab "$PHONE_VOCAB" \
      --annotated-e04 "$annotated_e04"

    phone_vocab_size="$(uv run --locked --no-sync python - "$PHONE_VOCAB" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['phone_vocab_size'])
PY
)"

    uv run --locked --no-sync python scripts/train_phone_head.py \
      --manifest "$PHONE_TRAIN_MANIFEST" \
      --phone-vocab-size "$phone_vocab_size" \
      --output "$PHONE_HEAD"
  else
    annotated_e04="${INPUT:-${RESULTS_DIR}/E04_phone_ready.jsonl}"
  fi

  INPUT="$annotated_e04" \
  PHONE_HEAD="$PHONE_HEAD" \
  ENCODER_FEATURE_DIR="$ENCODER_FEATURE_DIR" \
    bash experiments/E05_phone_rerank.sh
}

case "$phase" in
  E00)
    require_file "$MANIFEST"
    exec bash experiments/E00_tdt_greedy.sh
    ;;
  E01)
    require_file "$MANIFEST"
    exec bash experiments/E01_tdt_beam.sh
    ;;
  E02)
    require_file "$MANIFEST"
    require_file "$NGPU_LM"
    exec bash experiments/E02_ngpulm.sh
    ;;
  E03)
    require_file "$MANIFEST"
    require_file "$NGPU_LM"
    require_file "$CONTEXT_PHRASES"
    exec bash experiments/E03_gpu_pb.sh
    ;;
  E04)
    require_file "$MANIFEST"
    require_file "$NGPU_LM"
    require_file "$CONTEXT_PHRASES"
    exec bash experiments/E04_ctc_rerank.sh
    ;;
  E05)
    run_e05_full_phase
    ;;
  E06)
    require_file "$MANIFEST"
    require_file "$NGPU_LM"
    require_file "$CONTEXT_PHRASES"
    : "${E06_DRIVER:?E06 requires E06_DRIVER to a NeMo-3.0.0-specific in-beam fusion driver}"
    exec bash experiments/E06_inbeam.sh
    ;;
esac
