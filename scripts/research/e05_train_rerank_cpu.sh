#!/usr/bin/env bash
set -euo pipefail

ROOT="${JPA_CF_IMAGE_HOME:-/opt/jpacf}"
cd "$ROOT"
STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
BENCHMARK_INDEX="${BENCHMARK_INDEX:-${STATE_ROOT}/generated/eval/bench_index.jsonl}"
E04_INPUT="${E04_INPUT:-${STATE_ROOT}/results/E04_ctc_rerank.jsonl}"
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-${STATE_ROOT}/artifacts/encoder}"
TRAIN_FEATURE_DIR="${TRAIN_FEATURE_DIR:-${STATE_ROOT}/artifacts/encoder_train}"
PHONE_TRAIN_MANIFEST="${PHONE_TRAIN_MANIFEST:-${STATE_ROOT}/generated/phone_train.jsonl}"
PHONE_VOCAB="${PHONE_VOCAB:-${STATE_ROOT}/artifacts/phone_vocab.json}"
PHONE_HEAD="${PHONE_HEAD:-${STATE_ROOT}/artifacts/phone_head.pt}"
ANNOTATED_E04="${ANNOTATED_E04:-${STATE_ROOT}/results/E04_phone_ready.jsonl}"
E05_OUT="${E05_OUT:-${STATE_ROOT}/results/E05_phone_rerank.jsonl}"

for path in "$BENCHMARK_INDEX" "$E04_INPUT"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
[[ -d "$ENCODER_FEATURE_DIR" ]] || { echo "missing encoder feature directory: $ENCODER_FEATURE_DIR" >&2; exit 2; }

python scripts/prepare_phone_head_data.py \
  --benchmark "$BENCHMARK_INDEX" \
  --e04 "$E04_INPUT" \
  --feature-dir "$ENCODER_FEATURE_DIR" \
  --train-feature-dir "$TRAIN_FEATURE_DIR" \
  --train-manifest "$PHONE_TRAIN_MANIFEST" \
  --vocab "$PHONE_VOCAB" \
  --annotated-e04 "$ANNOTATED_E04"

phone_vocab_size="$(python - "$PHONE_VOCAB" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['phone_vocab_size'])
PY
)"

python scripts/train_phone_head.py \
  --manifest "$PHONE_TRAIN_MANIFEST" \
  --phone-vocab-size "$phone_vocab_size" \
  --output "$PHONE_HEAD" \
  --device cpu \
  --epochs "${PHONE_EPOCHS:-10}" \
  --lr "${PHONE_LR:-3e-4}"

python scripts/rerank_phone.py \
  --input "$ANNOTATED_E04" \
  --output "$E05_OUT" \
  --checkpoint "$PHONE_HEAD" \
  --feature-dir "$ENCODER_FEATURE_DIR" \
  --alpha "${PHONE_ALPHA:-0.20}" \
  --device cpu

echo "E05 CPU phone-head stage complete: $E05_OUT"
