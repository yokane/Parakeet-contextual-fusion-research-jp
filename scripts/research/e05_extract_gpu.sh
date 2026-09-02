#!/usr/bin/env bash
set -euo pipefail

ROOT="${JPA_CF_IMAGE_HOME:-/opt/jpacf}"
cd "$ROOT"
STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
MANIFEST="${MANIFEST:-${STATE_ROOT}/generated/eval/nemo_eval.jsonl}"
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-${STATE_ROOT}/artifacts/encoder}"

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
model_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))['repositories']['base_model']['revision'])
PY
)"

rm -rf "$ENCODER_FEATURE_DIR"
mkdir -p "$ENCODER_FEATURE_DIR"
uv run --locked --no-sync python scripts/extract_encoder_features.py \
  --manifest "$MANIFEST" \
  --output-dir "$ENCODER_FEATURE_DIR" \
  --model-revision "$model_revision"

count="$(find "$ENCODER_FEATURE_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
[[ "$count" -gt 0 ]] || { echo "no encoder features were produced" >&2; exit 2; }
echo "E05 encoder extraction complete: ${count} feature files"
