#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
EVAL_DIR="${EVAL_DIR:-${STATE_ROOT}/generated/eval}"
HF_CONFIG="${HF_CONFIG:-homophone8-audio}"
HF_SPLIT="${HF_SPLIT:-test}"

mise install --locked
mise run deps:sync

benchmark_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
payload=json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))
print(payload['repositories']['benchmark']['revision'])
PY
)"
[[ "$benchmark_revision" =~ ^[0-9a-f]{40}$ ]]

rm -rf "$EVAL_DIR"
mkdir -p "$EVAL_DIR"
uv run --locked --no-sync python scripts/materialize_hf_eval.py \
  --repo-id saeeew/JP-HomophoneBench \
  --revision "$benchmark_revision" \
  --config "$HF_CONFIG" \
  --split "$HF_SPLIT" \
  --output-dir "$EVAL_DIR" \
  --rehydrate-audio \
  --require-audio
uv run --locked --no-sync python scripts/validate_eval_manifest.py "$EVAL_DIR/nemo_eval.jsonl" --require-audio
uv run --locked --no-sync python scripts/validate_audio_coverage.py \
  --provenance "$EVAL_DIR/eval_provenance.json" \
  --required-category exact_homophone \
  --required-category near_homophone \
  --min-per-category "${MIN_AUDIO_PER_CATEGORY:-5}" \
  --min-total "${MIN_AUDIO_TOTAL:-10}" \
  --output "$EVAL_DIR/audio_coverage.json"

printf 'Common E00-E06 artifacts are ready in %s\n' "$EVAL_DIR"
printf 'benchmark_revision=%s\n' "$benchmark_revision"
printf 'manifest=%s\n' "$EVAL_DIR/nemo_eval.jsonl"
printf 'context_phrases=%s\n' "$EVAL_DIR/context_phrases.txt"
printf 'lm_corpus=%s\n' "$EVAL_DIR/lm_corpus.txt"
