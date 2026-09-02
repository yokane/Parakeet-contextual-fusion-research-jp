#!/usr/bin/env bash
set -euo pipefail

ROOT="${JPA_CF_IMAGE_HOME:-/opt/jpacf}"
cd "$ROOT"
STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
HF_BUCKET="${HF_BUCKET:-saeeew/J-PACF-YOMI-tdt-bucket}"
RESEARCH_KEY="${RESEARCH_KEY:?RESEARCH_KEY is required}"
TASK="${JPA_CF_RESEARCH_TASK:?JPA_CF_RESEARCH_TASK is required}"
EVAL_DIR="${STATE_ROOT}/generated/eval"
LM_DIR="${STATE_ROOT}/artifacts/lm"
RESULTS_DIR="${STATE_ROOT}/results"
export JPA_CF_STATE_ROOT="$STATE_ROOT" EVAL_DIR RESULTS_DIR

finish() {
  rc=$?
  echo "JPA_CF_RESEARCH_TASK rc=${rc} task=${TASK}"
  exit "$rc"
}
trap finish EXIT

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 2; }
mkdir -p "$STATE_ROOT" "$LM_DIR" "$RESULTS_DIR"

# Restore only the immutable delta snapshots required by this task. The current
# task's output is published under a new stage path and can never overwrite an
# existing workspace-cache key.
bash scripts/hf/hf-research-snapshot.sh pull "$RESEARCH_KEY" "$TASK" "$STATE_ROOT"

if [[ -f "$EVAL_DIR/nemo_eval.jsonl" && -d "$EVAL_DIR/audio" ]]; then
  uv run --locked --no-sync python scripts/research/rebase_eval_manifest.py \
    --manifest "$EVAL_DIR/nemo_eval.jsonl" \
    --audio-dir "$EVAL_DIR/audio"
fi

model_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))['repositories']['base_model']['revision'])
PY
)"

case "$TASK" in
  e02-encode)
    model_nemo="${STATE_ROOT}/scratch/model/parakeet-tdt_ctc-0.6b-ja.nemo"
    mkdir -p "$(dirname "$model_nemo")"
    test -s "$EVAL_DIR/lm_corpus.txt"
    uv run --locked --no-sync python scripts/materialize_locked_model.py --output "$model_nemo"
    uv run --locked --no-sync python scripts/research/ngram_lm_pipeline.py encode \
      --model-nemo "$model_nemo" \
      --model-revision "$model_revision" \
      --corpus "$EVAL_DIR/lm_corpus.txt" \
      --output "$LM_DIR/lm_corpus.encoded.txt" \
      --metadata "$LM_DIR/encoding-metadata.json"
    ;;
  e02-pack)
    uv run --locked --no-sync python scripts/research/ngram_lm_pipeline.py pack \
      --arpa "$LM_DIR/ja-6gram.arpa" \
      --encoding-metadata "$LM_DIR/encoding-metadata.json" \
      --estimation-metadata "$LM_DIR/estimation-metadata.json" \
      --model-revision "$model_revision" \
      --output "$LM_DIR/ja-6gram.nemo" \
      --metadata "$LM_DIR/package-metadata.json"
    ;;
  e05-extract)
    bash scripts/research/e05_extract_gpu.sh
    ;;
  E00|E01|E02|E03|E04|E06)
    uv run --locked --no-sync python scripts/research/check_phase_artifacts.py "$TASK" \
      --state-root "$STATE_ROOT"
    if [[ "$TASK" == "E06" ]]; then
      : "${E06_DRIVER:?E06_DRIVER is required for E06}"
      export E06_DRIVER
      export PHONE_HEAD="${PHONE_HEAD:-${STATE_ROOT}/artifacts/phone_head.pt}"
      export PHONE_VOCAB="${PHONE_VOCAB:-${STATE_ROOT}/artifacts/phone_vocab.json}"
      export ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-${STATE_ROOT}/artifacts/encoder}"
    fi
    bash scripts/container/run-phase.sh "$TASK"
    ;;
  *)
    echo "unsupported JPA_CF_RESEARCH_TASK: $TASK" >&2
    exit 2
    ;;
esac

bash scripts/hf/hf-research-snapshot.sh push "$RESEARCH_KEY" "$TASK" "$STATE_ROOT"

if [[ "$TASK" =~ ^E0[0-46]$ && -n "${JPA_CF_EVIDENCE_RUN_ID:-}" ]]; then
  run_dir="${STATE_ROOT}/dist/hf-runs/${JPA_CF_EVIDENCE_RUN_ID}"
  uv run --locked --no-sync python scripts/hf/build_run_bundle.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$run_dir" \
    --run-id "$JPA_CF_EVIDENCE_RUN_ID" \
    --workflow-kind "research-${TASK}" \
    --benchmark-index "$EVAL_DIR/bench_index.jsonl" \
    --execution-manifest "$EVAL_DIR/nemo_eval.jsonl"
  bash scripts/hf/hf-push-run.sh "$run_dir"
fi
