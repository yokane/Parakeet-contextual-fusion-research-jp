#!/usr/bin/env bash
set -euo pipefail

ROOT="${JPA_CF_IMAGE_HOME:-/opt/jpacf}"
cd "$ROOT"
STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
HF_BUCKET="${HF_BUCKET:-saeeew/J-PACF-YOMI-tdt-bucket}"
RESEARCH_KEY="${RESEARCH_KEY:?RESEARCH_KEY is required}"
TASK="${JPA_CF_RESEARCH_TASK:?JPA_CF_RESEARCH_TASK is required}"
REMOTE="hf://buckets/${HF_BUCKET}/workspace-cache/e00-e06/${RESEARCH_KEY}"
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
source scripts/hf/hf-identity.sh

# Reusable research state is mutable by design; immutable experiment evidence is
# published separately under runs/ after successful phase execution.
hf_bucket_cli buckets sync --token "$HF_TOKEN" "$REMOTE" "$STATE_ROOT"

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
    model_nemo="${STATE_ROOT}/artifacts/model/parakeet-tdt_ctc-0.6b-ja.nemo"
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
  E00|E01|E02|E03|E04|E05|E06)
    uv run --locked --no-sync python scripts/research/check_phase_artifacts.py "$TASK" \
      --state-root "$STATE_ROOT"
    if [[ "$TASK" == "E06" ]]; then
      : "${E06_DRIVER:?E06_DRIVER is required for E06}"
      export E06_DRIVER
    fi
    bash scripts/container/run-phase.sh "$TASK"
    ;;
  *)
    echo "unsupported JPA_CF_RESEARCH_TASK: $TASK" >&2
    exit 2
    ;;
esac

plan="$(mktemp -t jpacf-research-sync.XXXXXX.jsonl)"
hf_bucket_cli buckets sync --token "$HF_TOKEN" "$STATE_ROOT" "$REMOTE" --plan "$plan"
hf_bucket_cli buckets sync --token "$HF_TOKEN" --apply "$plan"
rm -f "$plan"

if [[ "$TASK" =~ ^E0[0-6]$ && -n "${JPA_CF_EVIDENCE_RUN_ID:-}" ]]; then
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
