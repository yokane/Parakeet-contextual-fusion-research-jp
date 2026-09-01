#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

INPUT="${INPUT:-${RESULTS_DIR}/E05_phone_rerank.jsonl}"
OUT="${RESULTS_DIR}/E07a_shisa_select.jsonl"
BENCHMARK_INDEX="${BENCHMARK_INDEX:-data/generated/bench_index.jsonl}"
SHISA_MODEL="${SHISA_MODEL:-shisa-ai/shisa-v2-qwen2.5-7b}"
SHISA_REVISION="${SHISA_REVISION:-2ba1a59}"
SHISA_TOP_K="${SHISA_TOP_K:-8}"
SHISA_SEED="${SHISA_SEED:-7}"
SHISA_DTYPE="${SHISA_DTYPE:-bfloat16}"
SHISA_CONTEXT_FIELD="${SHISA_CONTEXT_FIELD:-}"

require_file "${INPUT}"
require_file "${BENCHMARK_INDEX}"

args=(
  --input "${INPUT}"
  --output "${OUT}"
  --model "${SHISA_MODEL}"
  --revision "${SHISA_REVISION}"
  --top-k "${SHISA_TOP_K}"
  --seed "${SHISA_SEED}"
  --candidate-order stable_shuffle
  --dtype "${SHISA_DTYPE}"
)
if [[ -n "${SHISA_CONTEXT_FIELD}" ]]; then
  args+=(--context-field "${SHISA_CONTEXT_FIELD}")
fi

python scripts/shisa_nbest_select.py "${args[@]}"
python scripts/evaluate_shisa_selector.py \
  --benchmark "${BENCHMARK_INDEX}" \
  --input "${OUT}" \
  --parquet "${RESULTS_DIR}/E07a_metrics.parquet" \
  --summary "${RESULTS_DIR}/E07a_summary.json"

echo "E07a predictions: ${OUT}"
echo "E07a metrics: ${RESULTS_DIR}/E07a_metrics.parquet"
echo "E07a summary: ${RESULTS_DIR}/E07a_summary.json"
