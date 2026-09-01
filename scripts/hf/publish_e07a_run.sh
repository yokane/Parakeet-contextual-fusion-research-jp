#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

RUN_ID="${1:-}"
[[ -n "${RUN_ID}" ]] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe run ID: ${RUN_ID}" >&2; exit 2; }

RESULTS_DIR="${RESULTS_DIR:-results}"
PREDICTIONS="${RESULTS_DIR}/E07a_shisa_select.jsonl"
METRICS="${RESULTS_DIR}/E07a_metrics.parquet"
SUMMARY="${RESULTS_DIR}/E07a_summary.json"
BENCHMARK_INDEX="${BENCHMARK_INDEX:-data/generated/bench_index.jsonl}"
EXECUTION_MANIFEST="${MANIFEST:-data/generated/nemo_eval.jsonl}"
STAGE="dist/e07a-results/${RUN_ID}"
BUNDLE="dist/hf-runs/${RUN_ID}"

for path in "${PREDICTIONS}" "${METRICS}" "${SUMMARY}" "${BENCHMARK_INDEX}"; do
  [[ -f "${path}" ]] || { echo "Required E07a evidence missing: ${path}" >&2; exit 2; }
done

rm -rf "${STAGE}" "${BUNDLE}"
mkdir -p "${STAGE}"
cp "${PREDICTIONS}" "${STAGE}/E07a_shisa_select.jsonl"
cp "${METRICS}" "${STAGE}/metrics.parquet"
cp "${SUMMARY}" "${STAGE}/summary.json"

bundle_args=(
  --results-dir "${STAGE}"
  --output-dir "${BUNDLE}"
  --run-id "${RUN_ID}"
  --workflow-kind "E07a-shisa-nbest-selector"
  --benchmark-index "${BENCHMARK_INDEX}"
)
if [[ -f "${EXECUTION_MANIFEST}" ]]; then
  bundle_args+=(--execution-manifest "${EXECUTION_MANIFEST}")
fi

python scripts/hf/build_run_bundle.py "${bundle_args[@]}"
bash scripts/hf/hf-push-run.sh "${BUNDLE}"

echo "E07a Bucket run: hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/runs/${RUN_ID}"
