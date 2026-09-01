#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

EVAL_DIR="${EVAL_DIR:-data/generated/eval}"
export MANIFEST="${MANIFEST:-${EVAL_DIR}/nemo_eval.jsonl}"
export CONTEXT_PHRASES="${CONTEXT_PHRASES:-${EVAL_DIR}/context_phrases.txt}"
export NGPU_LM="${NGPU_LM:-artifacts/lm/ja-6gram.nemo}"
export RESULTS_DIR="${RESULTS_DIR:-results/staged}"
BENCHMARK_INDEX="${BENCHMARK_INDEX:-${EVAL_DIR}/bench_index.jsonl}"
RUN_E05="${RUN_E05:-auto}"
RUN_E06="${RUN_E06:-0}"
PHONE_HEAD="${PHONE_HEAD:-artifacts/phone_head.pt}"
PHONE_VOCAB="${PHONE_VOCAB:-artifacts/phone_vocab.json}"
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR:-artifacts/encoder}"
PHONE_TRAIN_MANIFEST="${PHONE_TRAIN_MANIFEST:-data/generated/phone_train.jsonl}"

mkdir -p "${RESULTS_DIR}"
for path in "${MANIFEST}" "${CONTEXT_PHRASES}" "${NGPU_LM}" "${BENCHMARK_INDEX}"; do
  test -f "${path}" || { echo "Missing prerequisite: ${path}" >&2; exit 2; }
done

model_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))['repositories']['base_model']['revision'])
PY
)"

bash experiments/E00_tdt_greedy.sh
bash experiments/E01_tdt_beam.sh
bash experiments/E02_ngpulm.sh
bash experiments/E03_gpu_pb.sh
bash experiments/E04_ctc_rerank.sh

uv run --locked --no-sync python scripts/collect_experiment_metrics.py \
  --benchmark "${BENCHMARK_INDEX}" \
  --execution-manifest "${MANIFEST}" \
  --result "E00=${RESULTS_DIR}/E00_tdt_greedy.jsonl" \
  --result "E01=${RESULTS_DIR}/E01_tdt_beam.jsonl" \
  --result "E02=${RESULTS_DIR}/E02_ngpulm.jsonl" \
  --result "E03=${RESULTS_DIR}/E03_gpu_pb.jsonl" \
  --result "E04=${RESULTS_DIR}/E04_ctc_rerank.jsonl" \
  --parquet "${RESULTS_DIR}/metrics_e00_e04.parquet" \
  --summary "${RESULTS_DIR}/summary_e00_e04.json"

gate_status=0
uv run --locked --no-sync python scripts/evaluate_e05_gate.py \
  --metrics "${RESULTS_DIR}/metrics_e00_e04.parquet" \
  --output "${RESULTS_DIR}/e05_gate.json" || gate_status=$?

if [[ "${RUN_E05}" == "0" || "${RUN_E05}" == "false" ]]; then
  echo "E05 disabled by RUN_E05=${RUN_E05}."
  exit 0
fi
if [[ "${RUN_E05}" == "auto" && "${gate_status}" -ne 0 ]]; then
  echo "E04 evidence gate did not justify E05; stopping after E04. See ${RESULTS_DIR}/e05_gate.json"
  exit 0
fi
if [[ "${gate_status}" -ne 0 && "${RUN_E05}" != "force" && "${RUN_E05}" != "1" && "${RUN_E05}" != "true" ]]; then
  echo "E05 gate failed. Set RUN_E05=force only for an explicitly documented ablation." >&2
  exit 3
fi

uv run --locked --no-sync python scripts/extract_encoder_features.py \
  --manifest "${MANIFEST}" \
  --output-dir "${ENCODER_FEATURE_DIR}" \
  --model-revision "${model_revision}"

uv run --locked --no-sync python scripts/prepare_phone_head_data.py \
  --benchmark "${BENCHMARK_INDEX}" \
  --e04 "${RESULTS_DIR}/E04_ctc_rerank.jsonl" \
  --feature-dir "${ENCODER_FEATURE_DIR}" \
  --train-manifest "${PHONE_TRAIN_MANIFEST}" \
  --vocab "${PHONE_VOCAB}" \
  --annotated-e04 "${RESULTS_DIR}/E04_phone_ready.jsonl"

phone_vocab_size="$(uv run --locked --no-sync python - "${PHONE_VOCAB}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['phone_vocab_size'])
PY
)"
uv run --locked --no-sync python scripts/train_phone_head.py \
  --manifest "${PHONE_TRAIN_MANIFEST}" \
  --phone-vocab-size "${phone_vocab_size}" \
  --output "${PHONE_HEAD}"

INPUT="${RESULTS_DIR}/E04_phone_ready.jsonl" \
PHONE_HEAD="${PHONE_HEAD}" \
ENCODER_FEATURE_DIR="${ENCODER_FEATURE_DIR}" \
bash experiments/E05_phone_rerank.sh

uv run --locked --no-sync python scripts/collect_experiment_metrics.py \
  --benchmark "${BENCHMARK_INDEX}" \
  --execution-manifest "${MANIFEST}" \
  --result "E00=${RESULTS_DIR}/E00_tdt_greedy.jsonl" \
  --result "E01=${RESULTS_DIR}/E01_tdt_beam.jsonl" \
  --result "E02=${RESULTS_DIR}/E02_ngpulm.jsonl" \
  --result "E03=${RESULTS_DIR}/E03_gpu_pb.jsonl" \
  --result "E04=${RESULTS_DIR}/E04_ctc_rerank.jsonl" \
  --result "E05=${RESULTS_DIR}/E05_phone_rerank.jsonl" \
  --parquet "${RESULTS_DIR}/metrics_e00_e05.parquet" \
  --summary "${RESULTS_DIR}/summary_e00_e05.json"

if [[ "${RUN_E06}" == "1" || "${RUN_E06}" == "true" ]]; then
  : "${E06_DRIVER:?RUN_E06 requires E06_DRIVER to a NeMo-3.0.0-specific in-beam driver}"
  bash experiments/E06_inbeam.sh
  uv run --locked --no-sync python scripts/collect_experiment_metrics.py \
    --benchmark "${BENCHMARK_INDEX}" \
    --execution-manifest "${MANIFEST}" \
    --result "E00=${RESULTS_DIR}/E00_tdt_greedy.jsonl" \
    --result "E01=${RESULTS_DIR}/E01_tdt_beam.jsonl" \
    --result "E02=${RESULTS_DIR}/E02_ngpulm.jsonl" \
    --result "E03=${RESULTS_DIR}/E03_gpu_pb.jsonl" \
    --result "E04=${RESULTS_DIR}/E04_ctc_rerank.jsonl" \
    --result "E05=${RESULTS_DIR}/E05_phone_rerank.jsonl" \
    --result "E06=${RESULTS_DIR}/E06_inbeam.jsonl" \
    --parquet "${RESULTS_DIR}/metrics_e00_e06.parquet" \
    --summary "${RESULTS_DIR}/summary_e00_e06.json"
fi
