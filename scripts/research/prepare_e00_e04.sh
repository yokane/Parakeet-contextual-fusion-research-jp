#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

HF_CONFIG="${HF_CONFIG:-homophone8-audio}"
HF_SPLIT="${HF_SPLIT:-test}"
EVAL_DIR="${EVAL_DIR:-data/generated/eval}"
MODEL_NEMO="${MODEL_NEMO:-artifacts/model/parakeet-tdt_ctc-0.6b-ja.nemo}"
NEMO_ROOT="${NEMO_ROOT:-.vendor/nemo-speech}"
KENLM_ROOT="${KENLM_ROOT:-.vendor/kenlm}"
NEMO_SPEECH_REVISION="${NEMO_SPEECH_REVISION:-fd6a877539710e2b98f28c43272ff81312f83417}"
KENLM_REVISION="${KENLM_REVISION:-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5}"
NGPU_LM="${NGPU_LM:-artifacts/lm/ja-6gram.nemo}"
JOBS="${JOBS:-$(nproc)}"

for command in git cmake c++ make; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "Missing build prerequisite: ${command}" >&2
    exit 2
  }
done

if [[ "${JPA_CF_CONTAINER_RUNTIME:-0}" == "1" ]]; then
  python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
else
  mise install --locked
  mise run deps:sync-gpu
  mise run hf:transport:sync
  uv run --locked --no-sync python scripts/repro/verify_platform.py --require-gpu
fi

benchmark_revision="$(uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('locks/hf-revisions.lock.json').read_text(encoding='utf-8'))
print(payload['repositories']['benchmark']['revision'])
PY
)"
[[ "${benchmark_revision}" =~ ^[0-9a-f]{40}$ ]] || exit 2

rm -rf "${EVAL_DIR}"
uv run --locked --no-sync python scripts/materialize_hf_eval.py \
  --repo-id saeeew/JP-HomophoneBench \
  --revision "${benchmark_revision}" \
  --config "${HF_CONFIG}" \
  --split "${HF_SPLIT}" \
  --output-dir "${EVAL_DIR}" \
  --rehydrate-audio \
  --require-audio
uv run --locked --no-sync python scripts/validate_eval_manifest.py "${EVAL_DIR}/nemo_eval.jsonl" --require-audio
uv run --locked --no-sync python scripts/validate_audio_coverage.py \
  --provenance "${EVAL_DIR}/eval_provenance.json" \
  --required-category exact_homophone \
  --required-category near_homophone \
  --min-per-category 5 \
  --min-total 10 \
  --output "${EVAL_DIR}/audio_coverage.json"

uv run --locked --no-sync python scripts/materialize_locked_model.py --output "${MODEL_NEMO}"

checkout_exact() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    rm -rf "${destination}"
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth=1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  actual="$(git -C "${destination}" rev-parse HEAD)"
  [[ "${actual}" == "${revision}" ]] || {
    echo "Revision mismatch for ${destination}: ${actual} != ${revision}" >&2
    exit 2
  }
}

checkout_exact https://github.com/NVIDIA-NeMo/Speech.git "${NEMO_ROOT}" "${NEMO_SPEECH_REVISION}"
checkout_exact https://github.com/kpu/kenlm.git "${KENLM_ROOT}" "${KENLM_REVISION}"

if [[ ! -x "${KENLM_ROOT}/build/bin/lmplz" || ! -x "${KENLM_ROOT}/build/bin/build_binary" ]]; then
  cmake -S "${KENLM_ROOT}" -B "${KENLM_ROOT}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCOMPILE_TESTS=OFF \
    -DENABLE_PYTHON=OFF
  cmake --build "${KENLM_ROOT}/build" --parallel "${JOBS}"
fi

if [[ ! -f "${NGPU_LM}" ]]; then
  NEMO_ROOT="${NEMO_ROOT}" \
  MODEL_NEMO="${MODEL_NEMO}" \
  LM_CORPUS="${EVAL_DIR}/lm_corpus.txt" \
  KENLM_BIN_DIR="${KENLM_ROOT}/build/bin" \
  OUT_DIR="$(dirname "${NGPU_LM}")" \
  bash scripts/train_ngpulm.sh
fi

test -f "${NGPU_LM}" || { echo "NGPU-LM was not created at ${NGPU_LM}" >&2; exit 2; }

cat <<EOF
E00-E04 preparation complete.
MANIFEST=${EVAL_DIR}/nemo_eval.jsonl
BENCHMARK_INDEX=${EVAL_DIR}/bench_index.jsonl
CONTEXT_PHRASES=${EVAL_DIR}/context_phrases.txt
LM_CORPUS=${EVAL_DIR}/lm_corpus.txt
NGPU_LM=${NGPU_LM}
MODEL_NEMO=${MODEL_NEMO}
EOF
