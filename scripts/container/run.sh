#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
IMAGE="${JPA_CF_IMAGE:-}"
[[ -n "$IMAGE" ]] || { echo "JPA_CF_IMAGE is required" >&2; exit 2; }
if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ && "${JPA_CF_ALLOW_MUTABLE_IMAGE:-0}" != "1" ]]; then
  echo "JPA_CF_IMAGE must be digest-pinned (or set JPA_CF_ALLOW_MUTABLE_IMAGE=1 for local-only development)" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
STATE_ROOT="${JPA_CF_STATE_ROOT:-${ROOT}/.jpacf-state}"
mkdir -p "$STATE_ROOT"/{hf,uv,xdg,torch,home,artifacts,generated,results,dist,vendor}
args=(
  run --rm --gpus all --ipc=host --shm-size "${JPA_CF_SHM_SIZE:-8g}"
  -w /workspace/project
  -e HOME=/cache/home
  -e JPA_CF_CONTAINER_RUNTIME=1
  -e JPA_CF_WORKSPACE=/workspace/project
  -e PYTHONPATH=/workspace/project/src:/opt/jpacf/src
  -e UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv
  -e HF_TRANSPORT_PROJECT=/opt/jpacf/tools/hf-bucket
  -e HF_HOME=/cache/huggingface
  -e HF_HUB_CACHE=/cache/huggingface/hub
  -e HF_XET_CACHE=/cache/huggingface/xet
  -e UV_CACHE_DIR=/cache/uv
  -e XDG_CACHE_HOME=/cache/xdg
  -e TORCH_HOME=/cache/torch
  -v "$ROOT:/workspace/project"
  -v "$STATE_ROOT/hf:/cache/huggingface"
  -v "$STATE_ROOT/uv:/cache/uv"
  -v "$STATE_ROOT/xdg:/cache/xdg"
  -v "$STATE_ROOT/torch:/cache/torch"
  -v "$STATE_ROOT/home:/cache/home"
  -v "$STATE_ROOT/artifacts:/workspace/project/artifacts"
  -v "$STATE_ROOT/generated:/workspace/project/data/generated"
  -v "$STATE_ROOT/results:/workspace/project/results"
  -v "$STATE_ROOT/dist:/workspace/project/dist"
  -v "$STATE_ROOT/vendor:/workspace/project/.vendor"
)
for name in \
  HF_TOKEN HF_BUCKET CUDA_VISIBLE_DEVICES \
  HF_CONFIG HF_SPLIT EVAL_DIR RESULTS_DIR RUN_E05 RUN_E06 E06_DRIVER \
  MANIFEST CONTEXT_PHRASES NGPU_LM MODEL_NEMO NEMO_ROOT KENLM_ROOT \
  BEAM_SIZE LM_ALPHA PB_ALPHA CTC_ALPHA PHONE_ALPHA BATCH_SIZE; do
  [[ -n "${!name:-}" ]] && args+=( -e "$name" )
done
exec docker "${args[@]}" "$IMAGE" "$@"
