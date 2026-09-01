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
mkdir -p "$STATE_ROOT"
args=(
  run --rm --gpus all --ipc=host --shm-size "${JPA_CF_SHM_SIZE:-8g}"
  -w /workspace/project
  -e JPA_CF_STATE_ROOT=/workspace/state
  -e JPA_CF_WORKSPACE=/workspace/project
  -v "$ROOT:/workspace/project"
  -v "$STATE_ROOT:/workspace/state"
)
if [[ "${JPA_CF_RUN_AS_ROOT:-0}" != "1" ]]; then
  args+=( --user "$(id -u):$(id -g)" )
fi
for name in \
  HF_TOKEN HF_BUCKET CUDA_VISIBLE_DEVICES JPA_CF_RESULTS_NAME JPA_CF_EVIDENCE_RUN_ID \
  GITHUB_SHA GITHUB_RUN_ID GITHUB_RUN_ATTEMPT \
  HF_CONFIG HF_SPLIT EVAL_DIR RESULTS_DIR RUN_E05 RUN_E06 E06_DRIVER \
  MANIFEST CONTEXT_PHRASES NGPU_LM MODEL_NEMO NEMO_ROOT KENLM_ROOT \
  BEAM_SIZE LM_ALPHA PB_ALPHA CTC_ALPHA PHONE_ALPHA BATCH_SIZE; do
  [[ -n "${!name:-}" ]] && args+=( -e "$name" )
done

# The image owns scripts/container/inside.sh as its ENTRYPOINT. Passing the
# wrapper again here would nest the startup contract and can bypass the locked
# project interpreter. Forward only the requested command/CMD arguments.
exec docker "${args[@]}" "$IMAGE" "$@"
