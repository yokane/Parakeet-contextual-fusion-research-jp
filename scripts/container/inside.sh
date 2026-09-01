#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${JPA_CF_STATE_ROOT:-/workspace/state}"
RESULTS_NAME="${JPA_CF_RESULTS_NAME:-staged}"
mkdir -p "$STATE_ROOT"/{hf,uv,xdg,torch,home,artifacts,generated,results,dist,vendor}

export JPA_CF_CONTAINER_RUNTIME=1
export JPA_CF_STATE_ROOT="$STATE_ROOT"
export HOME="${HOME:-$STATE_ROOT/home}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$STATE_ROOT/hf/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$STATE_ROOT/hf/xet}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STATE_ROOT/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STATE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$STATE_ROOT/torch}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/jpacf/.venv}"
# Do not rely on the inherited NGC/PATH ordering for Python. The image is
# built with a repository-owned, locked uv environment and Python entrypoints
# must execute that interpreter explicitly.
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r
VENV_PYTHON="$VIRTUAL_ENV/bin/python"
[[ -x "$VENV_PYTHON" ]] || {
  echo "repository-owned Python is missing or not executable: $VENV_PYTHON" >&2
  exit 2
}
export HF_TRANSPORT_PROJECT="${HF_TRANSPORT_PROJECT:-/opt/jpacf/tools/hf-bucket}"

export EVAL_DIR="${EVAL_DIR:-$STATE_ROOT/generated/eval}"
export RESULTS_DIR="${RESULTS_DIR:-$STATE_ROOT/results/$RESULTS_NAME}"
export MODEL_NEMO="${MODEL_NEMO:-$STATE_ROOT/artifacts/model/parakeet-tdt_ctc-0.6b-ja.nemo}"
export NGPU_LM="${NGPU_LM:-$STATE_ROOT/artifacts/lm/ja-6gram.nemo}"
export NEMO_ROOT="${NEMO_ROOT:-$STATE_ROOT/vendor/nemo-speech}"
export KENLM_ROOT="${KENLM_ROOT:-$STATE_ROOT/vendor/kenlm}"

if [[ -d /workspace/project/src ]]; then
  export PYTHONPATH="/workspace/project/src:/opt/jpacf/src${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="/opt/jpacf/src${PYTHONPATH:+:$PYTHONPATH}"
fi

case "${1:-}" in
  python|python3)
    shift
    exec "$VENV_PYTHON" "$@"
    ;;
esac

exec "$@"
