#!/usr/bin/env bash

HF_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
HF_REPOSITORY_ROOT="$(cd -- "${HF_HELPER_DIR}/../.." >/dev/null 2>&1 && pwd)"
HF_TRANSPORT_PROJECT="${HF_TRANSPORT_PROJECT:-${HF_REPOSITORY_ROOT}/tools/hf-bucket}"

hf_bucket_cli() {
    command -v uv >/dev/null 2>&1 || {
        printf '[hf-cli] ERROR: uv is required; enter through mise\n' >&2
        return 127
    }
    # Root mise.toml intentionally pins UV_PROJECT_ENVIRONMENT=.venv for the
    # ASR environment. Unset it here so the Bucket transport lock materializes
    # and runs in tools/hf-bucket/.venv instead of mutating the NeMo environment.
    env -u UV_PROJECT_ENVIRONMENT uv run --project "${HF_TRANSPORT_PROJECT}" --locked -- hf "$@"
}

hf_normalize_bucket_id() {
    local value="$1" namespace name
    value="${value#hf://buckets/}"
    value="${value%/}"
    [[ "$value" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || return 1
    namespace="${value%%/*}"
    name="${value#*/}"
    [[ "$namespace" != "." && "$namespace" != ".." && "$name" != "." && "$name" != ".." ]] || return 1
    printf '%s\n' "$value"
}

hf_normalize_model_repo_id() {
    local value="$1" namespace name
    value="${value#hf://models/}"
    value="${value#hf://}"
    value="${value%/}"
    [[ "$value" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || return 1
    namespace="${value%%/*}"
    name="${value#*/}"
    [[ "$namespace" != "." && "$namespace" != ".." && "$name" != "." && "$name" != ".." ]] || return 1
    printf '%s\n' "$value"
}
