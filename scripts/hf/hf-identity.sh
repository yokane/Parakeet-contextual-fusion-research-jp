#!/usr/bin/env bash

HF_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
HF_REPOSITORY_ROOT="$(cd -- "${HF_HELPER_DIR}/../.." >/dev/null 2>&1 && pwd)"
HF_TRANSPORT_PROJECT="${HF_TRANSPORT_PROJECT:-${HF_REPOSITORY_ROOT}/tools/hf-bucket}"

hf_bucket_cli() {
    command -v uv >/dev/null 2>&1 || {
        printf '[hf-cli] ERROR: uv is required\n' >&2
        return 127
    }
    # Bucket transport is intentionally isolated from the ASR environment.
    # In the GHCR runtime it is already materialized under /opt/jpacf and must
    # never be re-synchronized during a research run.
    if [[ "${JPA_CF_CONTAINER_RUNTIME:-0}" == "1" ]]; then
        env -u UV_PROJECT_ENVIRONMENT \
            uv run --project "${HF_TRANSPORT_PROJECT}" --locked --no-sync -- hf "$@"
    else
        env -u UV_PROJECT_ENVIRONMENT \
            uv run --project "${HF_TRANSPORT_PROJECT}" --locked -- hf "$@"
    fi
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
