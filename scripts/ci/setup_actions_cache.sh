#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"

root="${SELF_ACTIONS_CACHE_ROOT:-}"
workspace="${GITHUB_WORKSPACE}"

write_env() {
  printf '%s=%s\n' "$1" "$2" >> "${GITHUB_ENV}"
}

if [[ -n "${root}" ]]; then
  [[ "${root}" == /* ]] || {
    echo "SELF_ACTIONS_CACHE_ROOT must be an absolute path" >&2
    exit 2
  }
  mkdir -p "${root}"
  [[ -d "${root}" && -w "${root}" ]] || {
    echo "SELF_ACTIONS_CACHE_ROOT must be a writable directory" >&2
    exit 2
  }

  namespace="${root}/${GITHUB_REPOSITORY}"
  uv_cache="${namespace}/uv"
  hf_hub_cache="${namespace}/huggingface/hub"
  hf_xet_cache="${namespace}/huggingface/xet"
  xdg_cache="${namespace}/xdg"
  buildkit_cache="${namespace}/buildkit"
  mise_data="${namespace}/mise/data"
  mise_cache="${namespace}/mise/cache"

  mkdir -p \
    "${uv_cache}" \
    "${hf_hub_cache}" \
    "${hf_xet_cache}" \
    "${xdg_cache}" \
    "${buildkit_cache}" \
    "${mise_data}" \
    "${mise_cache}"

  for path in \
    "${uv_cache}" \
    "${hf_hub_cache}" \
    "${hf_xet_cache}" \
    "${xdg_cache}" \
    "${buildkit_cache}" \
    "${mise_data}" \
    "${mise_cache}"; do
    [[ -w "${path}" ]] || {
      echo "persistent cache namespace is not writable" >&2
      exit 2
    }
  done

  write_env SELF_CACHE_PERSISTENT true
  write_env UV_CACHE_DIR "${uv_cache}"
  write_env HF_HUB_CACHE "${hf_hub_cache}"
  write_env HF_XET_CACHE "${hf_xet_cache}"
  write_env XDG_CACHE_HOME "${xdg_cache}"
  write_env MISE_DATA_DIR "${mise_data}"
  write_env MISE_CACHE_DIR "${mise_cache}"
  write_env BUILDKIT_CACHE_DIR "${buildkit_cache}"
  write_env BUILDKIT_CACHE_FROM "type=local,src=${buildkit_cache}"
  write_env BUILDKIT_CACHE_TO "type=local,dest=${buildkit_cache}-next,mode=max"
  printf 'persistent=true\n' >> "${GITHUB_OUTPUT}"
else
  uv_cache="${workspace}/.cache/uv"
  hf_hub_cache="${workspace}/.cache/huggingface/hub"
  hf_xet_cache="${workspace}/.cache/huggingface/xet"
  xdg_cache="${workspace}/.cache/xdg"
  cache_scope="${ACTIONS_CACHE_SCOPE:-jpacf-${RUNNER_OS:-linux}-${RUNNER_ARCH:-x64}}"

  mkdir -p "${uv_cache}" "${hf_hub_cache}" "${hf_xet_cache}" "${xdg_cache}"

  write_env SELF_CACHE_PERSISTENT false
  write_env UV_CACHE_DIR "${uv_cache}"
  write_env HF_HUB_CACHE "${hf_hub_cache}"
  write_env HF_XET_CACHE "${hf_xet_cache}"
  write_env XDG_CACHE_HOME "${xdg_cache}"
  write_env BUILDKIT_CACHE_FROM "type=gha,scope=${cache_scope}"
  write_env BUILDKIT_CACHE_TO "type=gha,scope=${cache_scope},mode=max"
  printf 'persistent=false\n' >> "${GITHUB_OUTPUT}"
fi
