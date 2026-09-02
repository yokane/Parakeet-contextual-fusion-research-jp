#!/usr/bin/env bash
set -euo pipefail

codex_repo_root() {
  git rev-parse --show-toplevel
}

codex_find_mise() {
  if command -v mise >/dev/null 2>&1; then
    command -v mise
    return 0
  fi
  if [[ -x "${HOME}/.local/bin/mise" ]]; then
    printf '%s\n' "${HOME}/.local/bin/mise"
    return 0
  fi
  return 1
}

codex_install_mise_if_missing() {
  local mise_bin
  if mise_bin="$(codex_find_mise)"; then
    printf '%s\n' "$mise_bin"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || {
    echo "curl is required to install mise" >&2
    return 127
  }

  curl -fsSL https://mise.run | sh
  codex_find_mise
}

codex_trust_repo() {
  local mise_bin="$1"
  local root="$2"
  "$mise_bin" trust "$root/mise.toml" >/dev/null
}
