#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$repo_root"

state_root="${JPA_CF_STATE_ROOT:-/workspace/state}"
cache_root="${XDG_CACHE_HOME:-/home/vscode/.cache}"

# Named volumes are created by Docker as root. Normalize ownership before mise,
# uv, Hugging Face, or research scripts write persistent state into them.
sudo mkdir -p \
  "${MISE_DATA_DIR:-/home/vscode/.local/share/mise}" \
  "${MISE_CACHE_DIR:-${cache_root}/mise}" \
  "${UV_CACHE_DIR:-${cache_root}/uv}" \
  "${HF_HOME:-${cache_root}/huggingface}" \
  "$state_root"
sudo chown -R "$(id -u):$(id -g)" \
  "${MISE_DATA_DIR:-/home/vscode/.local/share/mise}" \
  "$cache_root" \
  "$state_root"

git config --global --add safe.directory "$repo_root" 2>/dev/null || true

# Keep the local toolchain aligned with GitHub Actions and fail if mise.lock
# cannot satisfy the repository's Linux/x86_64 contract.
mise trust "$repo_root/mise.toml"
mise --locked install
mise run deps:sync
mise run hf:transport:sync

git lfs install --local >/dev/null

printf '\nJ-PACF devcontainer ready\n'
printf '  Python: %s\n' "$(python --version 2>&1)"
printf '  uv:     %s\n' "$(uv --version 2>&1)"
printf '  mise:   %s\n' "$(mise --version 2>&1)"
printf '  state:  %s\n' "$state_root"
printf '\nRun `mise run ci` for the canonical CPU/static checks.\n'
