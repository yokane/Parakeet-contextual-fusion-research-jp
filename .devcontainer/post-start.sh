#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
state_root="${JPA_CF_STATE_ROOT:-/workspace/state}"
cache_root="${XDG_CACHE_HOME:-/home/vscode/.cache}"

# Rebuild/reopen can reattach named volumes created by a different container
# UID. Repair ownership before an interactive shell starts writing caches.
sudo mkdir -p \
  "${MISE_DATA_DIR:-/home/vscode/.local/share/mise}" \
  "$cache_root" \
  "$state_root"
sudo chown -R "$(id -u):$(id -g)" \
  "${MISE_DATA_DIR:-/home/vscode/.local/share/mise}" \
  "$cache_root" \
  "$state_root"

git config --global --add safe.directory "$repo_root" 2>/dev/null || true
