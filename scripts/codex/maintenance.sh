#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

ROOT="$(codex_repo_root)"
cd "$ROOT"

if ! MISE_BIN="$(codex_find_mise)"; then
  exec bash "$SCRIPT_DIR/setup.sh"
fi

codex_trust_repo "$MISE_BIN" "$ROOT"
"$MISE_BIN" install --locked
"$MISE_BIN" exec -- uv sync --locked --extra dev
"$MISE_BIN" run hf:transport:sync

bash "$SCRIPT_DIR/preflight.sh"
