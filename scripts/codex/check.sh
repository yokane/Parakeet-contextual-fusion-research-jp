#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

ROOT="$(codex_repo_root)"
cd "$ROOT"

bash "$SCRIPT_DIR/preflight.sh"
MISE_BIN="$(codex_find_mise)"
"$MISE_BIN" run ci
git diff --check

echo "Codex validation passed: canonical CPU/static CI is green."
