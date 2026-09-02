#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

ROOT="$(codex_repo_root)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Codex development setup requires Linux" >&2
  exit 2
fi
case "$(uname -m)" in
  x86_64|amd64) ;;
  *) echo "Codex development setup requires x86_64/amd64" >&2; exit 2 ;;
esac

MISE_BIN="$(codex_install_mise_if_missing)"
codex_trust_repo "$MISE_BIN" "$ROOT"

# Codex Cloud setup scripts have internet access. Materialize everything needed
# for the later agent phase, where internet access can remain disabled.
"$MISE_BIN" install --locked
"$MISE_BIN" exec -- uv sync --locked --extra dev
"$MISE_BIN" run hf:transport:sync

bash "$SCRIPT_DIR/preflight.sh"

cat <<'EOF'
Codex Cloud CPU/static environment is ready.
Use: bash scripts/codex/check.sh
GPU/NeMo validation remains delegated to the pinned GHCR/provider workflows.
EOF
