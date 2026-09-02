#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_common.sh"

ROOT="$(codex_repo_root)"
cd "$ROOT"

MISE_BIN="$(codex_find_mise)" || {
  echo "mise is missing; run: bash scripts/codex/setup.sh" >&2
  exit 2
}
codex_trust_repo "$MISE_BIN" "$ROOT"

[[ "$(uname -s)" == "Linux" ]] || { echo "expected Linux" >&2; exit 2; }
case "$(uname -m)" in
  x86_64|amd64) ;;
  *) echo "expected x86_64/amd64" >&2; exit 2 ;;
esac

test -x .venv/bin/python || {
  echo ".venv is missing; run: bash scripts/codex/setup.sh" >&2
  exit 2
}

python_version="$("$MISE_BIN" exec -- python -c 'import platform; print(platform.python_version())')"
uv_version="$("$MISE_BIN" exec -- uv --version | awk '{print $2}')"

[[ "$python_version" == "3.12.3" ]] || {
  echo "expected Python 3.12.3, got ${python_version}" >&2
  exit 2
}
[[ "$uv_version" == "0.12.1" ]] || {
  echo "expected uv 0.12.1, got ${uv_version}" >&2
  exit 2
}

"$MISE_BIN" exec -- python scripts/repro/verify_platform.py

git diff --check

printf 'Codex preflight OK: linux/amd64, python=%s, uv=%s, cpu/static mode\n' \
  "$python_version" "$uv_version"
