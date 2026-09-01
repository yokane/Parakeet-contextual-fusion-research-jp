#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
files=(
  stack.lock.yaml
  locks/hf-revisions.lock.json
  locks/containers.lock.json
  scripts/research/prepare_e00_e04.sh
  scripts/train_ngpulm.sh
  scripts/materialize_hf_eval.py
  scripts/materialize_locked_model.py
)
for path in "${files[@]}"; do
  [[ -f "$path" ]] || { echo "missing cache-key input: $path" >&2; exit 2; }
done
digest="$({ for path in "${files[@]}"; do sha256sum "$path"; done; } | sha256sum | awk '{print $1}')"
printf 'jpacf-%s\n' "${digest:0:24}"
