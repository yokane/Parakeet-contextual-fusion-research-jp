#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-fetch-candidate] %s\n' "$*"; }
fail(){ printf '[hf-fetch-candidate] ERROR: %s\n' "$*" >&2; exit 1; }

CANDIDATE_ID="${1:-}"
DESTINATION="${2:-artifacts/candidates/${CANDIDATE_ID}}"
[[ "$CANDIDATE_ID" =~ ^candidate-[0-9]{6}$ ]] || fail "candidate ID must match candidate-NNNNNN"
[[ $# -le 2 ]] || fail "Usage: $0 <candidate-NNNNNN> [destination]"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is required"
command -v hf >/dev/null 2>&1 || fail "hf CLI is unavailable"

if [[ -z "${HF_BUCKET:-}" ]]; then
  HF_BUCKET="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('configs/hf-storage.json').read_text(encoding='utf-8'))['bucket'])
PY
)"
fi
BUCKET="$(hf_normalize_bucket_id "$HF_BUCKET")" || fail "invalid HF_BUCKET: $HF_BUCKET"
[[ ! -e "$DESTINATION" ]] || fail "destination already exists: $DESTINATION"

PARENT="$(dirname -- "$DESTINATION")"
mkdir -p "$PARENT"
STAGING="$(mktemp -d "${PARENT}/.${CANDIDATE_ID}.staging.XXXXXX")"
cleanup(){ rm -rf "$STAGING"; }
trap cleanup EXIT

REMOTE="hf://buckets/${BUCKET}/candidates/${CANDIDATE_ID}"
hf buckets sync --token "$HF_TOKEN" "$REMOTE" "$STAGING"
python scripts/hf/validate_candidate.py "$STAGING" >/dev/null
printf '%s\n' "$CANDIDATE_ID" > "$STAGING/.candidate-id"
mv "$STAGING" "$DESTINATION"
trap - EXIT

log "Fetched ${REMOTE} -> ${DESTINATION}"
printf '%s\n' "$DESTINATION"
