#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-push-candidate] %s\n' "$*"; }
fail(){ printf '[hf-push-candidate] ERROR: %s\n' "$*" >&2; exit 1; }

SOURCE="${1:-}"
[[ $# -eq 1 && -d "$SOURCE" ]] || fail "Usage: $0 <candidate-directory>"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is required"
command -v hf >/dev/null 2>&1 || fail "hf CLI is unavailable"
command -v gh >/dev/null 2>&1 || fail "gh CLI is unavailable"

if [[ -z "${HF_BUCKET:-}" ]]; then
  HF_BUCKET="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('configs/hf-storage.json').read_text(encoding='utf-8'))['bucket'])
PY
)"
fi
BUCKET="$(hf_normalize_bucket_id "$HF_BUCKET")" || fail "invalid HF_BUCKET: $HF_BUCKET"
SOURCE="$(cd -- "$SOURCE" >/dev/null 2>&1 && pwd -P)"
[[ ! -e "$SOURCE/.candidate-id" ]] || fail "refusing to republish a fetched candidate containing .candidate-id"

python scripts/hf/validate_candidate.py "$SOURCE" >/dev/null
export JPA_CF_RELEASE="$(python - "$SOURCE/metadata.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("release") or "unknown")
PY
)"

CANDIDATE_ID="$(bash scripts/hf/hf-request-id.sh candidates)"
REMOTE="hf://buckets/${BUCKET}/candidates/${CANDIDATE_ID}"
PLAN="$(mktemp -t jpacf-candidate-plan.XXXXXX.jsonl)"
trap 'rm -f "$PLAN"' EXIT

hf buckets sync \
  --token "$HF_TOKEN" \
  "$SOURCE" \
  "$REMOTE" \
  --plan "$PLAN"

SUMMARY="$(python scripts/hf/validate_sync_plan.py "$PLAN" --expected-dest "$REMOTE")"
UPLOAD_COUNT="$(printf '%s\n' "$SUMMARY" | sed -n 's/^upload_count=//p')"
[[ "$UPLOAD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "sync plan did not contain a positive upload count"
log "Validated fresh candidate plan: ${UPLOAD_COUNT} uploads"

hf buckets sync --token "$HF_TOKEN" --apply "$PLAN"
log "Candidate ID: $CANDIDATE_ID"
log "Published: $REMOTE"
printf '%s\n' "$CANDIDATE_ID"
