#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-push-run] %s\n' "$*"; }
fail(){ printf '[hf-push-run] ERROR: %s\n' "$*" >&2; exit 1; }

RUN_DIRECTORY="${1:-}"
[[ $# -eq 1 && -d "$RUN_DIRECTORY" ]] || fail "Usage: $0 <run-directory>"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is required"
command -v uv >/dev/null 2>&1 || fail "uv is unavailable; enter through mise"

if [[ -z "${HF_BUCKET:-}" ]]; then
  HF_BUCKET="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('configs/hf-storage.json').read_text(encoding='utf-8'))['bucket'])
PY
)"
fi
BUCKET="$(hf_normalize_bucket_id "$HF_BUCKET")" || fail "invalid HF_BUCKET: $HF_BUCKET"
RUN_DIRECTORY="$(cd -- "$RUN_DIRECTORY" >/dev/null 2>&1 && pwd -P)"

RUN_ID="$(python scripts/hf/validate_run_bundle.py "$RUN_DIRECTORY" | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid run ID from bundle: $RUN_ID"
REMOTE="hf://buckets/${BUCKET}/runs/${RUN_ID}"

# Runs are append-only evidence. Any existing object under this run ID means a
# rerun would mutate history, so fail before creating a plan.
EXISTING="$(hf_bucket_cli buckets list "${BUCKET}/runs/${RUN_ID}" -R -q --token "$HF_TOKEN" || true)"
[[ -z "$EXISTING" ]] || fail "run already exists in Bucket: ${RUN_ID}"

PLAN="$(mktemp -t jpacf-run-plan.XXXXXX.jsonl)"
trap 'rm -f "$PLAN"' EXIT
hf_bucket_cli buckets sync \
  --token "$HF_TOKEN" \
  "$RUN_DIRECTORY" \
  "$REMOTE" \
  --plan "$PLAN"

SUMMARY="$(python scripts/hf/validate_sync_plan.py "$PLAN" --expected-dest "$REMOTE")"
UPLOAD_COUNT="$(printf '%s\n' "$SUMMARY" | sed -n 's/^upload_count=//p')"
[[ "$UPLOAD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "sync plan did not contain a positive upload count"
log "Validated append-only run plan: ${UPLOAD_COUNT} uploads"

hf_bucket_cli buckets sync --token "$HF_TOKEN" --apply "$PLAN"
log "Published run: $REMOTE"
printf '%s\n' "$RUN_ID"
