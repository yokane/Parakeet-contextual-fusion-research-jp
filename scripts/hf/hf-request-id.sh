#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-request-id] %s\n' "$*" >&2; }
fail(){ printf '[hf-request-id] ERROR: %s\n' "$*" >&2; exit 1; }

COLLECTION="${1:-}"
[[ "$COLLECTION" == "candidates" || "$COLLECTION" == "experiments" || "$COLLECTION" == "config" ]] \
  || fail "collection must be candidates, experiments, or config"
[[ $# -eq 1 ]] || fail "Usage: $0 <candidates|experiments|config>"
command -v gh >/dev/null 2>&1 || fail "GitHub CLI (gh) is required"

if [[ -z "${HF_BUCKET:-}" ]]; then
  HF_BUCKET="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('configs/hf-storage.json').read_text(encoding='utf-8'))['bucket'])
PY
)"
fi
BUCKET="$(hf_normalize_bucket_id "$HF_BUCKET")" || fail "invalid HF_BUCKET: $HF_BUCKET"

if [[ -z "${GH_TOKEN:-}" && -n "${HF_ALLOCATOR_GITHUB_TOKEN:-}" ]]; then
  export GH_TOKEN="$HF_ALLOCATOR_GITHUB_TOKEN"
fi
[[ -n "${GH_TOKEN:-}" ]] || fail "GH_TOKEN or HF_ALLOCATOR_GITHUB_TOKEN is required"

ALLOCATOR_REPOSITORY="${HF_ALLOCATOR_REPOSITORY:-${GITHUB_REPOSITORY:-yokane/Parakeet-contextual-fusion-research-jp}}"
ALLOCATOR_WORKFLOW="${HF_ALLOCATOR_WORKFLOW:-hf-central-allocator.yml}"
ALLOCATOR_REF="${HF_ALLOCATOR_REF:-main}"
REQUEST_ID="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${COLLECTION}-$(date -u +%s)-$$"

METADATA_JSON="$(python - <<'PY'
import json
import os
print(json.dumps({
    "source_repository": os.environ.get("GITHUB_REPOSITORY"),
    "source_run_id": os.environ.get("GITHUB_RUN_ID"),
    "source_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    "source_workflow": os.environ.get("GITHUB_WORKFLOW"),
    "source_sha": os.environ.get("GITHUB_SHA"),
    "candidate_release": os.environ.get("JPA_CF_RELEASE"),
}, separators=(",", ":")))
PY
)"

log "Dispatching ${COLLECTION} allocation to ${ALLOCATOR_REPOSITORY}"
gh workflow run "$ALLOCATOR_WORKFLOW" \
  --repo "$ALLOCATOR_REPOSITORY" \
  --ref "$ALLOCATOR_REF" \
  -f "request_id=${REQUEST_ID}" \
  -f "hf_bucket=${BUCKET}" \
  -f "collection=${COLLECTION}" \
  -f "metadata_json=${METADATA_JSON}"

RUN_ID=""
for _ in $(seq 1 60); do
  RUN_ID="$(gh run list \
    --repo "$ALLOCATOR_REPOSITORY" \
    --workflow "$ALLOCATOR_WORKFLOW" \
    --event workflow_dispatch \
    --limit 100 \
    --json databaseId,displayTitle \
    --jq ".[] | select(.displayTitle == \"HF allocate ${REQUEST_ID}\") | .databaseId" \
    | head -n 1)"
  [[ -n "$RUN_ID" ]] && break
  sleep 2
done
[[ -n "$RUN_ID" ]] || fail "central allocator run not found for request ${REQUEST_ID}"

gh run watch "$RUN_ID" --repo "$ALLOCATOR_REPOSITORY" --exit-status >/dev/null

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
gh run download "$RUN_ID" \
  --repo "$ALLOCATOR_REPOSITORY" \
  --name "hf-allocation-${REQUEST_ID}" \
  --dir "$TMP" >/dev/null

ID="$(python - "$TMP/allocation.json" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allocation_id = value.get("allocation_id")
if not isinstance(allocation_id, str) or not allocation_id:
    raise SystemExit("allocation response has no allocation_id")
print(allocation_id)
PY
)"
log "Allocated ${ID} via workflow run ${RUN_ID}"
printf '%s\n' "$ID"
