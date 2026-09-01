#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-allocate-id] %s\n' "$*" >&2; }
fail(){ printf '[hf-allocate-id] ERROR: %s\n' "$*" >&2; exit 1; }

COLLECTION="${1:-}"
[[ "$COLLECTION" == "candidates" || "$COLLECTION" == "experiments" || "$COLLECTION" == "config" ]] \
  || fail "collection must be candidates, experiments, or config"
[[ $# -eq 1 ]] || fail "Usage: $0 <candidates|experiments|config>"
[[ "${HF_ALLOCATOR_INTERNAL:-}" == "1" ]] || fail "allocation must run inside hf-central-allocator.yml"
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

case "$COLLECTION" in
  candidates)
    PREFIX="candidate"
    REMOTE_ROOT="hf://buckets/${BUCKET}/candidates"
    ;;
  experiments)
    PREFIX="experiment"
    REMOTE_ROOT="hf://buckets/${BUCKET}/experiments"
    ;;
  config)
    PREFIX="config"
    REMOTE_ROOT="hf://buckets/${BUCKET}/config/versions"
    ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LISTING="$WORK/listing.txt"
README="$WORK/README.md"

# One complete recursive listing is the allocation source of truth. The
# workflow-level global concurrency lock prevents two allocators from observing
# the same maximum sequence.
hf buckets list "$BUCKET" -R -q --token "$HF_TOKEN" > "$LISTING"
ID="$(python scripts/hf/next_sequence_id.py --prefix "$PREFIX" --listing "$LISTING")"
SEQUENCE="${ID##*-}"
CREATED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

HF_ALLOCATED_ID="$ID" \
HF_ALLOCATED_COLLECTION="$COLLECTION" \
HF_ALLOCATED_AT="$CREATED_AT" \
HF_ALLOCATED_SEQUENCE="$SEQUENCE" \
HF_ALLOCATED_BUCKET="$BUCKET" \
python - "$README" <<'PY'
import json
import os
import sys
from pathlib import Path

metadata_raw = os.environ.get("HF_ALLOCATION_METADATA_JSON", "{}")
try:
    metadata = json.loads(metadata_raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"HF_ALLOCATION_METADATA_JSON is invalid JSON: {exc}") from exc

payload = {
    "allocation_id": os.environ["HF_ALLOCATED_ID"],
    "collection": os.environ["HF_ALLOCATED_COLLECTION"],
    "bucket": os.environ["HF_ALLOCATED_BUCKET"],
    "sequence": int(os.environ["HF_ALLOCATED_SEQUENCE"]),
    "allocated_at": os.environ["HF_ALLOCATED_AT"],
    "metadata": metadata,
}
text = (
    f"# {payload['allocation_id']}\n\n"
    "This path is reserved by the J-PACF central allocator. Artifacts under this prefix are immutable by policy.\n\n"
    "```json\n"
    + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    + "\n```\n"
)
Path(sys.argv[1]).write_text(text, encoding="utf-8")
PY

hf buckets cp --token "$HF_TOKEN" "$README" "${REMOTE_ROOT}/${ID}/README.md" >/dev/null
log "Allocated ${COLLECTION}/${ID} in ${BUCKET}"
printf '%s\n' "$ID"
