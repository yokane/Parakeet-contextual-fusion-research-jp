#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

log(){ printf '[hf-bootstrap-bucket] %s\n' "$*"; }
fail(){ printf '[hf-bootstrap-bucket] ERROR: %s\n' "$*" >&2; exit 1; }

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
hf buckets info "$BUCKET" --token "$HF_TOKEN" >/dev/null

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LISTING="$WORK/listing.txt"
hf buckets list "$BUCKET" -R -q --token "$HF_TOKEN" > "$LISTING"

object_exists(){ grep -Fxq "$1" "$LISTING"; }
put_text_if_missing(){
  local remote_path="$1" local_name="$2" content="$3"
  if object_exists "$remote_path"; then
    log "Exists: $remote_path"
    return 0
  fi
  printf '%s' "$content" > "$WORK/$local_name"
  hf buckets cp --token "$HF_TOKEN" "$WORK/$local_name" "hf://buckets/${BUCKET}/${remote_path}" >/dev/null
  printf '%s\n' "$remote_path" >> "$LISTING"
  log "Created: $remote_path"
}

put_text_if_missing "README.md" "root.md" "# J-PACF-YOMI-TDT Work Bucket\n\nDevelopment/evaluation storage for J-PACF-YOMI-TDT. Accepted releases are promoted to saeeew/J-PACF-YOMI-tdt.\n"

for root_name in config candidates experiments runs benchmarks reference scripts tmp; do
  put_text_if_missing \
    "${root_name}/README.md" \
    "${root_name}.md" \
    "# ${root_name}\n\nManaged by yokane/Parakeet-contextual-fusion-research-jp. Do not mutate immutable-by-policy entries in place.\n"
done

put_text_if_missing \
  "config/current.json" \
  "current.json" \
  $'{\n  "schema_version": 1,\n  "active": null\n}\n'

log "Bucket bootstrap complete: hf://buckets/${BUCKET}"
