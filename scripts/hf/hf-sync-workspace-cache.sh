#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"
mode="${1:-}"
key="${2:-}"
[[ "$mode" == "push" || "$mode" == "pull" ]] || { echo "usage: $0 <push|pull> <cache-key>" >&2; exit 2; }
[[ "$key" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid cache key" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 2; }
if [[ -z "${HF_BUCKET:-}" ]]; then
  HF_BUCKET="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('configs/hf-storage.json').read_text(encoding='utf-8'))['bucket'])
PY
)"
fi
bucket="$(hf_normalize_bucket_id "$HF_BUCKET")"
remote="hf://buckets/${bucket}/workspace-cache/${key}"
state_root="${JPA_CF_STATE_ROOT:-${ROOT}/.jpacf-state}"
artifact_dir="${JPA_CF_ARTIFACT_DIR:-${state_root}/artifacts}"
generated_dir="${JPA_CF_GENERATED_DIR:-${state_root}/generated}"
mkdir -p "$artifact_dir" "$generated_dir"
if [[ "$mode" == "pull" ]]; then
  hf_bucket_cli buckets sync "${remote}/artifacts" "$artifact_dir" --token "$HF_TOKEN"
  hf_bucket_cli buckets sync "${remote}/generated" "$generated_dir" --token "$HF_TOKEN"
  echo "restored ${remote}"
  exit 0
fi
existing="$(hf_bucket_cli buckets list "${bucket}/workspace-cache/${key}" -R -q --token "$HF_TOKEN" || true)"
[[ -z "$existing" ]] || { echo "workspace cache already exists: ${key}" >&2; exit 3; }
plan_a="$(mktemp -t jpacf-artifacts.XXXXXX.jsonl)"
plan_g="$(mktemp -t jpacf-generated.XXXXXX.jsonl)"
trap 'rm -f "$plan_a" "$plan_g"' EXIT
hf_bucket_cli buckets sync "$artifact_dir" "${remote}/artifacts" --token "$HF_TOKEN" --plan "$plan_a"
hf_bucket_cli buckets sync "$generated_dir" "${remote}/generated" --token "$HF_TOKEN" --plan "$plan_g"
hf_bucket_cli buckets sync --token "$HF_TOKEN" --apply "$plan_a"
hf_bucket_cli buckets sync --token "$HF_TOKEN" --apply "$plan_g"
echo "published ${remote}"
