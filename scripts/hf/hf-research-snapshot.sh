#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hf-identity.sh"

CONFIG="${RESEARCH_ARTIFACT_CONFIG:-configs/research/e00-e06-artifacts.yaml}"
HF_BUCKET="${HF_BUCKET:-saeeew/J-PACF-YOMI-tdt-bucket}"

log() { printf '[hf-research-snapshot] %s\n' "$*"; }
fail() { printf '[hf-research-snapshot] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is required"
BUCKET="$(hf_normalize_bucket_id "$HF_BUCKET")" || fail "invalid HF_BUCKET: $HF_BUCKET"
command -v uv >/dev/null 2>&1 || fail "uv is required"

task_plan() {
  uv run --locked --no-sync python scripts/research/snapshot_plan.py "$@" --config "$CONFIG"
}

validate_key() {
  local key="$1"
  [[ "$key" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid research key: $key"
}

validate_ref() {
  local ref="$1" stage fingerprint extra
  IFS='/' read -r stage fingerprint extra <<< "$ref"
  [[ -n "$stage" && -n "$fingerprint" && -z "${extra:-}" ]] || fail "invalid snapshot ref: $ref"
  [[ "$stage" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid snapshot stage: $stage"
  [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || fail "invalid snapshot fingerprint: $fingerprint"
}

prefix_for_task() {
  task_plan "$1" --field json | python -c 'import json,sys; print(json.load(sys.stdin)["bucket_prefix"])'
}

output_stage() {
  task_plan "$1" --field output
}

output_ref() {
  task_plan "$1" --field output_ref
}

remote_for_ref() {
  local key="$1" task="$2" ref="$3" prefix
  validate_ref "$ref"
  prefix="$(prefix_for_task "$task")"
  printf 'hf://buckets/%s/%s/%s/%s\n' "$BUCKET" "$prefix" "$key" "$ref"
}

listing_for_ref() {
  local key="$1" task="$2" ref="$3" prefix
  validate_ref "$ref"
  prefix="$(prefix_for_task "$task")"
  hf_bucket_cli buckets list "${BUCKET}/${prefix}/${key}/${ref}" -R -q --token "$HF_TOKEN" 2>/dev/null || true
}

snapshot_exists() {
  local key="$1" task="$2" ref
  ref="$(output_ref "$task")"
  [[ -n "$(listing_for_ref "$key" "$task" "$ref")" ]]
}

pull_inputs() {
  local key="$1" task="$2" state_root="$3" ref remote listing
  mkdir -p "$state_root"
  while IFS= read -r ref; do
    [[ -n "$ref" ]] || continue
    listing="$(listing_for_ref "$key" "$task" "$ref")"
    [[ -n "$listing" ]] || fail "required snapshot is missing: key=$key ref=$ref"
    remote="$(remote_for_ref "$key" "$task" "$ref")"
    log "Restoring $remote"
    # hf buckets sync defaults to no-delete today. Keep it explicit because this
    # operation intentionally overlays several immutable stage deltas into one
    # local workspace; deleting files from an earlier stage would corrupt lineage.
    hf_bucket_cli buckets sync --no-delete --token "$HF_TOKEN" "$remote" "$state_root"
  done < <(task_plan "$task" --field input_refs)
}

build_delta() {
  local key="$1" task="$2" state_root="$3" staging="$4" stage ref fingerprint source_sha plan_json
  stage="$(output_stage "$task")"
  ref="$(output_ref "$task")"
  fingerprint="$(task_plan "$task" --field fingerprint)"
  source_sha="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  plan_json="$(task_plan "$task")"

  python - "$state_root" "$staging" "$task" "$stage" "$ref" "$fingerprint" "$key" "$source_sha" "$plan_json" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

state = Path(sys.argv[1]).resolve()
staging = Path(sys.argv[2]).resolve()
task = sys.argv[3]
stage = sys.argv[4]
output_ref = sys.argv[5]
fingerprint = sys.argv[6]
research_key = sys.argv[7]
source_sha = sys.argv[8]
plan = json.loads(sys.argv[9])
paths = [str(item) for item in plan.get("publish") or []]
if not paths:
    raise SystemExit("snapshot publish list is empty")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for rel in paths:
    src = state / rel
    dst = staging / rel
    if src.is_dir():
        if not any(src.iterdir()):
            raise SystemExit(f"empty snapshot directory: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file() and src.stat().st_size > 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        raise SystemExit(f"missing or empty snapshot output: {src}")

files = []
for path in sorted(item for item in staging.rglob("*") if item.is_file()):
    rel = path.relative_to(staging).as_posix()
    files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})
manifest = {
    "schema_version": 2,
    "research_key": research_key,
    "task": task,
    "stage": stage,
    "fingerprint": fingerprint,
    "output_ref": output_ref,
    "input_refs": [str(item) for item in plan.get("input_refs") or []],
    "source_git_sha": source_sha,
    "files": files,
}
manifest_path = staging / ".jpacf-snapshots" / f"{stage}-{fingerprint[:16]}.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

push_output() {
  local key="$1" task="$2" state_root="$3" ref remote existing staging plan summary uploads
  ref="$(output_ref "$task")"
  remote="$(remote_for_ref "$key" "$task" "$ref")"
  existing="$(listing_for_ref "$key" "$task" "$ref")"
  [[ -z "$existing" ]] || fail "immutable snapshot already exists: $remote"

  staging="$(mktemp -d -t jpacf-research-delta.XXXXXX)"
  plan="$(mktemp -t jpacf-research-plan.XXXXXX.jsonl)"
  build_delta "$key" "$task" "$state_root" "$staging"

  hf_bucket_cli buckets sync --token "$HF_TOKEN" "$staging" "$remote" --plan "$plan"
  summary="$(uv run --locked --no-sync python scripts/hf/validate_sync_plan.py "$plan" --expected-dest "$remote")"
  uploads="$(printf '%s\n' "$summary" | sed -n 's/^upload_count=//p')"
  [[ "$uploads" =~ ^[1-9][0-9]*$ ]] || fail "snapshot sync plan contained no uploads"
  hf_bucket_cli buckets sync --token "$HF_TOKEN" --apply "$plan"
  log "Published immutable delta snapshot: $remote (${uploads} uploads)"
  rm -rf "$staging"
  rm -f "$plan"
}

command="${1:-}"
case "$command" in
  exists)
    [[ $# -eq 3 ]] || fail "usage: $0 exists <research-key> <task>"
    validate_key "$2"
    if snapshot_exists "$2" "$3"; then
      printf 'true\n'
    else
      printf 'false\n'
    fi
    ;;
  remote)
    [[ $# -eq 3 ]] || fail "usage: $0 remote <research-key> <task>"
    validate_key "$2"
    ref="$(output_ref "$3")"
    remote_for_ref "$2" "$3" "$ref"
    ;;
  pull)
    [[ $# -eq 4 ]] || fail "usage: $0 pull <research-key> <task> <state-root>"
    validate_key "$2"
    pull_inputs "$2" "$3" "$4"
    ;;
  push)
    [[ $# -eq 4 ]] || fail "usage: $0 push <research-key> <task> <state-root>"
    validate_key "$2"
    push_output "$2" "$3" "$4"
    ;;
  plan)
    [[ $# -eq 2 ]] || fail "usage: $0 plan <task>"
    task_plan "$2"
    ;;
  *)
    fail "usage: $0 <exists|remote|pull|push|plan> ..."
    ;;
esac
