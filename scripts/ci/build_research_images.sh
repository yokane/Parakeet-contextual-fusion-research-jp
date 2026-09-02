#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${SOURCE_REPOSITORY:?SOURCE_REPOSITORY is required}"
: "${RUNTIME_IMAGE:?RUNTIME_IMAGE is required}"
: "${GHCR_PHASE_REPO:?GHCR_PHASE_REPO is required}"
: "${GHCR_TOOLS_REPO:?GHCR_TOOLS_REPO is required}"

KENLM_REVISION="${KENLM_REVISION:-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5}"
PUBLISH_CURRENT="${PUBLISH_CURRENT:-false}"
DOCKERHUB_REPOSITORY="${DOCKERHUB_REPOSITORY:-}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$KENLM_REVISION" =~ ^[0-9a-f]{40}$ ]]

mkdir -p dist
records="$(mktemp)"
trap 'rm -f "$records"' EXIT
LAST_PUBLISHED_REF=""

normalize_dockerhub_repo() {
  local value="$1"
  value="${value#docker.io/}"
  value="${value#https://docker.io/}"
  value="${value#https://hub.docker.com/r/}"
  [[ "$value" == */* ]] || return 1
  printf 'docker.io/%s\n' "$value"
}

DOCKERHUB_REPO=""
if [[ -n "$DOCKERHUB_REPOSITORY" ]]; then
  DOCKERHUB_REPO="$(normalize_dockerhub_repo "$DOCKERHUB_REPOSITORY")"
fi

resolve_digest() {
  local ref="$1" out digest
  out="$(docker buildx imagetools inspect "$ref")"
  digest="$(printf '%s\n' "$out" | sed -nE 's/^Digest:[[:space:]]*(sha256:[0-9a-f]{64})$/\1/p' | head -1)"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || return 2
  printf '%s\n' "$digest"
}

exists() {
  docker buildx imagetools inspect "$1" >/dev/null 2>&1
}

publish_one() {
  local name="$1" suffix="$2" dockerfile="$3" target="$4"
  shift 4
  local ghcr_ref="${GHCR_PHASE_REPO}:${suffix}" docker_ref="" selected="" registry="ghcr"
  if [[ "$name" == tools-* ]]; then
    ghcr_ref="${GHCR_TOOLS_REPO}:${suffix}"
  fi
  if [[ -n "$DOCKERHUB_REPO" ]]; then
    docker_ref="${DOCKERHUB_REPO}:${suffix}"
  fi

  if exists "$ghcr_ref"; then
    echo "Reusing immutable image: $ghcr_ref"
    selected="$ghcr_ref"
  else
    args=(
      docker buildx build --progress=plain --platform linux/amd64
      --file "$dockerfile" --provenance=false --push --tag "$ghcr_ref"
    )
    if [[ -n "$target" ]]; then
      args+=(--target "$target")
    fi
    args+=("$@" .)
    if "${args[@]}"; then
      selected="$ghcr_ref"
    else
      if [[ -z "$docker_ref" ]]; then
        echo "GHCR push failed for ${name} and Docker Hub fallback is not configured" >&2
        return 1
      fi
      echo "GHCR push failed for ${name}; retrying against public Docker Hub" >&2
      args=(
        docker buildx build --progress=plain --platform linux/amd64
        --file "$dockerfile" --provenance=false --push --tag "$docker_ref"
      )
      if [[ -n "$target" ]]; then
        args+=(--target "$target")
      fi
      args+=("$@" .)
      "${args[@]}"
      selected="$docker_ref"
      registry="dockerhub"
    fi
  fi

  local digest reference
  digest="$(resolve_digest "$selected")"
  reference="${selected%@*}@${digest}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$selected" "$digest" "$registry" "$reference" >> "$records"
  LAST_PUBLISHED_REF="$reference"
  echo "Published ${name}: ${reference}"
}

# Resolve the heavy parent remotely. Never docker-pull the multi-GB CUDA/NeMo
# image into a GitHub-hosted runner just to discover its digest.
runtime_resolved="$(bash scripts/container/resolve-remote-image.sh "$RUNTIME_IMAGE")"
[[ "$runtime_resolved" =~ @sha256:[0-9a-f]{64}$ ]]
echo "Resolved runtime: $runtime_resolved"

kenlm_suffix="kenlm-${KENLM_REVISION:0:12}"
publish_one tools-kenlm "$kenlm_suffix" docker/research/Dockerfile.kenlm kenlm-tools \
  --build-arg "KENLM_REVISION=${KENLM_REVISION}" \
  --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}"
kenlm_ref="$LAST_PUBLISHED_REF"

phone_suffix="phone-e05-${SOURCE_SHA}"
publish_one tools-e05-phone "$phone_suffix" docker/research/Dockerfile.e05-phone-cpu phone-cpu \
  --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}" \
  --build-arg "SOURCE_REVISION=${SOURCE_SHA}"

for phase in e00 e01 e03 e04 e05 e06; do
  publish_one "phase-${phase}" "phase-${phase}-${SOURCE_SHA}" docker/phases/Dockerfile "$phase" \
    --build-arg "RUNTIME_IMAGE=${runtime_resolved}" \
    --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}" \
    --build-arg "SOURCE_REVISION=${SOURCE_SHA}"
done

publish_one phase-e02 "phase-e02-${SOURCE_SHA}" docker/research/Dockerfile.e02 e02 \
  --build-arg "RUNTIME_IMAGE=${runtime_resolved}" \
  --build-arg "KENLM_TOOLS_IMAGE=${kenlm_ref}" \
  --build-arg "KENLM_REVISION=${KENLM_REVISION}" \
  --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}" \
  --build-arg "SOURCE_REVISION=${SOURCE_SHA}"

python - "$records" "$runtime_resolved" "$SOURCE_SHA" <<'PY'
import json,sys
from pathlib import Path
rows=[]
for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    name,tag,digest,registry,reference=line.split('\t')
    rows.append({'name':name,'tag':tag,'digest':digest,'registry':registry,'reference':reference})
payload={'schema_version':1,'source_sha':sys.argv[3],'runtime_parent':sys.argv[2],'images':rows}
Path('dist/research-images.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

if [[ "$PUBLISH_CURRENT" == "true" ]]; then
  while IFS=$'\t' read -r name tag _digest _registry reference; do
    case "$name" in
      phase-*) alias="${name}-current" ;;
      tools-kenlm) alias="kenlm-current" ;;
      tools-e05-phone) alias="phone-e05-current" ;;
      *) continue ;;
    esac
    repo="${tag%:*}"
    docker buildx imagetools create --tag "${repo}:${alias}" "$reference"
  done < "$records"
fi
