#!/usr/bin/env bash
set -euo pipefail

: "${REGISTRY:?REGISTRY is required}"
: "${IMAGE_NAME:?IMAGE_NAME is required}"
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${SOURCE_REPOSITORY:?SOURCE_REPOSITORY is required}"
: "${NVIDIA_BASE_IMAGE:?NVIDIA_BASE_IMAGE is required}"

command -v docker >/dev/null
command -v sha256sum >/dev/null
command -v python >/dev/null
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$NVIDIA_BASE_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]

image_repo="${REGISTRY}/${IMAGE_NAME}"

resolve_digest() {
  local ref="$1"
  local out digest
  out="$(docker buildx imagetools inspect "$ref")"
  digest="$(printf '%s\n' "$out" | sed -nE 's/^Digest:[[:space:]]*(sha256:[0-9a-f]{64})$/\1/p' | head -1)"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    printf 'could not resolve digest for %s\n%s\n' "$ref" "$out" >&2
    return 2
  }
  printf '%s\n' "$digest"
}

# Hash only inputs that can alter the expensive CUDA/NeMo/Python dependency
# rootfs. Source code is intentionally excluded so ordinary commits reuse it.
base_key="$({
  sha256sum Dockerfile.runtime-base .dockerignore pyproject.toml uv.lock locks/containers.lock.json
  find tools/hf-bucket -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
} | sha256sum | awk '{print $1}')"
[[ "$base_key" =~ ^[0-9a-f]{64}$ ]]

base_tag="${image_repo}:base-${base_key}"
base_current="${image_repo}:base-current"
base_digest=""

if docker buildx imagetools inspect "$base_tag" >/dev/null 2>&1; then
  echo "Reusing dependency base: $base_tag"
  base_digest="$(resolve_digest "$base_tag")"
else
  echo "Dependency base is missing; building it once: $base_tag"
  cache_args=()
  if docker buildx imagetools inspect "$base_current" >/dev/null 2>&1; then
    cache_args+=(--cache-from "type=registry,ref=${base_current}")
  fi

  metadata="${RUNNER_TEMP:-/tmp}/jpacf-base-metadata.json"
  rm -f "$metadata"
  docker buildx build \
    --progress=plain \
    --platform linux/amd64 \
    --file Dockerfile.runtime-base \
    --build-arg "NVIDIA_BASE_IMAGE=${NVIDIA_BASE_IMAGE}" \
    --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}" \
    --label "org.opencontainers.image.revision=deps-${base_key}" \
    --label "org.opencontainers.image.source=${SOURCE_REPOSITORY}" \
    --cache-to type=inline \
    "${cache_args[@]}" \
    --provenance=false \
    --push \
    --tag "$base_tag" \
    --tag "$base_current" \
    --metadata-file "$metadata" \
    .

  base_digest="$(python - "$metadata" <<'PY'
import json, re, sys
payload=json.load(open(sys.argv[1], encoding='utf-8'))
value=str(payload.get('containerimage.digest') or '')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
    raise SystemExit(2)
print(value)
PY
  )"
fi

[[ "$base_digest" =~ ^sha256:[0-9a-f]{64}$ ]]
base_reference="${image_repo}@${base_digest}"
echo "Pinned dependency base: $base_reference"

runtime_tag="${image_repo}:sha-${SOURCE_SHA}"
runtime_current="${image_repo}:runtime-current"
runtime_cache_args=()
if docker buildx imagetools inspect "$runtime_current" >/dev/null 2>&1; then
  runtime_cache_args+=(--cache-from "type=registry,ref=${runtime_current}")
fi

metadata="${RUNNER_TEMP:-/tmp}/jpacf-runtime-metadata.json"
rm -f "$metadata"

# docker-container Buildx with --push exports directly to GHCR. It deliberately
# avoids --load, so the multi-gigabyte parent rootfs is never materialized in the
# hosted runner's Docker image store during the canonical build.
docker buildx build \
  --progress=plain \
  --platform linux/amd64 \
  --file Dockerfile \
  --build-arg "BASE_IMAGE=${base_reference}" \
  --build-arg "SOURCE_REPOSITORY=${SOURCE_REPOSITORY}" \
  --label "org.opencontainers.image.revision=${SOURCE_SHA}" \
  --label "org.opencontainers.image.source=${SOURCE_REPOSITORY}" \
  --label "io.jpacf.runtime-base.digest=${base_digest}" \
  --cache-to type=inline \
  "${runtime_cache_args[@]}" \
  --provenance=false \
  --push \
  --tag "$runtime_tag" \
  --tag "$runtime_current" \
  --metadata-file "$metadata" \
  .

runtime_digest="$(python - "$metadata" <<'PY'
import json, re, sys
payload=json.load(open(sys.argv[1], encoding='utf-8'))
value=str(payload.get('containerimage.digest') or '')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
    raise SystemExit(2)
print(value)
PY
)"
[[ "$runtime_digest" =~ ^sha256:[0-9a-f]{64}$ ]]
runtime_reference="${image_repo}@${runtime_digest}"

# Remote manifest validation is enough here. Pulling the image back into the
# hosted Docker daemon would recreate the exact large-I/O failure this path is
# designed to avoid. GPU execution is verified later on the target runtime.
docker buildx imagetools inspect "$runtime_reference" >/dev/null

echo "Published runtime: $runtime_reference"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "base_key=$base_key"
    echo "base_digest=$base_digest"
    echo "base_reference=$base_reference"
    echo "digest=$runtime_digest"
    echo "published=$runtime_tag"
    echo "image=$runtime_reference"
  } >> "$GITHUB_OUTPUT"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### GHCR runtime build"
    echo
    echo "- dependency base: \`$base_reference\`"
    echo "- runtime: \`$runtime_reference\`"
    echo "- source: \`$SOURCE_SHA\`"
  } >> "$GITHUB_STEP_SUMMARY"
fi
