#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-${JPA_CF_IMAGE_TAG:-ghcr.io/yokane/jpacf-yomi-tdt-runtime:main}}"
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
docker pull "$IMAGE" >/dev/null
resolved="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
[[ "$resolved" =~ @sha256:[0-9a-f]{64}$ ]] || { echo "could not resolve immutable digest for $IMAGE" >&2; exit 2; }
printf '%s\n' "$resolved"
