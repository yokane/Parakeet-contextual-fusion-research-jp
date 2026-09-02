#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?image reference is required}"
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
if [[ "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  printf '%s\n' "$IMAGE"
  exit 0
fi
out="$(docker buildx imagetools inspect "$IMAGE")"
digest="$(printf '%s\n' "$out" | sed -nE 's/^Digest:[[:space:]]*(sha256:[0-9a-f]{64})$/\1/p' | head -1)"
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "could not resolve remote image digest for $IMAGE" >&2
  exit 2
}
name="$(python - "$IMAGE" <<'PY'
import sys
value=sys.argv[1]
if '@' in value:
    value=value.split('@',1)[0]
slash=value.rfind('/')
colon=value.rfind(':')
if colon > slash:
    value=value[:colon]
print(value)
PY
)"
printf '%s@%s\n' "$name" "$digest"
