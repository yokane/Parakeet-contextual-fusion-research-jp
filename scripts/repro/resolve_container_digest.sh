#!/usr/bin/env bash
set -euo pipefail

SOURCE_JSON="${1:-locks/upstream-sources.json}"
OUTPUT_JSON="${2:-locks/containers.lock.json}"

command -v docker >/dev/null 2>&1 || {
  echo "docker CLI is required to resolve OCI digests" >&2
  exit 2
}

readarray -t values < <(
  python - "$SOURCE_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = payload["base_container"]
print(base["repository"])
print(base["version_tag"])
PY
)
repository="${values[0]}"
version_tag="${values[1]}"
source_ref="${repository}:${version_tag}"

digest="$(docker buildx imagetools inspect "$source_ref" | awk '/^Digest:/ {print $2; exit}')"
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "failed to resolve a SHA-256 OCI digest for ${source_ref}" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"
python - "$OUTPUT_JSON" "$repository" "$version_tag" "$digest" <<'PY'
import json
import sys
from pathlib import Path

output, repository, version_tag, digest = sys.argv[1:]
payload = {
    "schema_version": 1,
    "images": {
        "nvidia_pytorch": {
            "repository": repository,
            "source_version_tag": version_tag,
            "digest": digest,
            "reference": f"{repository}@{digest}",
        }
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
