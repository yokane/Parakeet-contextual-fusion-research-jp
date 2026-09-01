#!/usr/bin/env bash
set -uo pipefail

python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu
rc=$?
printf 'JPA_CF_CANONICAL_VERIFY rc=%d\n' "$rc"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

# Keep the verified contract alive long enough for the controller to observe
# the success marker and record provider metadata. The workflow always destroys
# the instance after publishing evidence.
exec sleep infinity
