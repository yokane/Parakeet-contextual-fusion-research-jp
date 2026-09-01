run-once: 2026-09-02-prebuilt-base-thin-runtime-v2
source: previous-commit
purpose: prebuild immutable CUDA NeMo dependency base once, then direct-push thin runtime and verify exact digest on Vast RTX 4090
strategy: buildx-docker-container-direct-ghcr-push
