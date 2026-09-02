# syntax=docker/dockerfile:1.7
# The repository lock is intentionally Linux/x86_64-only. Keep the local
# development container on the same platform even when Docker Desktop runs on
# an ARM host; emulation is acceptable for CPU/static development.
FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG MISE_VERSION=2026.8.16

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      jq \
      libsndfile1 \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Match the mise version used by canonical GitHub Actions. Python and uv are
# still resolved from mise.lock at post-create time rather than baked here.
RUN curl -fsSL https://mise.run \
      | MISE_VERSION="v${MISE_VERSION}" MISE_INSTALL_PATH=/usr/local/bin/mise sh \
    && mise --version \
    && test "$(mise --version | awk '{print $2}')" = "${MISE_VERSION}"

ENV MISE_DATA_DIR=/home/vscode/.local/share/mise \
    MISE_CACHE_DIR=/home/vscode/.cache/mise \
    UV_CACHE_DIR=/home/vscode/.cache/uv \
    HF_HOME=/home/vscode/.cache/huggingface \
    HF_HUB_CACHE=/home/vscode/.cache/huggingface/hub \
    HF_XET_CACHE=/home/vscode/.cache/huggingface/xet \
    JPA_CF_STATE_ROOT=/workspace/state \
    PATH=/home/vscode/.local/share/mise/shims:${PATH} \
    PYTHONUNBUFFERED=1

RUN mkdir -p \
      /home/vscode/.local/share/mise \
      /home/vscode/.cache/mise \
      /home/vscode/.cache/uv \
      /home/vscode/.cache/huggingface \
      /workspace/state \
    && chown -R vscode:vscode /home/vscode/.local /home/vscode/.cache /workspace/state

USER vscode
