# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    UV_CACHE_DIR=/root/.cache/uv

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && printf '%s\n' 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libsndfile1 \
      ninja-build

WORKDIR /workspace/parakeet-context-fusion

# Dependency boundary. The project lock is copied before any source so source-only
# edits cannot invalidate the expensive ASR dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir uv==0.12.8 \
    && uv export \
      --locked \
      --extra nemo \
      --no-emit-project \
      --no-emit-package torch \
      --format requirements-txt \
      --output-file /tmp/runtime-requirements.txt \
    && uv pip install \
      --system \
      --require-hashes \
      --requirements /tmp/runtime-requirements.txt \
    && rm -f /tmp/runtime-requirements.txt

# Source boundary. The editable project install is intentionally after the locked
# third-party environment and never resolves dependencies.
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY configs ./configs
COPY experiments ./experiments
COPY hf_model ./hf_model
COPY locks ./locks
COPY mise.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e .

CMD ["python", "-m", "parakeet_context_fusion.cli", "--help"]
