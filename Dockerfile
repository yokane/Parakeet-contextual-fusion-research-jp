# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/workspace/parakeet-context-fusion/.venv

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
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/parakeet-context-fusion

# Bootstrap only the exact package manager first. uv then installs the exact
# Python interpreter used by mise.toml/stack.lock.yaml.
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir uv==0.12.1 \
    && uv python install 3.12.3

# Dependency boundary: only dependency metadata is copied before the expensive
# NeMo/CUDA environment is synchronized. Source-only edits do not invalidate it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
      --locked \
      --python 3.12.3 \
      --extra gpu \
      --extra research \
      --no-install-project

# Source boundary: frequently changed project files are copied only after the
# locked third-party environment has been materialized.
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY configs ./configs
COPY experiments ./experiments
COPY hf_model ./hf_model
COPY locks ./locks
COPY mise.toml stack.lock.yaml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
      --locked \
      --python 3.12.3 \
      --extra gpu \
      --extra research

ENV PATH="/workspace/parakeet-context-fusion/.venv/bin:${PATH}"

CMD ["python", "-m", "parakeet_context_fusion.cli", "--help"]
