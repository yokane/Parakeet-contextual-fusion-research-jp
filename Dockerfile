# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/yokane/Parakeet-contextual-fusion-research-jp

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.title="J-PACF-YOMI-TDT portable GPU runtime" \
      org.opencontainers.image.description="Pinned CUDA13/NeMo/PyTorch research runtime for WSL2, Vast.ai and CI"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JPA_CF_IMAGE_HOME=/opt/jpacf \
    JPA_CF_WORKSPACE=/workspace/project \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_XET_CACHE=/cache/huggingface/xet \
    UV_CACHE_DIR=/cache/uv \
    XDG_CACHE_HOME=/cache/xdg \
    TORCH_HOME=/cache/torch \
    HF_TRANSPORT_PROJECT=/opt/jpacf/tools/hf-bucket \
    UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && printf '%s\n' 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libsndfile1 \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/jpacf

# uv + Python are part of the image contract. Host Python/mise is never used by
# GPU experiments after the container has been built.
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir uv==0.12.1 \
    && uv python install 3.12.3

# Keep the expensive CUDA/NeMo dependency layer independent from source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/cache/uv \
    uv sync \
      --locked \
      --python 3.12.3 \
      --extra dev \
      --extra gpu \
      --no-install-project

# Bucket transport has a deliberately isolated environment so bucket operations
# can never re-resolve or mutate the NeMo GPU environment.
COPY tools/hf-bucket ./tools/hf-bucket
RUN --mount=type=cache,target=/cache/uv \
    env -u UV_PROJECT_ENVIRONMENT \
      uv sync --project tools/hf-bucket --locked

COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY configs ./configs
COPY experiments ./experiments
COPY hf_model ./hf_model
COPY locks ./locks
COPY mise.toml mise.lock stack.lock.yaml ./

RUN --mount=type=cache,target=/cache/uv \
    uv sync \
      --locked \
      --python 3.12.3 \
      --extra dev \
      --extra gpu

ENV PATH="/opt/jpacf/.venv/bin:${PATH}" \
    PYTHONPATH="/workspace/project/src:/opt/jpacf/src"

# Runtime state is always external. On WSL2 this is normally bind-mounted from
# a Linux filesystem; on Vast.ai /workspace can be a persistent Vast Volume.
RUN mkdir -p \
      /workspace/project \
      /cache/huggingface \
      /cache/uv \
      /cache/xdg \
      /cache/torch

WORKDIR /workspace/project

CMD ["python", "/opt/jpacf/scripts/container/verify_runtime.py", "--require-gpu"]
