# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/yokane/Parakeet-contextual-fusion-research-jp

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.title="J-PACF-YOMI-TDT portable GPU runtime" \
      org.opencontainers.image.description="Pinned CUDA13/NeMo/PyTorch research runtime for WSL2, Vast.ai and CI"

# Build-time Python state must remain inside the immutable image. In particular,
# never install uv-managed Python under /workspace/state: that path is replaced by
# a writable runtime volume and would turn .venv/bin/python into a broken symlink.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JPA_CF_IMAGE_HOME=/opt/jpacf \
    JPA_CF_WORKSPACE=/opt/jpacf \
    JPA_CF_STATE_ROOT=/workspace/state \
    HF_TRANSPORT_PROJECT=/opt/jpacf/tools/hf-bucket \
    UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/jpacf/.uv-python \
    UV_PYTHON_INSTALL_BIN=0

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

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --no-cache-dir uv==0.12.1 \
    && UV_CACHE_DIR=/root/.cache/uv uv python install 3.12.3

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
      uv sync \
        --locked \
        --python 3.12.3 \
        --extra dev \
        --extra gpu \
        --no-install-project

COPY tools/hf-bucket ./tools/hf-bucket
RUN --mount=type=cache,target=/root/.cache/uv \
    env -u UV_PROJECT_ENVIRONMENT UV_CACHE_DIR=/root/.cache/uv \
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

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
      uv sync \
        --locked \
        --python 3.12.3 \
        --extra dev \
        --extra gpu

# Fail during image construction instead of at GPU runtime if the project venv
# ever becomes coupled to a mutable runtime mount again.
RUN set -euo pipefail; \
    test -x /opt/jpacf/.venv/bin/python; \
    resolved="$(readlink -f /opt/jpacf/.venv/bin/python)"; \
    case "$resolved" in \
      /opt/jpacf/.uv-python/*) ;; \
      *) echo "project Python escaped immutable image state: $resolved" >&2; exit 2 ;; \
    esac

# KenLM is compiled only for E00-E04 preparation. Keep its small required Boost
# runtime/build dependency delta in a late layer so the large pinned CUDA/NeMo
# and Python dependency layers stay reusable across runtime revisions and Vast pulls.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      libboost-program-options-dev \
      libboost-system-dev \
      libboost-test-dev \
      libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime-only mutable state is deliberately configured after all build-time uv
# operations so mounting /workspace/state cannot hide the interpreter backing
# /opt/jpacf/.venv.
ENV HOME=/workspace/state/home \
    HF_HOME=/workspace/state/hf \
    HF_HUB_CACHE=/workspace/state/hf/hub \
    HF_XET_CACHE=/workspace/state/hf/xet \
    UV_CACHE_DIR=/workspace/state/uv \
    XDG_CACHE_HOME=/workspace/state/xdg \
    TORCH_HOME=/workspace/state/torch \
    PATH="/opt/jpacf/.venv/bin:${PATH}" \
    PYTHONPATH="/opt/jpacf/src"

RUN mkdir -p /workspace/state/{hf,uv,xdg,torch,home,artifacts,generated,results,dist,vendor}

WORKDIR /opt/jpacf

ENTRYPOINT ["bash", "/opt/jpacf/scripts/container/inside.sh"]
CMD ["python", "/opt/jpacf/scripts/container/verify_runtime.py", "--require-gpu"]
