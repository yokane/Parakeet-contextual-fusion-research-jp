# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/yokane/Parakeet-contextual-fusion-research-jp

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.title="J-PACF-YOMI-TDT portable GPU runtime" \
      org.opencontainers.image.description="Thin source/runtime layer on the pinned CUDA13/NeMo dependency base"

# The dependency base already owns Python, uv, torch, CUDA user-space, NeMo,
# Boost and the isolated HF Bucket tooling. Only source-dependent layers belong
# here so CI never has to re-export the multi-gigabyte dependency rootfs.
WORKDIR /opt/jpacf

COPY pyproject.toml uv.lock ./
COPY tools/hf-bucket ./tools/hf-bucket
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

RUN set -euo pipefail; \
    test -x /opt/jpacf/.venv/bin/python; \
    resolved="$(readlink -f /opt/jpacf/.venv/bin/python)"; \
    case "$resolved" in \
      /opt/jpacf/.uv-python/*) ;; \
      *) echo "project Python escaped immutable image state: $resolved" >&2; exit 2 ;; \
    esac

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
