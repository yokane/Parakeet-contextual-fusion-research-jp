# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

# Keep package-manager downloads in BuildKit cache mounts. These mounts are not
# committed to the image layer but can be exported by buildx (GHA/GHCR cache).
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

# Dependency boundary: only dependency metadata is copied before installation.
# A source-only edit therefore does not invalidate the expensive NeMo/Python
# dependency layer.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
requirements = list(project.get("dependencies", []))
requirements.extend(project.get("optional-dependencies", {}).get("nemo", []))
if not requirements:
    raise SystemExit("pyproject.toml contains no runtime dependencies")
subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])
PY

# Source boundary: these layers change frequently and intentionally come after
# dependency installation.
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY configs ./configs
COPY experiments ./experiments
COPY hf_model ./hf_model

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps -e .

CMD ["python", "-m", "parakeet_context_fusion.cli", "--help"]
