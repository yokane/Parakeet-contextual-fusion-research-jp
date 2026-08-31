ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libsndfile1 \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/parakeet-context-fusion
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY configs ./configs

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e '.[nemo]'

CMD ["python", "-m", "parakeet_context_fusion.cli", "--help"]
