# syntax=docker/dockerfile:1.7

# E00-E06 phase images are intentionally thin wrappers around one authoritative
# portable GPU runtime. Always prefer an immutable runtime digest here.
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS phase-base

ARG SOURCE_REPOSITORY=https://github.com/yokane/Parakeet-contextual-fusion-research-jp
ARG SOURCE_REVISION=unknown

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="J-PACF-YOMI-TDT E00-E06 phase runtime" \
      io.jpacf.runtime.kind="experiment-phase"

# The parent runtime already owns Python, CUDA, PyTorch, NeMo, uv, the project
# environment, and the experiment implementation. Only the phase dispatcher is
# overlaid here, so all targets share the same heavy root filesystem layers.
COPY scripts/container/run-phase.sh /opt/jpacf/scripts/container/run-phase.sh
RUN chmod 0755 /opt/jpacf/scripts/container/run-phase.sh

FROM phase-base AS e00
ENV JPA_CF_PHASE=E00
LABEL io.jpacf.phase="E00" io.jpacf.phase.description="TDT greedy baseline"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E00"]

FROM phase-base AS e01
ENV JPA_CF_PHASE=E01
LABEL io.jpacf.phase="E01" io.jpacf.phase.description="TDT MAES/beam decoding"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E01"]

FROM phase-base AS e02
ENV JPA_CF_PHASE=E02
LABEL io.jpacf.phase="E02" io.jpacf.phase.description="TDT beam plus N-gram GPU LM"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E02"]

FROM phase-base AS e03
ENV JPA_CF_PHASE=E03
LABEL io.jpacf.phase="E03" io.jpacf.phase.description="N-gram LM plus GPU phrase biasing"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E03"]

FROM phase-base AS e04
ENV JPA_CF_PHASE=E04
LABEL io.jpacf.phase="E04" io.jpacf.phase.description="Local hybrid CTC N-best reranking"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E04"]

FROM phase-base AS e05
ENV JPA_CF_PHASE=E05
LABEL io.jpacf.phase="E05" io.jpacf.phase.description="Frozen-encoder phoneme CTC reranking"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E05"]

FROM phase-base AS e06
ENV JPA_CF_PHASE=E06
LABEL io.jpacf.phase="E06" io.jpacf.phase.description="Version-isolated in-beam fusion"
CMD ["bash", "/opt/jpacf/scripts/container/run-phase.sh", "E06"]
