from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "research" / "e00-e06-artifacts.yaml"
PHASE_DOCKERFILE = ROOT / "docker" / "phases" / "Dockerfile"
E02_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.e02"
KENLM_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.kenlm"
PHONE_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.e05-phone-cpu"
BUILD_SCRIPT = ROOT / "scripts" / "ci" / "build_research_images.sh"
CPU_WORKFLOW = ROOT / ".github" / "workflows" / "research-artifacts-cpu.yml"
VAST_WORKFLOW = ROOT / ".github" / "workflows" / "research-phase-vast.yml"
IMAGE_WORKFLOW = ROOT / ".github" / "workflows" / "research-images.yml"
DOC = ROOT / "docs" / "e00-e06-research-artifacts.md"


def test_artifact_contract_covers_every_phase_and_executor() -> None:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    phases = payload["phases"]
    assert list(phases) == [f"E{i:02d}" for i in range(7)]
    assert phases["E00"]["executor"] == "vast"
    assert phases["E01"]["executor"] == "vast"
    assert phases["E02"]["executor"] == "vast"
    assert phases["E03"]["executor"] == "vast"
    assert phases["E04"]["executor"] == "vast"
    assert phases["E05"]["executor"] == "split"
    assert phases["E06"]["executor"] == "vast"
    tasks = payload["preparation_tasks"]
    assert tasks["common-hosted"]["executor"] == "github-hosted"
    assert tasks["e02-estimate-hosted"]["executor"] == "github-hosted"
    assert tasks["e05-phone-hosted"]["executor"] == "github-hosted"
    assert tasks["e02-encode-vast"]["executor"] == "vast"
    assert tasks["e02-pack-vast"]["executor"] == "vast"
    assert tasks["e05-extract-vast"]["executor"] == "vast"


def test_generic_phase_images_are_thin_runtime_overlays() -> None:
    text = PHASE_DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${RUNTIME_IMAGE} AS phase-base" in text
    for token in ("apt-get install", "pip install", "uv sync", "nemo-toolkit", "torch=="):
        assert token not in text
    for phase in range(7):
        assert f"FROM phase-base AS e{phase:02d}" in text


def test_e02_has_dedicated_kenlm_overlay_without_compiler_toolchain() -> None:
    text = E02_DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${RUNTIME_IMAGE} AS e02" in text
    assert "FROM ${KENLM_TOOLS_IMAGE} AS kenlm-tools" in text
    assert "COPY --from=kenlm-tools /opt/kenlm/bin" in text
    assert "ngram_lm_pipeline.py" in text
    assert "apt-get" not in text
    assert "cmake" not in text
    assert "git clone" not in text


def test_kenlm_tools_are_revision_pinned_and_multistage() -> None:
    text = KENLM_DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG KENLM_REVISION=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5" in text
    assert "FROM ${DEBIAN_IMAGE} AS build" in text
    assert "FROM ${DEBIAN_IMAGE} AS kenlm-tools" in text
    assert "--target lmplz build_binary" in text
    assert "COPY --from=build /opt/kenlm /opt/kenlm" in text


def test_e05_cpu_image_has_no_cuda_or_nemo_dependency() -> None:
    text = PHONE_DOCKERFILE.read_text(encoding="utf-8")
    assert "python:3.12.3-slim-bookworm" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "e05_train_rerank_cpu.sh" in text
    assert "nemo-toolkit" not in text
    assert "cu132" not in text
    assert "cuda" not in text.lower()


def test_build_script_uses_registry_as_cache_and_dockerhub_only_as_fallback() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "buildx imagetools inspect" in text
    assert "--push" in text
    assert "--load" not in text
    assert "type=gha" not in text
    assert "DOCKERHUB_ACCESS_TOKEN" not in text
    assert "DOCKERHUB_REPOSITORY" in text
    assert "GHCR push failed" in text
    assert "resolve-remote-image.sh" in text


def test_cpu_and_gpu_workflows_keep_compute_boundary_explicit() -> None:
    cpu = CPU_WORKFLOW.read_text(encoding="utf-8")
    vast = VAST_WORKFLOW.read_text(encoding="utf-8")
    images = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in cpu
    assert "vastai create instance" not in cpu
    assert "options: [common, e02-estimate, e05-phone]" in cpu
    assert "vastai create instance" in vast
    assert "vastai destroy instance" in vast
    assert "if: ${{ always() }}" in vast
    assert "docker-container" in images
    assert "DOCKERHUB_ACCESS_TOKEN" in images
    assert "DOCKERHUB_REPOSITORY" in images
    assert "type=gha" not in images


def test_research_doc_names_all_phases_and_storage_planes() -> None:
    text = DOC.read_text(encoding="utf-8")
    for phase in range(7):
        assert f"### E{phase:02d}" in text
    assert "workspace-cache/e00-e06/<research-key>/" in text
    assert "DOCKERHUB_ACCESS_TOKEN" in text
    assert "DOCKERHUB_REPOSITORY" in text
    assert "GitHub-hosted" in text
    assert "Vast" in text
