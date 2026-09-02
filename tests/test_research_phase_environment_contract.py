from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "research" / "e00-e06-artifacts.yaml"
PHASE_DOCKERFILE = ROOT / "docker" / "phases" / "Dockerfile"
E02_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.e02"
KENLM_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.kenlm"
PHONE_DOCKERFILE = ROOT / "docker" / "research" / "Dockerfile.e05-phone-cpu"
BUILD_SCRIPT = ROOT / "scripts" / "ci" / "build_research_images.sh"
SNAPSHOT_HELPER = ROOT / "scripts" / "hf" / "hf-research-snapshot.sh"
SNAPSHOT_PLAN = ROOT / "scripts" / "research" / "snapshot_plan.py"
STAGE_FINGERPRINTS = ROOT / "scripts" / "research" / "stage_fingerprints.py"
CPU_WORKFLOW = ROOT / ".github" / "workflows" / "research-artifacts-cpu.yml"
VAST_WORKFLOW = ROOT / ".github" / "workflows" / "research-phase-vast.yml"
IMAGE_WORKFLOW = ROOT / ".github" / "workflows" / "research-images.yml"
DOC = ROOT / "docs" / "e00-e06-research-artifacts.md"
FINGERPRINT_DOC = ROOT / "docs" / "research-snapshot-fingerprints.md"


def test_artifact_contract_covers_every_phase_and_executor() -> None:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    phases = payload["phases"]
    assert list(phases) == [f"E{i:02d}" for i in range(7)]
    for phase in ("E00", "E01", "E02", "E03", "E04", "E06"):
        assert phases[phase]["executor"] == "vast"
    assert phases["E05"]["executor"] == "github-hosted"
    tasks = payload["preparation_tasks"]
    assert tasks["common-hosted"]["executor"] == "github-hosted"
    assert tasks["e02-estimate-hosted"]["executor"] == "github-hosted"
    assert tasks["e05-phone-hosted"]["executor"] == "github-hosted"
    assert tasks["e02-encode-vast"]["executor"] == "vast"
    assert tasks["e02-pack-vast"]["executor"] == "vast"
    assert tasks["e05-extract-vast"]["executor"] == "vast"


def test_snapshot_lineage_is_content_addressed_and_delta_based() -> None:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["bucket_prefix"] == "workspace-cache/e00-e06"
    snapshots = payload["snapshot_tasks"]
    assert snapshots["common"]["inputs"] == []
    assert snapshots["e02-estimate"]["inputs"] == ["e02-encode"]
    assert snapshots["e02-pack"]["inputs"] == ["e02-encode", "e02-estimate"]
    assert snapshots["e05-phone"]["inputs"] == ["common", "phase-e04", "e05-extract"]
    assert snapshots["E06"]["inputs"] == ["common", "e02-pack", "e05-extract", "e05-phone"]
    assert "generated/eval" not in snapshots["e02-pack"]["publish"]
    assert "artifacts/model" not in snapshots["e02-encode"]["publish"]
    for task, spec in snapshots.items():
        assert spec["fingerprint_sources"], task
    assert snapshots["E06"]["fingerprint_external"] == ["JPA_CF_E06_DRIVER_SHA256"]


def test_stage_fingerprint_invalidation_is_transitive_but_selective(tmp_path: Path) -> None:
    (tmp_path / "common.txt").write_text("common-v1\n", encoding="utf-8")
    (tmp_path / "phone.txt").write_text("phone-v1\n", encoding="utf-8")
    (tmp_path / "e06.txt").write_text("e06-v1\n", encoding="utf-8")
    config = {
        "schema_version": 3,
        "bucket_prefix": "workspace-cache/e00-e06",
        "snapshot_tasks": {
            "common": {
                "inputs": [],
                "output": "common",
                "publish": ["out/common"],
                "fingerprint_sources": ["common.txt"],
            },
            "e05-phone": {
                "inputs": ["common"],
                "output": "e05-phone",
                "publish": ["out/phone"],
                "fingerprint_sources": ["phone.txt"],
            },
            "E06": {
                "inputs": ["common", "e05-phone"],
                "output": "phase-e06",
                "publish": ["out/e06"],
                "fingerprint_sources": ["e06.txt"],
                "fingerprint_external": ["JPA_CF_E06_DRIVER_SHA256"],
            },
        },
    }
    config_path = tmp_path / "contract.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env["JPA_CF_E06_DRIVER_SHA256"] = "a" * 64

    def calculate() -> dict[str, str]:
        proc = subprocess.run(
            [
                sys.executable,
                str(STAGE_FINGERPRINTS),
                "--config",
                str(config_path),
                "--root",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(proc.stdout)

    first = calculate()
    (tmp_path / "phone.txt").write_text("phone-v2\n", encoding="utf-8")
    second = calculate()
    assert first["common"] == second["common"]
    assert first["e05-phone"] != second["e05-phone"]
    assert first["E06"] != second["E06"]

    (tmp_path / "common.txt").write_text("common-v2\n", encoding="utf-8")
    third = calculate()
    assert second["common"] != third["common"]
    assert second["e05-phone"] != third["e05-phone"]
    assert second["E06"] != third["E06"]


def test_snapshot_transport_rejects_overwrite_and_uses_fingerprint_refs() -> None:
    helper = SNAPSHOT_HELPER.read_text(encoding="utf-8")
    planner = SNAPSHOT_PLAN.read_text(encoding="utf-8")
    fingerprints = STAGE_FINGERPRINTS.read_text(encoding="utf-8")
    assert "immutable snapshot already exists" in helper
    assert "--plan" in helper and "--apply" in helper
    assert 'task_plan "$task" --field input_refs' in helper
    assert 'task_plan "$1" --field output_ref' in helper
    assert "--no-delete" in helper
    assert ".jpacf-snapshots" in helper
    assert '"fingerprint": fingerprint' in helper
    assert "trap '" not in helper or " RETURN" not in helper
    assert '"input_refs"' in planner and '"output_ref"' in planner
    assert '"fingerprint"' in planner
    assert "JPA_CF_STAGE_FINGERPRINTS_B64" in planner
    assert 'choices=[' in planner and '"input_refs"' in planner and '"output_ref"' in planner
    assert "jpacf-stage-fingerprint-v1" in fingerprints
    assert "fingerprint_external" in fingerprints


def test_generic_phase_images_are_thin_runtime_overlays() -> None:
    text = PHASE_DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${RUNTIME_IMAGE} AS phase-base" in text
    for token in ("apt-get install", "pip install", "uv sync", "nemo-toolkit", "torch=="):
        assert token not in text
    for phase in range(7):
        assert f"FROM phase-base AS e{phase:02d}" in text
    assert "hf-research-snapshot.sh" in text
    assert "snapshot_plan.py" in text
    assert "stage_fingerprints.py" in text


def test_e02_has_dedicated_kenlm_overlay_without_compiler_toolchain() -> None:
    text = E02_DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${RUNTIME_IMAGE} AS e02" in text
    assert "FROM ${KENLM_TOOLS_IMAGE} AS kenlm-tools" in text
    assert "COPY --from=kenlm-tools /opt/kenlm/bin" in text
    assert "ngram_lm_pipeline.py" in text
    assert "hf-research-snapshot.sh" in text
    assert "stage_fingerprints.py" in text
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


def test_cpu_and_gpu_workflows_keep_compute_and_image_identity_explicit() -> None:
    cpu = CPU_WORKFLOW.read_text(encoding="utf-8")
    vast = VAST_WORKFLOW.read_text(encoding="utf-8")
    images = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in cpu
    assert "vastai create instance" not in cpu
    assert "options: [common, e02-estimate, e05-phone]" in cpu
    assert "stage_fingerprints.py --field b64" in cpu
    assert "phone-e05-${GITHUB_SHA}" in cpu
    assert "kenlm-4cb443e60b7b" in cpu
    assert "vastai create instance" in vast
    assert "vastai destroy instance" in vast
    assert "stage_fingerprints.py --field b64" in vast
    assert "phase-${phase}-${GITHUB_SHA}" in vast
    assert "JPA_CF_STAGE_FINGERPRINTS_B64" in vast
    assert "E06_DRIVER_SHA256_INPUT" in vast
    assert "JPA_CF_E06_DRIVER_SHA256" in vast
    assert "E05" not in vast.split("options:", 1)[1].split("]", 1)[0]
    assert "DOCKERHUB_REPOSITORY" in vast
    assert "docker-container" in images
    assert "DOCKERHUB_ACCESS_TOKEN" in images
    assert "DOCKERHUB_REPOSITORY" in images
    assert "type=gha" not in images


def test_research_doc_names_all_phases_and_storage_planes() -> None:
    text = DOC.read_text(encoding="utf-8")
    for phase in range(7):
        assert f"E{phase:02d}" in text
    assert "workspace-cache/" in text
    assert "e00-e06/" in text
    assert "immutable delta snapshot" in text
    assert "DOCKERHUB_ACCESS_TOKEN" in text
    assert "DOCKERHUB_REPOSITORY" in text
    assert "GitHub-hosted" in text
    assert "Vast" in text

    fingerprint_text = FINGERPRINT_DOC.read_text(encoding="utf-8")
    assert "<stage-fingerprint>" in fingerprint_text
    assert "JPA_CF_E06_DRIVER_SHA256" in fingerprint_text
    assert "source-matched" in fingerprint_text
