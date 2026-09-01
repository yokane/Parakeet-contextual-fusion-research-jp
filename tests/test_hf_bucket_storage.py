from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def load_script(name: str):
    path = Path("scripts/hf") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_sequence_id_uses_highest_canonical_or_nested_id() -> None:
    module = load_script("next_sequence_id")
    listing = [
        "candidates/candidate-000001/README.md",
        "candidates/candidate-000009/metadata.json",
        "candidates/legacy/candidate-000012/metadata.json",
        "runs/gh-123/metrics.json",
    ]
    assert module.next_sequence_id("candidate", listing) == "candidate-000013"


def test_next_sequence_id_rejects_unsafe_prefix() -> None:
    module = load_script("next_sequence_id")
    with pytest.raises(ValueError, match="invalid allocation prefix"):
        module.next_sequence_id("../candidate", [])


def test_upload_only_sync_plan_is_accepted(tmp_path: Path) -> None:
    module = load_script("validate_sync_plan")
    destination = "hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/candidates/candidate-000001"
    rows = [
        {
            "type": "header",
            "source": "/tmp/candidate",
            "dest": destination,
            "timestamp": "2026-09-01T00:00:00Z",
            "summary": {"upload": 2},
        },
        {"type": "operation", "action": "upload", "path": "metadata.json", "reason": "new"},
        {
            "type": "operation",
            "action": "upload",
            "path": "release_manifest.json",
            "reason": "new",
        },
    ]
    plan = tmp_path / "plan.jsonl"
    plan.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert module.validate_fresh_upload_plan(module.load_jsonl(plan), destination) == 2


@pytest.mark.parametrize("action", ["delete", "download", "skip"])
def test_sync_plan_rejects_non_upload_operations(action: str) -> None:
    module = load_script("validate_sync_plan")
    rows = [
        {
            "type": "header",
            "source": "/tmp/source",
            "dest": "hf://buckets/x/y/candidates/candidate-000001",
        },
        {"type": "operation", "action": action, "path": "old.bin", "reason": "mutation"},
    ]
    with pytest.raises(ValueError, match="only upload operations"):
        module.validate_fresh_upload_plan(rows)


def test_sync_plan_rejects_wrong_destination() -> None:
    module = load_script("validate_sync_plan")
    rows = [
        {
            "type": "header",
            "source": "/tmp/source",
            "dest": "hf://buckets/x/y/candidates/candidate-000001",
        },
        {"type": "operation", "action": "upload", "path": "metadata.json"},
    ]
    with pytest.raises(ValueError, match="unexpected sync destination"):
        module.validate_fresh_upload_plan(
            rows, "hf://buckets/x/y/candidates/candidate-000002"
        )


def canonical_bucket_listing() -> list[str]:
    return [
        "README.md",
        "config/README.md",
        "config/current.json",
        "candidates/README.md",
        "experiments/README.md",
        "runs/README.md",
        "benchmarks/README.md",
        "reference/README.md",
        "runtime/README.md",
        "scripts/README.md",
        "tmp/README.md",
        "workspace-cache/README.md",
        "candidates/candidate-000001/README.md",
    ]


def storage_config() -> dict[str, object]:
    return json.loads(Path("configs/hf-storage.json").read_text(encoding="utf-8"))


def test_bucket_layout_accepts_canonical_tree() -> None:
    module = load_script("validate_bucket_layout")
    report = module.validate_layout(canonical_bucket_listing(), storage_config())
    assert report["status"] == "ok"
    assert report["bucket"] == "saeeew/J-PACF-YOMI-tdt-bucket"


def test_bucket_layout_rejects_missing_root() -> None:
    module = load_script("validate_bucket_layout")
    listing = [path for path in canonical_bucket_listing() if path != "runs/README.md"]
    with pytest.raises(ValueError, match="runs"):
        module.validate_layout(listing, storage_config())


def test_dockerfile_caches_dependencies_before_copying_source() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")
    metadata_copy = text.index("COPY pyproject.toml uv.lock ./")
    dependency_install = text.index("--no-install-project")
    source_copy = text.index("COPY src ./src")
    project_install = text.rindex("uv sync")

    assert metadata_copy < dependency_install < source_copy < project_install
    assert "--mount=type=cache,target=/root/.cache/uv" in text
    assert "--mount=type=cache,target=/var/cache/apt,sharing=locked" in text
    assert "--mount=type=cache,target=/var/lib/apt,sharing=locked" in text


def test_ghcr_build_uses_runner_selected_buildkit_cache_contract() -> None:
    text = Path(".github/workflows/ghcr-runtime.yml").read_text(encoding="utf-8")
    assert "run: bash scripts/ci/setup_actions_cache.sh" in text
    assert "cache-from: ${{ env.BUILDKIT_CACHE_FROM }}" in text
    assert "cache-to: ${{ env.BUILDKIT_CACHE_TO }}" in text
    assert "if: ${{ env.SELF_CACHE_PERSISTENT == 'true' }}" in text
    assert "mv \"${BUILDKIT_CACHE_DIR}-next\" \"${BUILDKIT_CACHE_DIR}\"" in text
    assert "tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}" in text


def test_candidate_roundtrip_detects_tampering(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    candidate_dir = tmp_path / "candidate"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_hf_model_release.py",
            "--release",
            "candidate-test",
            "--output-root",
            str(release_root),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/hf/prepare_candidate.py",
            "--release-dir",
            str(release_root / "candidate-test"),
            "--output-dir",
            str(candidate_dir),
        ],
        check=True,
    )
    valid = subprocess.run(
        [sys.executable, "scripts/hf/validate_candidate.py", str(candidate_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    assert (candidate_dir / "README.md").is_file()

    with (candidate_dir / "fusion_config.yaml").open("a", encoding="utf-8") as handle:
        handle.write("\n# tampered\n")
    tampered = subprocess.run(
        [sys.executable, "scripts/hf/validate_candidate.py", str(candidate_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "SHA-256 mismatch" in tampered.stderr
