from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "phases" / "Dockerfile"
RUNNER = ROOT / "scripts" / "container" / "run-phase.sh"
CLEANUP = ROOT / ".github" / "workflows" / "actions-cache-cleanup.yml"


def test_phase_dockerfile_has_all_named_targets() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${RUNTIME_IMAGE} AS phase-base" in text
    for index in range(7):
        phase = f"E{index:02d}"
        target = phase.lower()
        assert f"FROM phase-base AS {target}" in text
        assert f'ENV JPA_CF_PHASE={phase}' in text
        assert f'"{phase}"]' in text


def test_phase_dockerfile_does_not_reinstall_heavy_dependencies() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    forbidden = ("apt-get install", "uv sync", "pip install", "nemo-toolkit", "torch==")
    for token in forbidden:
        assert token not in text


def test_phase_runner_covers_all_phases_and_keeps_e06_explicit() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for index in range(7):
        assert f"E{index:02d})" in text
    assert "E05_PREPARE" in text
    assert "E06_DRIVER:?" in text
    assert "--locked --no-sync" in text


def test_actions_cache_cleanup_is_write_scoped_and_selective() -> None:
    text = CLEANUP.read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "last_accessed_at" in text
    assert "refs/pull/" in text
    assert "/actions/caches/${id}" in text
    assert "workflow_dispatch:" in text
