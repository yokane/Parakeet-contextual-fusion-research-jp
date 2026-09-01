from __future__ import annotations

from pathlib import Path


def dockerfile_text() -> str:
    return Path("Dockerfile").read_text(encoding="utf-8")


def test_managed_python_lives_outside_runtime_state_mount() -> None:
    text = dockerfile_text()
    assert "UV_PYTHON_INSTALL_DIR=/opt/jpacf/.uv-python" in text
    assert "UV_PYTHON_INSTALL_BIN=0" in text
    assert "UV_PYTHON_INSTALL_DIR=/workspace/state" not in text
    assert "project Python escaped immutable image state" in text
    assert "/opt/jpacf/.uv-python/*" in text


def test_runtime_state_environment_is_declared_after_project_sync() -> None:
    text = dockerfile_text()
    final_sync = text.rindex("uv sync")
    runtime_home = text.index("ENV HOME=/workspace/state/home")
    assert final_sync < runtime_home


def test_uv_build_cache_mount_matches_uv_cache_directory() -> None:
    text = dockerfile_text()
    assert "--mount=type=cache,target=/cache/uv" not in text
    assert text.count("--mount=type=cache,target=/root/.cache/uv") >= 4
    assert text.count("UV_CACHE_DIR=/root/.cache/uv") >= 4


def test_runtime_cache_remains_on_writable_state_volume() -> None:
    text = dockerfile_text()
    assert "UV_CACHE_DIR=/workspace/state/uv" in text
    assert "HF_HOME=/workspace/state/hf" in text
    assert "TORCH_HOME=/workspace/state/torch" in text
