from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
SETUP = ROOT / "scripts" / "codex" / "setup.sh"
MAINTENANCE = ROOT / "scripts" / "codex" / "maintenance.sh"
PREFLIGHT = ROOT / "scripts" / "codex" / "preflight.sh"
CHECK = ROOT / "scripts" / "codex" / "check.sh"
DOC = ROOT / "docs" / "codex-cloud.md"


def test_codex_cloud_contract_files_exist() -> None:
    for path in (AGENTS, SETUP, MAINTENANCE, PREFLIGHT, CHECK, DOC):
        assert path.is_file(), path


def test_setup_materializes_locked_cpu_environment_only() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "mise.run" in text
    assert "install --locked" in text
    assert "uv sync --locked --extra dev" in text
    assert "hf:transport:sync" in text
    assert "--extra gpu" not in text
    assert "HF_TOKEN" not in text


def test_maintenance_refreshes_cached_environment() -> None:
    text = MAINTENANCE.read_text(encoding="utf-8")
    assert "install --locked" in text
    assert "uv sync --locked --extra dev" in text
    assert "preflight.sh" in text


def test_preflight_enforces_exact_toolchain_and_cpu_platform() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert '"3.12.3"' in text
    assert '"0.12.1"' in text
    assert "verify_platform.py" in text
    assert "--require-gpu" not in text
    assert "git diff --check" in text


def test_check_uses_canonical_ci() -> None:
    text = CHECK.read_text(encoding="utf-8")
    assert 'run ci' in text
    assert "preflight.sh" in text
    assert "git diff --check" in text


def test_agents_keeps_gpu_and_hf_publish_outside_codex_cloud() -> None:
    text = AGENTS.read_text(encoding="utf-8")
    assert "Do not try to turn the Codex Cloud container into the authoritative GPU runtime" in text
    assert "must not require `HF_TOKEN`" in text
    assert "bash scripts/codex/check.sh" in text
