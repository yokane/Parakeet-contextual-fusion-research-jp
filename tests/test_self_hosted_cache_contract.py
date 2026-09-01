from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/ci/setup_actions_cache.sh")


def run_setup(tmp_path: Path, *, root: str | None) -> tuple[subprocess.CompletedProcess[str], str, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    github_env = tmp_path / "github-env"
    github_output = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_REPOSITORY": "example/parakeet-context-fusion",
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_ENV": str(github_env),
            "GITHUB_OUTPUT": str(github_output),
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
        }
    )
    if root is None:
        env.pop("SELF_ACTIONS_CACHE_ROOT", None)
    else:
        env["SELF_ACTIONS_CACHE_ROOT"] = root
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    env_text = github_env.read_text(encoding="utf-8") if github_env.exists() else ""
    output_text = github_output.read_text(encoding="utf-8") if github_output.exists() else ""
    return result, env_text, output_text


def test_persistent_cache_uses_runner_supplied_repo_namespace(tmp_path: Path) -> None:
    root = tmp_path / "physical-cache"
    result, env_text, output_text = run_setup(tmp_path, root=str(root))

    assert result.returncode == 0, result.stderr
    namespace = root / "example" / "parakeet-context-fusion"
    assert "persistent=true\n" in output_text
    assert f"UV_CACHE_DIR={namespace / 'uv'}\n" in env_text
    assert f"HF_HUB_CACHE={namespace / 'huggingface' / 'hub'}\n" in env_text
    assert f"HF_XET_CACHE={namespace / 'huggingface' / 'xet'}\n" in env_text
    assert f"XDG_CACHE_HOME={namespace / 'xdg'}\n" in env_text
    assert f"MISE_DATA_DIR={namespace / 'mise' / 'data'}\n" in env_text
    assert f"BUILDKIT_CACHE_FROM=type=local,src={namespace / 'buildkit'}\n" in env_text
    assert "HF_HOME=" not in env_text
    assert "SELF_ACTIONS_CACHE_ROOT=" not in env_text


def test_relative_persistent_cache_root_is_rejected(tmp_path: Path) -> None:
    result, _, _ = run_setup(tmp_path, root="relative/cache")
    assert result.returncode == 2
    assert "absolute path" in result.stderr


def test_fallback_uses_workspace_and_remote_buildkit_cache(tmp_path: Path) -> None:
    result, env_text, output_text = run_setup(tmp_path, root=None)

    assert result.returncode == 0, result.stderr
    workspace = tmp_path / "workspace"
    assert "persistent=false\n" in output_text
    assert f"UV_CACHE_DIR={workspace / '.cache' / 'uv'}\n" in env_text
    assert f"HF_HUB_CACHE={workspace / '.cache' / 'huggingface' / 'hub'}\n" in env_text
    assert "BUILDKIT_CACHE_FROM=type=gha,scope=jpacf-Linux-X64\n" in env_text
    assert "MISE_DATA_DIR=" not in env_text
