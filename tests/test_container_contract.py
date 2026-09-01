from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_bakes_gpu_environment_outside_source_mount() -> None:
    dockerfile = read("Dockerfile")
    assert "WORKDIR /opt/jpacf" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/jpacf/.venv" in dockerfile
    assert "--extra gpu" in dockerfile
    assert "--extra dev" in dockerfile
    assert "--extra research" not in dockerfile
    assert "JPA_CF_STATE_ROOT=/workspace/state" in dockerfile


def test_host_runner_uses_single_state_mount_and_digest_contract() -> None:
    runner = read("scripts/container/run.sh")
    assert "@sha256:" in runner
    assert '--gpus all' in runner
    assert '-v "$STATE_ROOT:/workspace/state"' in runner
    assert "mise run" not in runner
    assert "uv sync" not in runner


def test_container_preparation_does_not_resync_gpu_environment() -> None:
    preparation = read("scripts/research/prepare_e00_e04.sh")
    assert 'JPA_CF_CONTAINER_RUNTIME:-0' in preparation
    assert "python /opt/jpacf/scripts/container/verify_runtime.py --require-gpu" in preparation
    assert "uv run --locked --no-sync" in preparation


def test_staged_gpu_action_executes_through_container_runner() -> None:
    workflow = read(".github/workflows/e00-e06-staged-gpu.yml")
    assert "scripts/container/resolve-image.sh" in workflow
    assert "scripts/container/run.sh" in workflow
    assert "mise-action" not in workflow
    assert "deps:sync-gpu" not in workflow


def test_portable_cache_is_separated_from_immutable_run_evidence() -> None:
    storage = read("configs/hf-storage.json")
    assert '"runtime": "runtime"' in storage
    assert '"workspace_cache": "workspace-cache"' in storage
    assert '"runtime/"' in storage
    assert '"workspace-cache/"' in storage
    assert '"runs/"' in storage
