from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
DOCKERFILE = ROOT / ".devcontainer" / "Dockerfile"
POST_CREATE = ROOT / ".devcontainer" / "post-create.sh"
POST_START = ROOT / ".devcontainer" / "post-start.sh"
README = ROOT / ".devcontainer" / "README.md"


def test_devcontainer_is_cpu_static_and_uses_locked_mise_contract() -> None:
    config = json.loads(DEVCONTAINER.read_text(encoding="utf-8"))
    assert config["remoteUser"] == "vscode"
    assert config["build"]["args"]["MISE_VERSION"] == "2026.8.16"
    assert config["postCreateCommand"] == "bash .devcontainer/post-create.sh"
    assert config["postStartCommand"] == "bash .devcontainer/post-start.sh"
    assert config["containerEnv"]["JPA_CF_STATE_ROOT"] == "/workspace/state"
    assert not any("HF_TOKEN" in key for key in config.get("containerEnv", {}))


def test_devcontainer_does_not_duplicate_the_gpu_runtime() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    post_create = POST_CREATE.read_text(encoding="utf-8")
    assert "FROM --platform=linux/amd64" in dockerfile
    assert "MISE_VERSION=2026.8.16" in dockerfile
    assert "nemo-toolkit" not in dockerfile
    assert "torch==" not in dockerfile
    assert "--extra gpu" not in post_create
    assert "mise --locked install" in post_create
    assert "mise run deps:sync" in post_create
    assert "mise run hf:transport:sync" in post_create


def test_devcontainer_persists_caches_and_research_state_outside_source() -> None:
    config = json.loads(DEVCONTAINER.read_text(encoding="utf-8"))
    mounts = "\n".join(config["mounts"])
    assert "jpacf-dev-mise" in mounts
    assert "jpacf-dev-cache" in mounts
    assert "jpacf-dev-state" in mounts
    assert "target=/workspace/state" in mounts
    assert "/workspace/state" in POST_START.read_text(encoding="utf-8")


def test_devcontainer_docs_preserve_canonical_gpu_runtime_boundary() -> None:
    text = README.read_text(encoding="utf-8")
    assert "mise run ci" in text
    assert "ghcr.io/yokane/jpacf-yomi-tdt-runtime" in text
    assert "runtime-current" in text
    assert "exact digest" in text
    assert "HF_TOKEN" in text
