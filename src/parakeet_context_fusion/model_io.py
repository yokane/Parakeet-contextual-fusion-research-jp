from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LockedModelIdentity:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    size: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_model_identity(
    lock_path: Path = Path("locks/hf-revisions.lock.json"),
    *,
    logical_name: str = "base_model",
    required_revision: str | None = None,
) -> LockedModelIdentity:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    repositories = payload.get("repositories") or {}
    entry = repositories.get(logical_name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"missing Hugging Face lock entry: {logical_name}")
    repo_id = str(entry.get("repo_id") or "")
    revision = str(entry.get("revision") or "")
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"invalid locked Hugging Face revision for {logical_name}: {revision!r}")
    if required_revision is not None and required_revision != revision:
        raise RuntimeError(
            f"requested model revision {required_revision!r} does not match image/repository lock {revision!r}"
        )
    files = entry.get("required_files") or []
    if len(files) != 1 or not isinstance(files[0], dict):
        raise RuntimeError(f"{logical_name} must lock exactly one required model file")
    file_entry = files[0]
    filename = str(file_entry.get("path") or "")
    sha256 = str(file_entry.get("sha256") or "")
    size = int(file_entry.get("size") or 0)
    if not filename:
        raise RuntimeError(f"missing required model filename for {logical_name}")
    if not HEX64.fullmatch(sha256):
        raise RuntimeError(f"invalid locked model SHA-256 for {logical_name}")
    if size <= 0:
        raise RuntimeError(f"invalid locked model size for {logical_name}")
    return LockedModelIdentity(repo_id, revision, filename, sha256, size)


def materialize_locked_model(
    identity: LockedModelIdentity,
    *,
    cache_dir: Path | None = None,
) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=identity.repo_id,
            filename=identity.filename,
            revision=identity.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    ).resolve()
    size = path.stat().st_size
    if size != identity.size:
        raise RuntimeError(
            f"locked model size mismatch for {identity.repo_id}: {size} != {identity.size}"
        )
    digest = sha256_file(path)
    if digest != identity.sha256:
        raise RuntimeError(
            f"locked model SHA-256 mismatch for {identity.repo_id}: {digest} != {identity.sha256}"
        )
    return path


def restore_locked_asr_model(
    *,
    lock_path: Path = Path("locks/hf-revisions.lock.json"),
    required_revision: str | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> Any:
    identity = load_locked_model_identity(lock_path, required_revision=required_revision)
    path = materialize_locked_model(identity, cache_dir=cache_dir)
    import nemo.collections.asr as nemo_asr
    import torch

    model = nemo_asr.models.ASRModel.restore_from(str(path))
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(resolved_device)
