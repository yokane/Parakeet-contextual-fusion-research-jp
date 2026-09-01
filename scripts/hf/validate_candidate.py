#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_candidate import (
    BASE_MODEL,
    MODEL_FAMILY,
    MODEL_ID,
    load_release_manifest,
    reject_unexpected_candidate_files,
    sha256,
)


def validate_candidate(candidate_dir: Path) -> dict[str, object]:
    metadata_path = candidate_dir / "metadata.json"
    readme_path = candidate_dir / "README.md"
    if not metadata_path.is_file():
        raise ValueError(f"metadata.json is missing: {metadata_path}")
    if not readme_path.is_file() or readme_path.stat().st_size == 0:
        raise ValueError(f"candidate README.md is missing or empty: {readme_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected model_id: {metadata.get('model_id')!r}")
    if metadata.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"unexpected model_family: {metadata.get('model_family')!r}")
    if metadata.get("base_model") != BASE_MODEL:
        raise ValueError(f"unexpected base_model: {metadata.get('base_model')!r}")

    manifest = load_release_manifest(candidate_dir)
    reject_unexpected_candidate_files(candidate_dir, manifest)
    manifest_hash = sha256(candidate_dir / "release_manifest.json")
    if metadata.get("release_manifest_sha256") != manifest_hash:
        raise ValueError("metadata.json does not match release_manifest.json SHA-256")
    if metadata.get("release") != manifest.get("release"):
        raise ValueError("candidate metadata release does not match release manifest")

    return {
        "model_id": MODEL_ID,
        "release": manifest.get("release"),
        "release_kind": manifest.get("release_kind"),
        "source_git_sha": manifest.get("source_git_sha"),
        "release_manifest_sha256": manifest_hash,
        "files": len(manifest.get("files") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a J-PACF HF Bucket candidate")
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args()
    summary = validate_candidate(args.candidate_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
