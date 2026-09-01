#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

MODEL_ID = "saeeew/J-PACF-YOMI-tdt"
MODEL_FAMILY = "J-PACF-YOMI-TDT"
BASE_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_manifest(release_dir: Path) -> dict[str, Any]:
    path = release_dir / "release_manifest.json"
    if not path.is_file():
        raise ValueError(f"release_manifest.json is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected model_id: {manifest.get('model_id')!r}")
    if manifest.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"unexpected model_family: {manifest.get('model_family')!r}")
    if manifest.get("base_model") != BASE_MODEL:
        raise ValueError(f"unexpected base_model: {manifest.get('base_model')!r}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release manifest must contain a non-empty files list")
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("release manifest file entry must be an object")
        filename = str(entry.get("filename") or "")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"unsafe release filename: {filename!r}")
        artifact = release_dir / filename
        if not artifact.is_file():
            raise ValueError(f"release artifact is missing: {artifact}")
        expected = str(entry.get("sha256") or "")
        actual = sha256(artifact)
        if expected != actual:
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual} != {expected}")
    return manifest


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_candidate(release_dir: Path, output_dir: Path) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    manifest = load_release_manifest(release_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for source in release_dir.iterdir():
        if source.is_file():
            link_or_copy(source, output_dir / source.name)

    release_manifest = output_dir / "release_manifest.json"
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "base_model": BASE_MODEL,
        "release": manifest.get("release"),
        "release_kind": manifest.get("release_kind"),
        "source_git_sha": manifest.get("source_git_sha"),
        "contains_standalone_checkpoint": bool(manifest.get("contains_standalone_checkpoint")),
        "contains_phone_head": bool(manifest.get("contains_phone_head")),
        "release_manifest_sha256": sha256(release_manifest),
        "candidate_identity": "allocated-by-hf-bucket",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a validated J-PACF release as an HF Bucket candidate")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = prepare_candidate(args.release_dir, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
