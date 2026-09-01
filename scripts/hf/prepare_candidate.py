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
CANDIDATE_GENERATED_FILES = frozenset({"metadata.json", "README.md"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_filenames(manifest: dict[str, Any]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release manifest must contain a non-empty files list")
    filenames: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("release manifest file entry must be an object")
        filename = str(entry.get("filename") or "")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"unsafe release filename: {filename!r}")
        if filename in filenames:
            raise ValueError(f"duplicate release filename: {filename!r}")
        filenames.add(filename)
    return filenames


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
    for filename in manifest_filenames(manifest):
        artifact = release_dir / filename
        if not artifact.is_file():
            raise ValueError(f"release artifact is missing: {artifact}")
        entry = next(item for item in manifest["files"] if item.get("filename") == filename)
        expected = str(entry.get("sha256") or "")
        actual = sha256(artifact)
        if expected != actual:
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual} != {expected}")
    return manifest


def expected_candidate_files(manifest: dict[str, Any]) -> set[str]:
    return manifest_filenames(manifest) | {"release_manifest.json"} | set(CANDIDATE_GENERATED_FILES)


def reject_unexpected_candidate_files(candidate_dir: Path, manifest: dict[str, Any]) -> None:
    actual = {path.name for path in candidate_dir.iterdir() if path.is_file()}
    expected = expected_candidate_files(manifest)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        raise ValueError(f"unexpected candidate files: {unexpected}")
    if missing:
        raise ValueError(f"candidate files are missing: {missing}")
    non_files = sorted(path.name for path in candidate_dir.iterdir() if not path.is_file())
    if non_files:
        raise ValueError(f"unexpected candidate entries: {non_files}")


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

    for filename in sorted(manifest_filenames(manifest) | {"release_manifest.json"}):
        link_or_copy(release_dir / filename, output_dir / filename)

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
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {MODEL_FAMILY} candidate",
                "",
                "Immutable-by-policy development candidate for later validated promotion.",
                "",
                f"- Release: `{metadata['release']}`",
                f"- Kind: `{metadata['release_kind']}`",
                f"- Base model: `{BASE_MODEL}`",
                f"- Source Git SHA: `{metadata['source_git_sha']}`",
                f"- Release manifest SHA-256: `{metadata['release_manifest_sha256']}`",
                "",
                "The canonical accepted release, if promoted, is published under",
                f"`{MODEL_ID}/artifacts/{metadata['release']}/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reject_unexpected_candidate_files(output_dir, manifest)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a validated J-PACF release as an HF Bucket candidate"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = prepare_candidate(args.release_dir, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
