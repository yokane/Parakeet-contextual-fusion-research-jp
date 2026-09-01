#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def copy_artifact(
    source: Path | None,
    destination: Path,
    *,
    role: str,
    required_suffixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if source is None:
        return None
    if not source.is_file():
        raise SystemExit(f"{role} does not exist: {source}")
    if required_suffixes and source.suffix.lower() not in required_suffixes:
        raise SystemExit(
            f"{role} must use one of {required_suffixes}, got {source.suffix!r}: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "role": role,
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_name": source.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible J-PACF-YOMI-TDT Hugging Face release directory"
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("dist/hf-model"))
    parser.add_argument("--fusion-config", type=Path, default=Path("hf_model/fusion_config.yaml"))
    parser.add_argument(
        "--research-manifest", type=Path, default=Path("hf_model/research_manifest.json")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phone-head", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--saturation", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()

    if not args.release or "/" in args.release or "\\" in args.release or args.release in {".", ".."}:
        raise SystemExit("--release must be a single path-safe name")

    release_dir = args.output_root / args.release
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    files: list[dict[str, Any]] = []
    for source, name, role, suffixes in [
        (args.fusion_config, "fusion_config.yaml", "fusion_config", (".yaml", ".yml")),
        (args.research_manifest, "research_manifest.json", "research_manifest", (".json",)),
        (args.checkpoint, "J-PACF-YOMI-TDT.nemo", "nemo_checkpoint", (".nemo",)),
        (args.phone_head, "phone_head.pt", "phone_head", (".pt", ".pth")),
        (args.metrics, "metrics.json", "metrics", (".json",)),
        (args.saturation, "saturation.json", "saturation_analysis", (".json",)),
        (args.environment, "environment.json", "runtime_environment", (".json",)),
        (args.notes, "RELEASE_NOTES.md", "release_notes", (".md",)),
    ]:
        entry = copy_artifact(source, release_dir / name, role=role, required_suffixes=suffixes)
        if entry is not None:
            files.append(entry)

    manifest = {
        "schema_version": 1,
        "model_id": "saeeew/J-PACF-YOMI-tdt",
        "model_family": "J-PACF-YOMI-TDT",
        "release": args.release,
        "created_at": datetime.now(UTC).isoformat(),
        "source_git_sha": git_sha(),
        "base_model": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "base_model_license": "cc-by-4.0",
        "benchmark": "saeeew/JP-HomophoneBench",
        "contains_standalone_checkpoint": args.checkpoint is not None,
        "contains_phone_head": args.phone_head is not None,
        "release_kind": (
            "checkpoint" if args.checkpoint is not None else "scorer-or-config"
        ),
        "files": sorted(files, key=lambda entry: str(entry["role"])),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"release_dir={release_dir}")


if __name__ == "__main__":
    main()
