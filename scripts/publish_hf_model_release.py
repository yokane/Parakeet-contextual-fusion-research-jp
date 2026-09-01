#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a versioned J-PACF-YOMI-TDT release bundle to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", default="saeeew/J-PACF-YOMI-tdt")
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--commit-message")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    manifest_path = args.release_dir / "release_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"release manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("release")) != args.release:
        raise SystemExit(
            f"manifest release mismatch: expected {args.release!r}, got {manifest.get('release')!r}"
        )
    if str(manifest.get("model_id")) != args.repo_id:
        raise SystemExit(
            f"manifest model_id mismatch: expected {args.repo_id!r}, got {manifest.get('model_id')!r}"
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.release_dir),
        path_in_repo=f"artifacts/{args.release}",
        commit_message=(
            args.commit_message or f"Publish J-PACF-YOMI-TDT release {args.release}"
        ),
    )
    print(commit)


if __name__ == "__main__":
    main()
