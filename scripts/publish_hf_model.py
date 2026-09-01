#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish J-PACF-YOMI-TDT model metadata/configs to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", default="saeeew/J-PACF-YOMI-tdt")
    parser.add_argument("--folder", type=Path, default=Path("hf_model"))
    parser.add_argument("--commit-message", default="Update J-PACF-YOMI-TDT research scaffold")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if not args.folder.is_dir():
        raise SystemExit(f"model folder does not exist: {args.folder}")
    required = [args.folder / "README.md", args.folder / "fusion_config.yaml", args.folder / "research_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"required model scaffold files are missing: {missing}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.folder),
        commit_message=args.commit_message,
    )
    print(f"Published {args.folder} to {args.repo_id}")
    print(commit)


if __name__ == "__main__":
    main()
