#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def repo_info(api: HfApi, repo_id: str, repo_type: str) -> Any:
    if repo_type == "model":
        return api.model_info(repo_id=repo_id)
    if repo_type == "dataset":
        return api.dataset_info(repo_id=repo_id)
    raise SystemExit(f"unsupported Hugging Face repo_type: {repo_type}")


def file_lock(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
) -> dict[str, Any]:
    paths = api.get_paths_info(
        repo_id=repo_id,
        paths=[filename],
        repo_type=None if repo_type == "model" else repo_type,
        revision=revision,
    )
    if len(paths) != 1 or getattr(paths[0], "path", None) != filename:
        raise SystemExit(f"could not resolve required Hugging Face file {repo_id}/{filename}")
    item = paths[0]
    lfs = getattr(item, "lfs", None)
    sha256 = str(getattr(lfs, "sha256", "") or "") if lfs is not None else ""
    if not HEX64.fullmatch(sha256):
        raise SystemExit(
            f"required file {repo_id}/{filename} is not exposed with an LFS SHA-256; "
            "the research model contract requires a content hash"
        )
    return {
        "path": filename,
        "sha256": sha256,
        "size": int(getattr(item, "size", 0)),
        "blob_id": str(getattr(item, "blob_id", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve configured Hugging Face repos and required files to immutable identities"
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "locks/upstream-sources.json")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    configured = sources["hugging_face"]
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    repositories: dict[str, dict[str, Any]] = {}
    for logical_name, entry in sorted(configured.items()):
        repo_id = str(entry["repo_id"])
        repo_type = str(entry["repo_type"])
        info = repo_info(api, repo_id, repo_type)
        revision = str(info.sha or "")
        if not HEX40.fullmatch(revision):
            raise SystemExit(f"could not resolve full revision for {repo_id}: {revision!r}")
        resolved: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
        }
        required_file = str(entry.get("required_file") or "").strip()
        if required_file:
            resolved["required_files"] = [
                file_lock(
                    api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    filename=required_file,
                )
            ]
        repositories[logical_name] = resolved

    payload = {"schema_version": 2, "repositories": repositories}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
