#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve configured Hugging Face repos to full commit revisions")
    parser.add_argument("--sources", type=Path, default=ROOT / "locks/upstream-sources.json")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    configured = sources["hugging_face"]
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    repositories: dict[str, dict[str, str]] = {}
    for logical_name, entry in sorted(configured.items()):
        repo_id = str(entry["repo_id"])
        repo_type = str(entry["repo_type"])
        if repo_type == "model":
            info = api.model_info(repo_id=repo_id)
        elif repo_type == "dataset":
            info = api.dataset_info(repo_id=repo_id)
        else:
            raise SystemExit(f"unsupported Hugging Face repo_type: {repo_type}")
        revision = str(info.sha or "")
        if len(revision) != 40:
            raise SystemExit(f"could not resolve full revision for {repo_id}: {revision!r}")
        repositories[logical_name] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
        }

    payload = {"schema_version": 1, "repositories": repositories}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
