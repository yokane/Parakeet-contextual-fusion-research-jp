#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

from parakeet_context_fusion.benchmark import CORE_CATEGORIES

SPLIT_FILE_RE = re.compile(r"^(train|validation|test)(?:[-.].*)?\.parquet$")


def load_release(release_dir: Path) -> DatasetDict:
    files = {
        split: str(release_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
        if (release_dir / f"{split}.jsonl").exists()
    }
    if not files:
        raise FileNotFoundError(f"no split JSONL files under {release_dir}")
    return load_dataset("json", data_files=files)


def permitted_license(value: str, policy: str) -> bool:
    normalized = value.lower().replace("_", "-")
    if policy == "research":
        return True
    return (
        "-nc-" not in normalized
        and not normalized.endswith("-nc")
        and "noncommercial" not in normalized
    )


def filter_dataset(
    dataset: DatasetDict,
    *,
    license_policy: str,
    include_aux: bool,
) -> DatasetDict:
    def keep(row: dict[str, Any]) -> bool:
        category_ok = include_aux or row.get("category") in CORE_CATEGORIES
        license_value = str((row.get("source") or {}).get("license") or "unknown")
        return category_ok and permitted_license(license_value, license_policy)

    filtered = DatasetDict()
    for split, ds in dataset.items():
        value = ds.filter(keep, desc=f"filter {split}")
        if len(value):
            filtered[split] = value
    if not filtered:
        raise RuntimeError("publication filter removed every row")
    return filtered


def category_counts(dataset: DatasetDict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in dataset.values():
        for category in split["category"]:
            counts[str(category)] = counts.get(str(category), 0) + 1
    return dict(sorted(counts.items()))


def source_counts(dataset: DatasetDict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in dataset.values():
        for source in split["source"]:
            repo_id = str((source or {}).get("dataset", "unknown"))
            counts[repo_id] = counts.get(repo_id, 0) + 1
    return dict(sorted(counts.items()))


def config_name_from_parent(parent: str) -> str:
    if parent in {"", ".", "data"}:
        return "default"
    parts = PurePosixPath(parent).parts
    if len(parts) >= 2 and parts[0] == "data":
        return parts[-1]
    return parts[-1]


def discover_hub_configs(api: HfApi, repo_id: str) -> list[dict[str, Any]]:
    """Reconstruct README `configs` metadata from uploaded Parquet shards.

    DatasetDict.push_to_hub(config_name=...) writes the data shards first and also
    maintains README metadata. This publisher subsequently replaces README with a
    richer custom card, so we must preserve all named config/split mappings in the
    replacement front matter. Discovering from repo files also preserves configs
    published by earlier workflow runs.
    """
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        posix = PurePosixPath(path)
        if posix.suffix.lower() != ".parquet":
            continue
        match = SPLIT_FILE_RE.match(posix.name)
        if match is None:
            continue
        split = match.group(1)
        config_name = config_name_from_parent(posix.parent.as_posix())
        grouped[config_name][split].append(path)

    configs: list[dict[str, Any]] = []
    split_order = {"train": 0, "validation": 1, "test": 2}
    for config_name in sorted(grouped, key=lambda value: (value == "default", value)):
        data_files: list[dict[str, Any]] = []
        for split, paths in sorted(
            grouped[config_name].items(),
            key=lambda item: split_order.get(item[0], 99),
        ):
            ordered = sorted(paths)
            data_files.append(
                {
                    "split": split,
                    "path": ordered[0] if len(ordered) == 1 else ordered,
                }
            )
        configs.append({"config_name": config_name, "data_files": data_files})
    return configs


def render_card(
    *,
    repo_id: str,
    config_name: str,
    release_stats: dict[str, Any],
    published: DatasetDict,
    license_policy: str,
    configs: list[dict[str, Any]],
) -> str:
    published_categories = category_counts(published)
    published_sources = source_counts(published)
    front_matter: dict[str, Any] = {
        "language": ["ja"],
        "task_categories": ["automatic-speech-recognition"],
        "license": "other",
        "pretty_name": "JP-HomophoneBench",
        "tags": [
            "japanese",
            "asr",
            "homophone",
            "phoneme",
            "contextual-asr",
            "context-biasing",
            "speech-recognition",
        ],
    }
    if configs:
        front_matter["configs"] = configs

    lines = [
        "---",
        yaml.safe_dump(
            front_matter,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip(),
        "---",
        "",
        "# JP-HomophoneBench",
        "",
        "A deterministic Japanese ASR benchmark index for separating eight error/disambiguation classes:",
        "",
    ]
    lines.extend(f"- `{category}`" for category in CORE_CATEGORIES)
    lines.extend(
        [
            "",
            "## Important design rule",
            "",
            "This repository is metadata-first. Source audio is **not redistributed by default**. Each row stores source repository/config/split/row identifiers so audio can be rehydrated under the original source license.",
            "",
            "`exact_homophone` and `semantic_only` are intentionally separate. A phoneme scorer is not expected to solve `semantic_only` by itself.",
            "",
            f"HF repository: `{repo_id}`",
            f"Most recently published configuration: `{config_name}`",
            f"Publication license policy: `{license_policy}`",
            "",
            "## Available configurations",
            "",
        ]
    )
    if configs:
        lines.extend(f"- `{item['config_name']}`" for item in configs)
    else:
        lines.append("- No Parquet-backed configuration metadata was discovered.")
    lines.extend(["", "## Published rows in this run", "", "### Splits", ""])
    for split, ds in published.items():
        lines.append(f"- `{split}`: {len(ds):,}")
    lines.extend(["", "### Categories", ""])
    for category, count in published_categories.items():
        lines.append(f"- `{category}`: {count:,}")
    lines.extend(["", "### Upstream sources", ""])
    for source, count in published_sources.items():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(
        [
            "",
            "## Licensing and provenance",
            "",
            "The combined benchmark uses `license: other` because rows retain their upstream license. Inspect `source.license` before redistribution or commercial use.",
            "",
            "## Loading",
            "",
            "```python",
            "from datasets import load_dataset",
            f'ds = load_dataset("{repo_id}", "{config_name}")',
            "print(ds)",
            "```",
            "",
            "## Build summary",
            "",
            "```json",
            json.dumps(release_stats, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def upload_text(
    api: HfApi,
    *,
    repo_id: str,
    path_in_repo: str,
    text: str,
    message: str,
) -> None:
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a validated JP-HomophoneBench release")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="namespace/JP-HomophoneBench")
    parser.add_argument("--config-name", default="homophone8")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--license-policy",
        choices=("permissive", "research"),
        default="permissive",
    )
    parser.add_argument("--include-aux", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/benchmark.schema.json"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("HF token missing: login with `hf auth login` or set HF_TOKEN")

    stats_path = args.release_dir / "stats.json"
    release_stats = (
        json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    )
    dataset = filter_dataset(
        load_release(args.release_dir),
        license_policy=args.license_policy,
        include_aux=args.include_aux,
    )
    counts = category_counts(dataset)
    missing = [category for category in CORE_CATEGORIES if counts.get(category, 0) == 0]
    if missing and not args.allow_incomplete:
        raise SystemExit(f"refusing to publish incomplete core8 config; missing={missing}")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    dataset.push_to_hub(
        args.repo_id,
        config_name=args.config_name,
        private=args.private,
        token=args.token,
        max_shard_size=args.max_shard_size,
        embed_external_files=False,
        commit_message=f"publish JP-HomophoneBench {args.config_name}",
    )

    configs = discover_hub_configs(api, args.repo_id)
    if not any(item["config_name"] == args.config_name for item in configs):
        raise RuntimeError(
            f"published config {args.config_name!r} could not be reconstructed from Hub Parquet files: {configs}"
        )
    card = render_card(
        repo_id=args.repo_id,
        config_name=args.config_name,
        release_stats=release_stats,
        published=dataset,
        license_policy=args.license_policy,
        configs=configs,
    )
    upload_text(
        api,
        repo_id=args.repo_id,
        path_in_repo="README.md",
        text=card,
        message="docs: update JP-HomophoneBench dataset card without dropping configs",
    )
    if stats_path.exists():
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo=f"provenance/{args.config_name}/stats.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="chore: upload benchmark release statistics",
        )
    if args.schema.exists():
        api.upload_file(
            path_or_fileobj=str(args.schema),
            path_in_repo="schema/benchmark.schema.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="chore: upload benchmark JSON schema",
        )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "config": args.config_name,
                "private": args.private,
                "license_policy": args.license_policy,
                "splits": {name: len(ds) for name, ds in dataset.items()},
                "categories": counts,
                "hub_configs": configs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
