#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parakeet_context_fusion.benchmark import (
    CORE_CATEGORIES,
    classify_readings,
    difficulty_vector,
    make_group_id,
    stable_split,
)
from parakeet_context_fusion.japanese_g2p import reading_to_phones

ALLOWED = {"near_homophone", "voicing", "long_vowel", "geminate", "moraic_nasal"}


def make_record(index: int, row: dict[str, str]) -> dict[str, Any]:
    category = row["category"].strip()
    if category not in ALLOWED:
        raise ValueError(f"supplement row {index + 2}: unsupported category {category!r}")
    target_reading = row["target_reading"].strip()
    candidate_reading = row["candidate_reading"].strip()
    relation = classify_readings(target_reading, candidate_reading, near_threshold=1.0)
    if relation.category != category:
        raise ValueError(
            f"supplement row {index + 2}: declared {category!r}, derived {relation.category!r}"
        )
    target_phones = list(reading_to_phones(target_reading))
    candidate_phones = list(reading_to_phones(candidate_reading))
    group_id = make_group_id(target_reading, candidate_reading)
    split = stable_split(group_id)
    target = {
        "surface": row["target_surface"].strip(),
        "reading": target_reading,
        "phones": target_phones,
    }
    candidate = {
        "surface": row["candidate_surface"].strip(),
        "reading": candidate_reading,
        "phones": candidate_phones,
        "relation": category,
        "phone_distance": relation.phone_distance,
    }
    return {
        "id": f"permissive-{category}-{index:04d}",
        "group_id": group_id,
        "split": split,
        "audio_filepath": None,
        "audio_ref": {
            "kind": "none",
            "repo_id": None,
            "config": None,
            "split": "manual",
            "row_id": str(index),
        },
        "text": target["surface"],
        "target": target,
        "candidates": [
            {**target, "relation": "target", "phone_distance": 0.0},
            candidate,
        ],
        "category": category,
        "source": {
            "dataset": "JP-HomophoneBench/permissive-phonetic-core",
            "config": "cc0-synthetic",
            "split": "manual",
            "source_id": str(index),
            "revision": "v0.1",
            "license": "cc0-1.0",
            "synthetic": True,
            "audio_redistributed": False,
        },
        "difficulty": difficulty_vector(
            category=category,
            phone_distance=relation.phone_distance,
            candidate_count=2,
            has_context=False,
        ),
        "metadata": {
            "derived_relation": relation.category,
            "relation_reason": relation.reason,
            "context_required": False,
            "notes": row.get("notes") or None,
            "supplement": "permissive-core8-v0.1",
        },
    }


def load_existing(release_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = release_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_release(release_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda item: str(item["id"]))
    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        split_rows[str(row["split"])].append(row)

    hashes: dict[str, str] = {}
    for split, values in split_rows.items():
        path = release_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    all_path = release_dir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for value in rows:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    hashes[all_path.name] = hashlib.sha256(all_path.read_bytes()).hexdigest()

    categories = Counter(str(row["category"]) for row in rows)
    sources = Counter(str((row.get("source") or {}).get("dataset", "unknown")) for row in rows)
    licenses = Counter(str((row.get("source") or {}).get("license", "unknown")) for row in rows)
    stats = {
        "records": len(rows),
        "splits": {name: len(values) for name, values in split_rows.items()},
        "categories": dict(sorted(categories.items())),
        "sources": dict(sorted(sources.items())),
        "licenses": dict(sorted(licenses.items())),
        "core8_missing": [category for category in CORE_CATEGORIES if not categories[category]],
        "sha256": hashes,
        "augmentation": "permissive-core8-v0.1",
    }
    (release_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add CC0 phonetic fixtures for permissive core8")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    args = parser.parse_args()

    existing = load_existing(args.release_dir)
    existing_ids = {str(row["id"]) for row in existing}
    with args.seed.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "category", "target_surface", "target_reading",
            "candidate_surface", "candidate_reading",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"supplement TSV missing columns: {sorted(missing)}")
        additions = [make_record(index, row) for index, row in enumerate(reader)]

    additions = [row for row in additions if str(row["id"]) not in existing_ids]
    write_release(args.release_dir, existing + additions)
    print(json.dumps({"added": len(additions), "release_dir": str(args.release_dir)}, indent=2))


if __name__ == "__main__":
    main()
