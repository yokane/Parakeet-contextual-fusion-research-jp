#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from parakeet_context_fusion.benchmark import CORE_CATEGORIES


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def validate_release(release_dir: Path, schema_path: Path, require_core8: bool) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    category_counts: Counter[str] = Counter()
    errors: list[str] = []

    for split in ("train", "validation", "test"):
        path = release_dir / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        for row_index, row in enumerate(load_rows(path), 1):
            prefix = f"{path.name}:{row_index}"
            for error in validator.iter_errors(row):
                errors.append(f"{prefix}: schema: {error.message}")
            record_id = str(row.get("id", ""))
            if record_id in ids:
                errors.append(f"{prefix}: duplicate id {record_id}")
            ids.add(record_id)
            if row.get("split") != split:
                errors.append(f"{prefix}: embedded split={row.get('split')!r}, expected {split!r}")
            group_id = str(row.get("group_id", ""))
            if group_id:
                group_splits[group_id].add(split)
            category = str(row.get("category", ""))
            category_counts[category] += 1

            target = row.get("target") or {}
            candidates = [item for item in row.get("candidates", []) if item.get("relation") != "target"]
            if category in {"exact_homophone", "semantic_only"} and candidates:
                target_phones = target.get("phones") or []
                for candidate in candidates:
                    if candidate.get("phones") != target_phones:
                        errors.append(f"{prefix}: {category} candidate is not phone-identical")
                    if float(candidate.get("phone_distance", -1)) != 0.0:
                        errors.append(f"{prefix}: {category} candidate phone_distance must be 0")
            if category == "semantic_only":
                if not row.get("text") or row.get("text") == target.get("surface"):
                    errors.append(f"{prefix}: semantic_only requires sentence context")
                if not row.get("metadata", {}).get("context_required"):
                    errors.append(f"{prefix}: semantic_only must set context_required=true")
            if category == "pitch_accent" and target.get("pitch_accent") is None:
                errors.append(f"{prefix}: pitch_accent requires target.pitch_accent")
            if not row.get("source", {}).get("license"):
                errors.append(f"{prefix}: source.license is required for publication provenance")

    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"group leakage: {group_id} appears in {sorted(splits)}")

    if require_core8:
        missing = [category for category in CORE_CATEGORIES if category_counts[category] == 0]
        if missing:
            errors.append(f"missing core8 categories: {missing}")

    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:100])
        suffix = "" if len(errors) <= 100 else f"\n... {len(errors) - 100} more"
        raise SystemExit(f"release validation failed ({len(errors)} errors):\n{preview}{suffix}")

    print(json.dumps({
        "records": len(ids),
        "groups": len(group_splits),
        "categories": dict(sorted(category_counts.items())),
        "core8_complete": all(category_counts[item] > 0 for item in CORE_CATEGORIES),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schemas/benchmark.schema.json"))
    parser.add_argument("--require-core8", action="store_true")
    args = parser.parse_args()
    validate_release(args.release_dir, args.schema, args.require_core8)


if __name__ == "__main__":
    main()
