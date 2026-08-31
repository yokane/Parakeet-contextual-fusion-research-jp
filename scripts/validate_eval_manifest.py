#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from parakeet_context_fusion.benchmark import CORE_CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JP-HomophoneBench/NeMo JSONL evaluation manifests")
    parser.add_argument("input", type=Path)
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()

    seen: set[str] = set()
    categories: Counter[str] = Counter()
    errors: list[str] = []
    rows = 0
    audio_rows = 0
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc}")
            continue
        identifier = str(row.get("benchmark_id") or row.get("id") or "")
        if not identifier:
            errors.append(f"line {line_number}: missing benchmark_id/id")
        elif identifier in seen:
            errors.append(f"line {line_number}: duplicate id {identifier}")
        else:
            seen.add(identifier)
        category = str(row.get("category") or "")
        if category not in CORE_CATEGORIES:
            errors.append(f"line {line_number}: invalid core8 category {category!r}")
        else:
            categories[category] += 1
        if not isinstance(row.get("text"), str):
            errors.append(f"line {line_number}: text must be a string")
        audio = row.get("audio_filepath")
        if audio:
            audio_rows += 1
            path = Path(str(audio)).expanduser()
            if not path.exists():
                errors.append(f"line {line_number}: audio file does not exist: {path}")
            duration = row.get("duration")
            if duration is not None and float(duration) <= 0:
                errors.append(f"line {line_number}: duration must be positive")
        elif args.require_audio:
            errors.append(f"line {line_number}: audio_filepath is required")

    report = {
        "input": str(args.input),
        "rows": rows,
        "audio_rows": audio_rows,
        "unique_ids": len(seen),
        "categories": dict(sorted(categories.items())),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
