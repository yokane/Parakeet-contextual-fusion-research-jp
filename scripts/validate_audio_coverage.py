#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_report(
    provenance: dict[str, Any],
    *,
    required_categories: list[str],
    min_per_category: int,
    min_total: int,
) -> dict[str, Any]:
    total_categories = {
        str(key): int(value) for key, value in (provenance.get("categories") or {}).items()
    }
    runnable_categories = {
        str(key): int(value)
        for key, value in (provenance.get("runnable_categories") or {}).items()
    }
    category_names = sorted(total_categories.keys() | runnable_categories.keys())
    categories: dict[str, dict[str, int | float | None]] = {}
    for category in category_names:
        total = total_categories.get(category, 0)
        runnable = runnable_categories.get(category, 0)
        categories[category] = {
            "total": total,
            "runnable": runnable,
            "coverage": (runnable / total) if total else None,
        }

    runnable_total = int(provenance.get("runnable_audio_records") or 0)
    failures: list[str] = []
    if runnable_total < min_total:
        failures.append(f"runnable_total={runnable_total} < min_total={min_total}")
    for category in required_categories:
        runnable = runnable_categories.get(category, 0)
        if runnable < min_per_category:
            failures.append(
                f"category={category} runnable={runnable} < min_per_category={min_per_category}"
            )

    return {
        "repo_id": provenance.get("repo_id"),
        "config": provenance.get("config"),
        "splits": provenance.get("splits"),
        "records": int(provenance.get("records") or 0),
        "runnable_audio_records": runnable_total,
        "overall_coverage": (
            runnable_total / int(provenance.get("records"))
            if int(provenance.get("records") or 0) > 0
            else None
        ),
        "categories": categories,
        "requirements": {
            "required_categories": required_categories,
            "min_per_category": min_per_category,
            "min_total": min_total,
        },
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate category-level runnable-audio coverage before acoustic ASR claims"
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--required-category", action="append", default=[])
    parser.add_argument("--min-per-category", type=int, default=1)
    parser.add_argument("--min-total", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.min_per_category < 0 or args.min_total < 0:
        raise SystemExit("minimum counts must be non-negative")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    report = build_report(
        provenance,
        required_categories=list(dict.fromkeys(args.required_category)),
        min_per_category=args.min_per_category,
        min_total=args.min_total,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("audio coverage requirements were not met")


if __name__ == "__main__":
    main()
