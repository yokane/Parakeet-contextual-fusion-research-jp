#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def validate_fresh_upload_plan(rows: list[dict[str, Any]], expected_dest: str | None = None) -> int:
    if not rows or rows[0].get("type") != "header":
        raise ValueError("sync plan must start with a header row")
    header = rows[0]
    if expected_dest is not None and str(header.get("dest")) != expected_dest:
        raise ValueError(f"unexpected sync destination: {header.get('dest')!r} != {expected_dest!r}")

    operations = [row for row in rows[1:] if row.get("type") == "operation"]
    if len(operations) != len(rows) - 1:
        raise ValueError("sync plan contains an unknown row type")
    if not operations:
        raise ValueError("sync plan contains no file operations")

    forbidden = [
        (str(row.get("action")), str(row.get("path")))
        for row in operations
        if row.get("action") != "upload"
    ]
    if forbidden:
        formatted = ", ".join(f"{action}:{path}" for action, path in forbidden[:10])
        raise ValueError(
            "fresh immutable publish accepts only upload operations; "
            f"refusing plan operations: {formatted}"
        )
    return len(operations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an append-only Hugging Face Bucket sync plan")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--expected-dest")
    args = parser.parse_args()
    count = validate_fresh_upload_plan(load_jsonl(args.plan), args.expected_dest)
    print(f"upload_count={count}")


if __name__ == "__main__":
    main()
