#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[1] / "schemas" / "benchmark.schema.json")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = 0
    records = 0
    with args.manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records += 1
            record = json.loads(line)
            for error in validator.iter_errors(record):
                errors += 1
                path = ".".join(str(item) for item in error.absolute_path)
                print(f"{args.manifest}:{line_number}:{path}: {error.message}")
    if errors:
        raise SystemExit(f"validation failed: {errors} errors in {records} records")
    print(f"validated {records} records")


if __name__ == "__main__":
    main()
