#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve immutable HF research snapshot lineage")
    parser.add_argument("task")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/e00-e06-artifacts.yaml"),
    )
    parser.add_argument("--field", choices=["json", "inputs", "output", "publish"], default="json")
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = payload.get("snapshot_tasks") or {}
    if args.task not in tasks:
        raise SystemExit(f"unknown snapshot task: {args.task}")

    spec = dict(tasks[args.task] or {})
    inputs = [str(item) for item in spec.get("inputs") or []]
    output = str(spec.get("output") or "")
    publish = [str(item) for item in spec.get("publish") or []]

    for value in [*inputs, output]:
        if not value or not SAFE.fullmatch(value):
            raise SystemExit(f"unsafe snapshot stage: {value!r}")
    for value in publish:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise SystemExit(f"unsafe publish path: {value!r}")

    plan = {
        "schema_version": int(payload.get("schema_version") or 1),
        "bucket_prefix": str(payload.get("bucket_prefix") or "workspace-cache/e00-e06"),
        "task": args.task,
        "inputs": inputs,
        "output": output,
        "publish": publish,
    }

    if args.field == "json":
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.field == "inputs":
        print("\n".join(inputs))
    elif args.field == "output":
        print(output)
    else:
        print("\n".join(publish))


if __name__ == "__main__":
    main()
