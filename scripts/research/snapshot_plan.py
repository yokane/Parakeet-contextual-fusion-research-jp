#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _fingerprints(config: Path) -> dict[str, str]:
    encoded = os.environ.get("JPA_CF_STAGE_FINGERPRINTS_B64", "")
    if encoded:
        try:
            payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
        except Exception as exc:
            raise SystemExit("invalid JPA_CF_STAGE_FINGERPRINTS_B64") from exc
    else:
        tool = Path(__file__).with_name("stage_fingerprints.py")
        proc = subprocess.run(
            [sys.executable, str(tool), "--config", str(config), "--field", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)

    if not isinstance(payload, dict) or not payload:
        raise SystemExit("stage fingerprint mapping is empty")
    mapping = {str(key): str(value).lower() for key, value in payload.items()}
    for task, value in mapping.items():
        if not HEX64.fullmatch(value):
            raise SystemExit(f"invalid stage fingerprint for {task}: {value!r}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve immutable HF research snapshot lineage")
    parser.add_argument("task")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/e00-e06-artifacts.yaml"),
    )
    parser.add_argument(
        "--field",
        choices=[
            "json",
            "inputs",
            "input_refs",
            "output",
            "output_ref",
            "publish",
            "fingerprint",
        ],
        default="json",
    )
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

    fingerprints = _fingerprints(args.config)
    if args.task not in fingerprints:
        raise SystemExit(f"missing fingerprint for task: {args.task}")

    stage_producer: dict[str, str] = {}
    for task_name, raw_task in tasks.items():
        stage = str((raw_task or {}).get("output") or "")
        if not stage:
            raise SystemExit(f"snapshot task has no output: {task_name}")
        if stage in stage_producer:
            raise SystemExit(f"duplicate snapshot output stage: {stage}")
        stage_producer[stage] = str(task_name)

    input_refs: list[str] = []
    for stage in inputs:
        producer = stage_producer.get(stage)
        if producer is None:
            raise SystemExit(f"snapshot input has no producer: {stage}")
        try:
            fingerprint = fingerprints[producer]
        except KeyError as exc:
            raise SystemExit(f"missing fingerprint for producer task: {producer}") from exc
        input_refs.append(f"{stage}/{fingerprint}")

    fingerprint = fingerprints[args.task]
    output_ref = f"{output}/{fingerprint}"
    plan = {
        "schema_version": int(payload.get("schema_version") or 1),
        "bucket_prefix": str(payload.get("bucket_prefix") or "workspace-cache/e00-e06"),
        "task": args.task,
        "inputs": inputs,
        "input_refs": input_refs,
        "output": output,
        "output_ref": output_ref,
        "fingerprint": fingerprint,
        "publish": publish,
    }

    if args.field == "json":
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.field == "inputs":
        print("\n".join(inputs))
    elif args.field == "input_refs":
        print("\n".join(input_refs))
    elif args.field == "output":
        print(output)
    elif args.field == "output_ref":
        print(output_ref)
    elif args.field == "fingerprint":
        print(fingerprint)
    else:
        print("\n".join(publish))


if __name__ == "__main__":
    main()
