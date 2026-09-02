#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

HEX64 = re.compile(r"^[0-9a-f]{64}$")
DOMAIN = "jpacf-stage-fingerprint-v1"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_source(root: Path, value: str) -> list[Path]:
    candidate = root / value
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return sorted(path for path in candidate.rglob("*") if path.is_file())

    matches: list[Path] = []
    for path in sorted(root.glob(value)):
        if path.is_file():
            matches.append(path)
        elif path.is_dir():
            matches.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    if not matches:
        raise SystemExit(f"fingerprint source matched no files: {value}")
    return matches


def compute_fingerprints(
    payload: dict[str, Any],
    *,
    root: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    tasks = payload.get("snapshot_tasks") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise SystemExit("snapshot_tasks is empty")

    producer: dict[str, str] = {}
    for task, raw_spec in tasks.items():
        spec = dict(raw_spec or {})
        output = str(spec.get("output") or "")
        if not output:
            raise SystemExit(f"snapshot task has no output: {task}")
        if output in producer:
            raise SystemExit(f"duplicate snapshot output stage: {output}")
        producer[output] = str(task)

    memo: dict[str, str] = {}
    visiting: set[str] = set()

    def fingerprint(task: str) -> str:
        if task in memo:
            return memo[task]
        if task in visiting:
            raise SystemExit(f"snapshot dependency cycle at task: {task}")
        if task not in tasks:
            raise SystemExit(f"unknown snapshot task: {task}")
        visiting.add(task)

        spec = dict(tasks[task] or {})
        inputs = [str(item) for item in spec.get("inputs") or []]
        sources = [str(item) for item in spec.get("fingerprint_sources") or []]
        if not sources:
            raise SystemExit(f"snapshot task has no fingerprint_sources: {task}")

        values = spec.get("fingerprint_values") or {}
        if not isinstance(values, dict):
            raise SystemExit(f"fingerprint_values must be a mapping: {task}")

        external_names = [str(item) for item in spec.get("fingerprint_external") or []]
        external: dict[str, str] = {}
        for name in external_names:
            value = str(env.get(name) or "")
            if not value:
                raise SystemExit(f"required fingerprint environment is missing: {name}")
            if name.endswith("SHA256") and not HEX64.fullmatch(value.lower()):
                raise SystemExit(f"{name} must be a full SHA-256 hex digest")
            external[name] = value.lower() if name.endswith("SHA256") else value

        files: list[dict[str, str]] = []
        seen: set[Path] = set()
        for source in sources:
            for path in _expand_source(root, source):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    rel = resolved.relative_to(root.resolve()).as_posix()
                except ValueError as exc:
                    raise SystemExit(f"fingerprint source escaped repository root: {path}") from exc
                files.append({"path": rel, "sha256": _hash_file(resolved)})
        files.sort(key=lambda row: row["path"])

        upstream: list[dict[str, str]] = []
        for stage in inputs:
            upstream_task = producer.get(stage)
            if upstream_task is None:
                raise SystemExit(f"snapshot input has no producer: task={task} stage={stage}")
            upstream.append(
                {
                    "stage": stage,
                    "task": upstream_task,
                    "fingerprint": fingerprint(upstream_task),
                }
            )

        canonical = {
            "domain": DOMAIN,
            "task": task,
            "output": str(spec.get("output") or ""),
            "publish": [str(item) for item in spec.get("publish") or []],
            "files": files,
            "values": values,
            "external": external,
            "upstream": upstream,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        memo[task] = hashlib.sha256(encoded).hexdigest()
        visiting.remove(task)
        return memo[task]

    for task in tasks:
        fingerprint(str(task))
    return {str(task): memo[str(task)] for task in tasks}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute content-addressed implementation fingerprints for E00-E06 snapshot stages"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/e00-e06-artifacts.yaml"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--task")
    parser.add_argument("--field", choices=["json", "b64", "fingerprint"], default="json")
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mapping = compute_fingerprints(payload, root=args.root.resolve())

    if args.field == "fingerprint":
        if not args.task:
            raise SystemExit("--task is required with --field fingerprint")
        try:
            print(mapping[args.task])
        except KeyError as exc:
            raise SystemExit(f"unknown snapshot task: {args.task}") from exc
        return

    encoded = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if args.field == "b64":
        print(base64.b64encode(encoded.encode("utf-8")).decode("ascii"))
    else:
        print(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
