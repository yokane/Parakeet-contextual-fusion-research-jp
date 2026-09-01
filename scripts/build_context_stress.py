#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def row_phrases(row: dict[str, Any]) -> set[str]:
    phrases: set[str] = set()
    target = row.get("target") or {}
    target_surface = str(target.get("surface") or row.get("target_surface") or "").strip()
    if target_surface:
        phrases.add(target_surface)
    candidates = row.get("candidates") or []
    for candidate in candidates:
        surface = str((candidate or {}).get("surface") or "").strip()
        if surface:
            phrases.add(surface)
    for surface in row.get("candidate_surfaces") or []:
        value = str(surface or "").strip()
        if value:
            phrases.add(value)
    return phrases


def deterministic_key(seed: int, phrase: str) -> str:
    return hashlib.sha256(f"{seed}\0{phrase}".encode()).hexdigest()


def parse_counts(raw: str) -> list[int]:
    counts: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise SystemExit("distractor counts must be non-negative")
        counts.add(value)
    if not counts:
        raise SystemExit("at least one distractor count is required")
    return sorted(counts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build nested global context lists that keep all runnable target/hard-negative phrases "
            "fixed while adding deterministic unrelated distractors"
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distractor-counts", default="0,10,100")
    parser.add_argument("--external-distractors", type=Path)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--allow-short-pool",
        action="store_true",
        help="cap requested distractor counts at available pool size instead of failing",
    )
    args = parser.parse_args()

    benchmark_rows = read_jsonl(args.benchmark)
    execution_rows = read_jsonl(args.execution_manifest)
    runnable_ids = {
        str(row.get("benchmark_id") or row.get("id") or "").strip()
        for row in execution_rows
        if str(row.get("benchmark_id") or row.get("id") or "").strip()
    }
    if not runnable_ids:
        raise SystemExit("execution manifest contains no benchmark IDs")

    by_id = {str(row.get("id") or ""): row for row in benchmark_rows if row.get("id")}
    missing_ids = sorted(runnable_ids - by_id.keys())
    if missing_ids:
        raise SystemExit(f"execution manifest references IDs absent from benchmark: {missing_ids[:10]}")

    required_phrases: set[str] = set()
    distractor_pool: set[str] = set()
    for benchmark_id, row in by_id.items():
        phrases = row_phrases(row)
        if benchmark_id in runnable_ids:
            required_phrases.update(phrases)
        else:
            distractor_pool.update(phrases)

    if args.external_distractors is not None:
        with args.external_distractors.open("r", encoding="utf-8") as handle:
            for line in handle:
                phrase = line.strip()
                if phrase and not phrase.startswith("#"):
                    distractor_pool.add(phrase)

    distractor_pool.difference_update(required_phrases)
    ordered_distractors = sorted(
        distractor_pool,
        key=lambda phrase: (deterministic_key(args.seed, phrase), phrase),
    )
    required = sorted(required_phrases)
    if not required:
        raise SystemExit("no target/candidate phrases found for runnable rows")

    counts = parse_counts(args.distractor_counts)
    maximum = counts[-1]
    if maximum > len(ordered_distractors) and not args.allow_short_pool:
        raise SystemExit(
            f"requested {maximum} distractors but only {len(ordered_distractors)} are available; "
            "provide --external-distractors or --allow-short-pool"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for requested in counts:
        actual = min(requested, len(ordered_distractors))
        selected = ordered_distractors[:actual]
        phrases = required + selected
        filename = f"context_d{requested:05d}.txt"
        text = "\n".join(phrases) + "\n"
        path = args.output_dir / filename
        path.write_text(text, encoding="utf-8")
        cases.append(
            {
                "requested_distractors": requested,
                "actual_distractors": actual,
                "required_phrases": len(required),
                "total_phrases": len(phrases),
                "file": filename,
                "sha256": sha256_text(text),
            }
        )

    manifest = {
        "schema_version": 1,
        "benchmark": str(args.benchmark),
        "execution_manifest": str(args.execution_manifest),
        "seed": args.seed,
        "runnable_rows": len(runnable_ids),
        "required_phrase_count": len(required),
        "available_distractor_count": len(ordered_distractors),
        "external_distractors": (
            str(args.external_distractors) if args.external_distractors is not None else None
        ),
        "cases": cases,
        "interpretation": (
            "Each case preserves every target/candidate phrase used by runnable rows and changes only "
            "the number of unrelated distractor phrases. requested_distractors is the stress axis; "
            "total_phrases includes the fixed required phrase set."
        ),
    }
    manifest_path = args.output_dir / "context_stress_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
