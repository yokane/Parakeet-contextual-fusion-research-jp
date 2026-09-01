#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from build_run_bundle import sha256


def validate_run_bundle(run_dir: Path) -> dict[str, object]:
    required = ["run-context.json", "samples.jsonl", "metrics.json", "run.parquet"]
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required run artifact missing or empty: {name}")

    context = json.loads((run_dir / "run-context.json").read_text(encoding="utf-8"))
    run_id = context.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run-context.json has no run_id")

    jsonl_count = sum(
        1 for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )
    parquet_count = pq.read_table(run_dir / "run.parquet").num_rows
    if jsonl_count != parquet_count:
        raise ValueError(
            f"samples.jsonl and run.parquet count mismatch: {jsonl_count} != {parquet_count}"
        )
    if int(context.get("sample_count", -1)) != parquet_count:
        raise ValueError("run-context sample_count does not match run.parquet")

    digest = sha256(run_dir / "run.parquet")
    if context.get("run_parquet_sha256") != digest:
        raise ValueError("run-context run_parquet_sha256 does not match run.parquet")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a JSON object")

    return {"run_id": run_id, "sample_count": parquet_count, "run_parquet_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a J-PACF HF Bucket run bundle")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run_bundle(args.run_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
