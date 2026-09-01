#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_run_bundle(
    *,
    results_dir: Path,
    output_dir: Path,
    run_id: str,
    workflow_kind: str,
    benchmark_index: Path | None,
    execution_manifest: Path | None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(f"unsafe run ID: {run_id!r}")
    results_dir = results_dir.resolve()
    metrics_parquet = results_dir / "metrics.parquet"
    if not metrics_parquet.is_file():
        raise ValueError(f"metrics.parquet is required: {metrics_parquet}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    evidence = output_dir / "evidence"
    shutil.copytree(results_dir, evidence / "results")

    run_parquet = output_dir / "run.parquet"
    shutil.copy2(metrics_parquet, run_parquet)
    rows = [dict(row) for row in pq.read_table(run_parquet).to_pylist()]
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_path = results_dir / "summary.json"
    metrics: dict[str, Any]
    if summary_path.is_file():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = loaded if isinstance(loaded, dict) else {"summary": loaded}
    else:
        metrics = {"rows": len(rows)}
    write_json(output_dir / "metrics.json", metrics)

    inputs_dir = evidence / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for source, name in [
        (benchmark_index, "bench_index.jsonl"),
        (execution_manifest, "nemo_eval.jsonl"),
    ]:
        if source is not None:
            if not source.is_file():
                raise ValueError(f"run input is missing: {source}")
            shutil.copy2(source, inputs_dir / name)

    context = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow_kind": workflow_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "J-PACF-YOMI-TDT",
        "model": os.environ.get("MODEL_NAME", "nvidia/parakeet-tdt_ctc-0.6b-ja"),
        "benchmark": "saeeew/JP-HomophoneBench",
        "source_repository": os.environ.get("GITHUB_REPOSITORY"),
        "source_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "source_run_id": os.environ.get("GITHUB_RUN_ID"),
        "source_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_sha": os.environ.get("GITHUB_SHA"),
        "sample_count": len(rows),
        "run_parquet_sha256": sha256(run_parquet),
    }
    write_json(output_dir / "run-context.json", context)
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an append-only J-PACF HF Bucket run bundle")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-kind", required=True)
    parser.add_argument("--benchmark-index", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    args = parser.parse_args()
    context = build_run_bundle(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        workflow_kind=args.workflow_kind,
        benchmark_index=args.benchmark_index,
        execution_manifest=args.execution_manifest,
    )
    print(json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
