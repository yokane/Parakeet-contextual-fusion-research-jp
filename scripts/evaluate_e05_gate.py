#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (center - margin) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide whether E05 phone reranking is justified by recoverable E04 near-homophone errors"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/e05_gate.json"))
    parser.add_argument("--experiment", default="E04")
    parser.add_argument("--oracle-k", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-headroom", type=float, default=0.03)
    parser.add_argument("--min-wilson-lower", type=float, default=0.0)
    args = parser.parse_args()

    rows = pq.read_table(args.metrics).to_pylist()
    near = [
        row
        for row in rows
        if row.get("experiment") == args.experiment and row.get("category") == "near_homophone"
    ]
    exact = [
        row
        for row in rows
        if row.get("experiment") == args.experiment and row.get("category") == "exact_homophone"
    ]
    oracle_key = f"oracle_at_{args.oracle_k}"

    evaluable = [
        row
        for row in near
        if row.get("entity_correct") is not None and row.get(oracle_key) is not None
    ]
    recoverable = [
        row for row in evaluable if not bool(row["entity_correct"]) and float(row[oracle_key]) >= 1.0
    ]
    count = len(evaluable)
    recoverable_count = len(recoverable)
    recoverable_rate = recoverable_count / count if count else 0.0
    lower = wilson_lower(recoverable_count, count)
    near_accuracy = (
        sum(1.0 if row.get("entity_correct") else 0.0 for row in evaluable) / count if count else None
    )
    near_oracle = sum(float(row[oracle_key]) for row in evaluable) / count if count else None
    headroom = (near_oracle - near_accuracy) if near_accuracy is not None and near_oracle is not None else 0.0

    exact_evaluable = [row for row in exact if row.get("entity_correct") is not None]
    exact_accuracy = (
        sum(1.0 if row.get("entity_correct") else 0.0 for row in exact_evaluable) / len(exact_evaluable)
        if exact_evaluable
        else None
    )

    passed = (
        count >= args.min_count
        and headroom >= args.min_headroom
        and lower > args.min_wilson_lower
        and recoverable_count > 0
    )
    payload = {
        "experiment": args.experiment,
        "decision": "run_e05" if passed else "stop_before_e05",
        "passed": passed,
        "near_homophone": {
            "count": count,
            "entity_accuracy": near_accuracy,
            f"oracle_at_{args.oracle_k}": near_oracle,
            "recoverable_count": recoverable_count,
            "recoverable_rate": recoverable_rate,
            "recoverable_rate_wilson95_lower": lower,
            "oracle_headroom": headroom,
        },
        "exact_homophone_control": {
            "count": len(exact_evaluable),
            "entity_accuracy": exact_accuracy,
            "note": "Exact-homophone errors are a semantic/context control and are not expected to be fixed by E05 phones.",
        },
        "thresholds": {
            "min_count": args.min_count,
            "min_headroom": args.min_headroom,
            "min_wilson_lower": args.min_wilson_lower,
            "oracle_k": args.oracle_k,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
