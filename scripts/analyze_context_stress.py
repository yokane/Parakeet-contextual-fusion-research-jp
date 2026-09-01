#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

MetricFn = Callable[[dict[str, Any]], float | None]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    pairs: list[tuple[float, float]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, int | float | None]:
    if not pairs:
        return {"n": 0, "delta": None, "low": None, "high": None}
    deltas = [new - old for old, new in pairs]
    observed = mean(deltas)
    if len(deltas) == 1 or samples <= 0:
        return {"n": len(deltas), "delta": observed, "low": observed, "high": observed}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        values.append(sum(sample) / len(sample))
    values.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "n": len(deltas),
        "delta": observed,
        "low": percentile(values, tail),
        "high": percentile(values, 1.0 - tail),
    }


def entity_metric(row: dict[str, Any]) -> float | None:
    value = row.get("entity_correct")
    return None if value is None else float(bool(value))


def fpr_metric(row: dict[str, Any]) -> float | None:
    value = row.get("distractor_false_positive")
    return None if value is None else float(bool(value))


def cer_metric(row: dict[str, Any]) -> float | None:
    value = row.get("cer")
    return None if value is None else float(value)


METRICS: dict[str, MetricFn] = {
    "entity_accuracy": entity_metric,
    "distractor_false_positive_rate": fpr_metric,
    "cer": cer_metric,
}


def paired_metric(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    metric: MetricFn,
    *,
    category: str | None,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for benchmark_id in sorted(baseline.keys() & current.keys()):
        old = baseline[benchmark_id]
        new = current[benchmark_id]
        if category is not None and (
            str(old.get("category")) != category or str(new.get("category")) != category
        ):
            continue
        old_value = metric(old)
        new_value = metric(new)
        if old_value is not None and new_value is not None:
            pairs.append((old_value, new_value))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure J-PACF robustness as unrelated context distractors are added"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--stress-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/context_stress.json"))
    parser.add_argument("--experiment-prefix", default="E03_ctxd_")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-entity-drop", type=float, default=0.02)
    parser.add_argument("--max-fpr-increase", type=float, default=0.02)
    parser.add_argument("--max-cer-increase", type=float, default=0.01)
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    rows = [dict(row) for row in pq.read_table(args.metrics).to_pylist()]
    stress = json.loads(args.stress_manifest.read_text(encoding="utf-8"))
    cases = stress.get("cases") or []
    if not cases:
        raise SystemExit("stress manifest contains no cases")

    by_experiment: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        experiment = str(row.get("experiment") or "")
        benchmark_id = str(row.get("benchmark_id") or "")
        if not experiment or not benchmark_id:
            continue
        by_experiment.setdefault(experiment, {})[benchmark_id] = row

    ordered_cases = sorted(cases, key=lambda item: int(item["requested_distractors"]))
    baseline_case = ordered_cases[0]
    baseline_name = f"{args.experiment_prefix}{int(baseline_case['requested_distractors']):05d}"
    baseline = by_experiment.get(baseline_name)
    if baseline is None:
        raise SystemExit(f"baseline experiment is missing from metrics: {baseline_name}")

    categories = sorted(
        {str(row.get("category")) for row in baseline.values() if row.get("category") is not None}
    )

    def compare(
        experiment: str,
        current: dict[str, dict[str, Any]],
        *,
        category: str | None,
        seed_offset: int,
    ) -> dict[str, Any]:
        entity = paired_bootstrap_ci(
            paired_metric(baseline, current, entity_metric, category=category),
            samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed + seed_offset,
        )
        fpr = paired_bootstrap_ci(
            paired_metric(baseline, current, fpr_metric, category=category),
            samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed + seed_offset + 1,
        )
        cer = paired_bootstrap_ci(
            paired_metric(baseline, current, cer_metric, category=category),
            samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed + seed_offset + 2,
        )
        entity_low = entity.get("low")
        fpr_high = fpr.get("high")
        cer_high = cer.get("high")
        within_entity = entity_low is None or float(entity_low) >= -args.max_entity_drop
        within_fpr = fpr_high is None or float(fpr_high) <= args.max_fpr_increase
        within_cer = cer_high is None or float(cer_high) <= args.max_cer_increase
        return {
            "experiment": experiment,
            "category": category or "all",
            "entity_accuracy_delta": entity,
            "distractor_fpr_delta": fpr,
            "cer_delta": cer,
            "within_threshold": within_entity and within_fpr and within_cer,
            "threshold_checks": {
                "entity": within_entity,
                "fpr": within_fpr,
                "cer": within_cer,
            },
        }

    reports: list[dict[str, Any]] = []
    for case_index, case in enumerate(ordered_cases):
        requested = int(case["requested_distractors"])
        experiment = f"{args.experiment_prefix}{requested:05d}"
        current = by_experiment.get(experiment)
        if current is None:
            raise SystemExit(f"stress experiment is missing from metrics: {experiment}")
        overall = compare(experiment, current, category=None, seed_offset=case_index * 100)
        by_category = {
            category: compare(
                experiment,
                current,
                category=category,
                seed_offset=case_index * 100 + category_index * 10,
            )
            for category_index, category in enumerate(categories, 1)
        }
        reports.append(
            {
                **case,
                "experiment": experiment,
                "overall": overall,
                "by_category": by_category,
            }
        )

    robust_cases = [item for item in reports if item["overall"]["within_threshold"]]
    report = {
        "metrics": str(args.metrics),
        "stress_manifest": str(args.stress_manifest),
        "baseline_experiment": baseline_name,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "thresholds": {
            "max_entity_drop": args.max_entity_drop,
            "max_fpr_increase": args.max_fpr_increase,
            "max_cer_increase": args.max_cer_increase,
        },
        "cases": reports,
        "max_robust_requested_distractors": (
            max(int(item["requested_distractors"]) for item in robust_cases)
            if robust_cases
            else None
        ),
        "interpretation": (
            "Robustness is measured against the smallest distractor case. A case remains inside the "
            "default envelope only when the confidence bounds for entity-accuracy loss, distractor-FPR "
            "increase, and CER increase all stay within configured tolerances."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
