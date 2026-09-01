#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
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
) -> dict[str, float | int | None]:
    if not pairs:
        return {"n": 0, "delta": None, "low": None, "high": None}
    deltas = [new - old for old, new in pairs]
    observed = mean(deltas)
    if len(deltas) == 1 or samples <= 0:
        return {"n": len(deltas), "delta": observed, "low": observed, "high": observed}

    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "n": len(deltas),
        "delta": observed,
        "low": percentile(bootstrap, tail),
        "high": percentile(bootstrap, 1.0 - tail),
    }


def binomial_cdf_half(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    numerator = sum(math.comb(n, i) for i in range(k + 1))
    return numerator / (2**n)


def mcnemar_exact(previous: list[bool], current: list[bool]) -> dict[str, int | float | None]:
    if len(previous) != len(current):
        raise ValueError("paired McNemar vectors must have the same length")
    wins = sum((not old) and new for old, new in zip(previous, current, strict=True))
    losses = sum(old and (not new) for old, new in zip(previous, current, strict=True))
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(1.0, 2.0 * binomial_cdf_half(min(wins, losses), discordant))
    return {
        "n": len(previous),
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "p_value": p_value,
    }


def entity_metric(row: dict[str, Any]) -> float | None:
    value = row.get("entity_correct")
    return None if value is None else float(bool(value))


def distractor_fpr_metric(row: dict[str, Any]) -> float | None:
    value = row.get("distractor_false_positive")
    return None if value is None else float(bool(value))


def bias_fpr_metric(row: dict[str, Any]) -> float | None:
    value = row.get("bias_false_positive")
    return None if value is None else float(bool(value))


def neg_cer_metric(row: dict[str, Any]) -> float | None:
    value = row.get("cer")
    return None if value is None else -float(value)


METRICS: dict[str, MetricFn] = {
    "entity_accuracy": entity_metric,
    "distractor_false_positive_rate": distractor_fpr_metric,
    "bias_false_positive_rate": bias_fpr_metric,
    "negative_cer": neg_cer_metric,
}


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def paired_rows(
    by_experiment: dict[str, dict[str, dict[str, Any]]],
    previous: str,
    current: str,
    *,
    category: str | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    old = by_experiment.get(previous, {})
    new = by_experiment.get(current, {})
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for benchmark_id in sorted(old.keys() & new.keys()):
        old_row = old[benchmark_id]
        new_row = new[benchmark_id]
        if category is not None and (
            str(old_row.get("category")) != category or str(new_row.get("category")) != category
        ):
            continue
        result.append((old_row, new_row))
    return result


def metric_pairs(rows: list[tuple[dict[str, Any], dict[str, Any]]], metric: MetricFn) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for old, new in rows:
        old_value = metric(old)
        new_value = metric(new)
        if old_value is not None and new_value is not None:
            values.append((old_value, new_value))
    return values


def aggregate_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows)}
    for name, metric in METRICS.items():
        values = [value for row in rows if (value := metric(row)) is not None]
        output[name] = mean(values)
    return output


def compare_adjacent(
    *,
    by_experiment: dict[str, dict[str, dict[str, Any]]],
    previous: str,
    current: str,
    category: str | None,
    benefit_metric: str,
    risk_metric: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    min_gain: float,
    min_risk_increase: float,
    alpha: float,
) -> dict[str, Any]:
    paired = paired_rows(by_experiment, previous, current, category=category)
    benefit_fn = METRICS[benefit_metric]
    risk_fn = METRICS[risk_metric]
    benefit = paired_bootstrap_ci(
        metric_pairs(paired, benefit_fn),
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    risk = paired_bootstrap_ci(
        metric_pairs(paired, risk_fn),
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed + 1,
    )

    previous_bool: list[bool] = []
    current_bool: list[bool] = []
    if benefit_metric == "entity_accuracy":
        for old, new in paired:
            old_value = old.get("entity_correct")
            new_value = new.get("entity_correct")
            if old_value is None or new_value is None:
                continue
            previous_bool.append(bool(old_value))
            current_bool.append(bool(new_value))
    mcnemar = mcnemar_exact(previous_bool, current_bool) if previous_bool else None

    benefit_high = benefit.get("high")
    risk_delta = risk.get("delta")
    risk_low = risk.get("low")
    statistically_no_entity_gain = bool(
        mcnemar is not None
        and mcnemar["p_value"] is not None
        and float(mcnemar["p_value"]) >= alpha
        and int(mcnemar["wins"]) <= int(mcnemar["losses"]) + 1
    )
    plateau = bool(
        benefit_high is not None
        and float(benefit_high) <= min_gain
        and (benefit_metric != "entity_accuracy" or statistically_no_entity_gain)
    )
    risk_rising = bool(
        risk_delta is not None
        and float(risk_delta) >= min_risk_increase
        and risk_low is not None
        and float(risk_low) > 0.0
    )
    return {
        "previous": previous,
        "current": current,
        "category": category or "all",
        "benefit_metric": benefit_metric,
        "risk_metric": risk_metric,
        "benefit_delta": benefit,
        "risk_delta": risk,
        "mcnemar_entity": mcnemar,
        "plateau": plateau,
        "risk_rising": risk_rising,
        "saturation_transition": plateau and risk_rising,
    }


def validate_sweep(sweep: dict[str, Any]) -> None:
    if not sweep.get("name"):
        raise ValueError("sweep is missing name")
    points = sweep.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"sweep {sweep['name']!r} needs at least two points")
    for point in points:
        if not isinstance(point, dict) or not point.get("experiment") or "value" not in point:
            raise ValueError(f"invalid sweep point: {point!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect contextual-bias saturation from paired E00-E06 metric rows"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/saturation.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--min-risk-increase", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    rows = read_metrics(args.metrics)
    spec = json.loads(args.sweep.read_text(encoding="utf-8"))
    sweeps = spec.get("sweeps") or []
    if not isinstance(sweeps, list) or not sweeps:
        raise SystemExit("sweep spec must contain a non-empty 'sweeps' array")

    by_experiment: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        experiment = str(row.get("experiment") or "")
        benchmark_id = str(row.get("benchmark_id") or "")
        if not experiment or not benchmark_id:
            continue
        by_experiment[experiment][benchmark_id] = row
        raw_by_experiment[experiment].append(row)

    available_categories = sorted(
        {str(row.get("category")) for row in rows if row.get("category") is not None}
    )
    report_sweeps: list[dict[str, Any]] = []
    for sweep_index, sweep in enumerate(sweeps):
        validate_sweep(sweep)
        benefit_metric = str(sweep.get("benefit_metric") or "entity_accuracy")
        risk_metric = str(sweep.get("risk_metric") or "distractor_false_positive_rate")
        if benefit_metric not in METRICS or risk_metric not in METRICS:
            raise SystemExit(
                f"unsupported metric in sweep {sweep['name']!r}: benefit={benefit_metric}, risk={risk_metric}"
            )
        points = sorted(sweep["points"], key=lambda point: float(point["value"]))
        missing_experiments = [
            str(point["experiment"])
            for point in points
            if str(point["experiment"]) not in by_experiment
        ]
        if missing_experiments:
            raise SystemExit(
                f"sweep {sweep['name']!r} references missing experiments: {missing_experiments}"
            )

        transitions: list[dict[str, Any]] = []
        category_transitions: dict[str, list[dict[str, Any]]] = {}
        for point_index in range(1, len(points)):
            previous = str(points[point_index - 1]["experiment"])
            current = str(points[point_index]["experiment"])
            transition = compare_adjacent(
                by_experiment=by_experiment,
                previous=previous,
                current=current,
                category=None,
                benefit_metric=benefit_metric,
                risk_metric=risk_metric,
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed + sweep_index * 1000 + point_index * 10,
                min_gain=args.min_gain,
                min_risk_increase=args.min_risk_increase,
                alpha=args.alpha,
            )
            transition["previous_value"] = points[point_index - 1]["value"]
            transition["current_value"] = points[point_index]["value"]
            transitions.append(transition)

        requested_categories = sweep.get("categories") or available_categories
        for category_index, category in enumerate(requested_categories):
            category_name = str(category)
            values: list[dict[str, Any]] = []
            for point_index in range(1, len(points)):
                previous = str(points[point_index - 1]["experiment"])
                current = str(points[point_index]["experiment"])
                transition = compare_adjacent(
                    by_experiment=by_experiment,
                    previous=previous,
                    current=current,
                    category=category_name,
                    benefit_metric=benefit_metric,
                    risk_metric=risk_metric,
                    bootstrap_samples=args.bootstrap_samples,
                    confidence=args.confidence,
                    seed=(
                        args.seed
                        + sweep_index * 1000
                        + category_index * 100
                        + point_index * 10
                    ),
                    min_gain=args.min_gain,
                    min_risk_increase=args.min_risk_increase,
                    alpha=args.alpha,
                )
                transition["previous_value"] = points[point_index - 1]["value"]
                transition["current_value"] = points[point_index]["value"]
                values.append(transition)
            category_transitions[category_name] = values

        saturation = next(
            (transition for transition in transitions if transition["saturation_transition"]),
            None,
        )
        report_sweeps.append(
            {
                "name": sweep["name"],
                "parameter": sweep.get("parameter") or sweep["name"],
                "benefit_metric": benefit_metric,
                "risk_metric": risk_metric,
                "points": [
                    {
                        **point,
                        "aggregate": aggregate_experiment(
                            raw_by_experiment[str(point["experiment"])]
                        ),
                    }
                    for point in points
                ],
                "transitions": transitions,
                "by_category": category_transitions,
                "saturation_detected": saturation is not None,
                "saturation_at": (
                    {
                        "from": saturation["previous_value"],
                        "to": saturation["current_value"],
                        "experiment": saturation["current"],
                    }
                    if saturation is not None
                    else None
                ),
            }
        )

    report = {
        "metrics": str(args.metrics),
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "alpha": args.alpha,
        "thresholds": {
            "min_gain": args.min_gain,
            "min_risk_increase": args.min_risk_increase,
        },
        "sweeps": report_sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
