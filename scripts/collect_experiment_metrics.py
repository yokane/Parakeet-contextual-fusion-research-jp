#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from parakeet_context_fusion.metrics import cer, oracle_at_k, reciprocal_rank


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict):
        for key in ("results", "predictions", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]
        return [parsed]
    raise ValueError(f"unsupported result format in {path}")


def prediction_text(row: dict[str, Any]) -> str:
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict) and first.get("text") is not None:
            return str(first["text"])
        return str(first)
    for key in ("pred_text", "prediction", "hypothesis", "transcript"):
        if row.get(key) is not None:
            return str(row[key])
    pred_keys = sorted(key for key in row if key.startswith("pred_text"))
    for key in pred_keys:
        if row.get(key) is not None:
            return str(row[key])
    return ""


def ranked_texts(row: dict[str, Any]) -> list[str]:
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        values: list[str] = []
        for item in candidates:
            if isinstance(item, dict):
                values.append(str(item.get("text") or ""))
            else:
                values.append(str(item))
        return values
    prediction = prediction_text(row)
    return [prediction] if prediction else []


def target_and_distractors(benchmark: dict[str, Any]) -> tuple[str, list[str]]:
    target = str((benchmark.get("target") or {}).get("surface") or "")
    distractors: list[str] = []
    for candidate in benchmark.get("candidates") or []:
        surface = str((candidate or {}).get("surface") or "")
        if surface and surface != target and surface not in distractors:
            distractors.append(surface)
    return target, distractors


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    cer_values = [float(row["cer"]) for row in rows if row.get("cer") is not None]
    entity_rows = [row for row in rows if row.get("entity_correct") is not None]
    nbest_rows = [row for row in rows if row.get("mrr") is not None]
    negative_rows = [row for row in rows if row.get("is_negative")]
    result: dict[str, Any] = {
        "count": len(rows),
        "cer": mean(cer_values),
        "entity_accuracy": mean([1.0 if row["entity_correct"] else 0.0 for row in entity_rows]),
        "distractor_false_positive_rate": mean(
            [1.0 if row.get("distractor_false_positive") else 0.0 for row in entity_rows]
        ),
        "mrr": mean([float(row["mrr"]) for row in nbest_rows]),
        "negative_count": len(negative_rows),
        "bias_false_positive_rate": mean(
            [1.0 if row.get("bias_false_positive") else 0.0 for row in negative_rows]
        ),
    }
    for k in ks:
        values = [float(row[f"oracle_at_{k}"]) for row in nbest_rows if row.get(f"oracle_at_{k}") is not None]
        result[f"oracle_at_{k}"] = mean(values)
    return result


def paired_comparison(rows: list[dict[str, Any]], *, baseline: str) -> dict[str, Any]:
    by_experiment: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_experiment[str(row["experiment"])][str(row["benchmark_id"])] = row
    base = by_experiment.get(baseline, {})
    comparisons: dict[str, Any] = {}
    for experiment, items in sorted(by_experiment.items()):
        if experiment == baseline:
            continue
        paired = [(base[key], items[key]) for key in sorted(base.keys() & items.keys())]
        cer_deltas = [float(new["cer"]) - float(old["cer"]) for old, new in paired if old.get("cer") is not None and new.get("cer") is not None]
        entity_deltas = [
            (1 if new.get("entity_correct") else 0) - (1 if old.get("entity_correct") else 0)
            for old, new in paired
            if old.get("entity_correct") is not None and new.get("entity_correct") is not None
        ]
        comparisons[experiment] = {
            "paired_count": len(paired),
            "mean_cer_delta": mean(cer_deltas),
            "mean_entity_accuracy_delta": mean([float(value) for value in entity_deltas]),
            "entity_wins": sum(value > 0 for value in entity_deltas),
            "entity_losses": sum(value < 0 for value in entity_deltas),
            "entity_ties": sum(value == 0 for value in entity_deltas),
        }
    return comparisons


def parse_result_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be EXPERIMENT=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--result must be EXPERIMENT=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E00-E06 results into category-aware Parquet metrics")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, type=parse_result_spec)
    parser.add_argument("--parquet", type=Path, default=Path("results/metrics.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--baseline", default="E00")
    parser.add_argument("--oracle-k", type=int, action="append", default=[])
    args = parser.parse_args()
    ks = sorted(set(args.oracle_k or [1, 4, 8, 16, 32]))

    benchmark_rows = read_records(args.benchmark)
    benchmark_by_id = {str(row.get("id")): row for row in benchmark_rows if row.get("id") is not None}
    output_rows: list[dict[str, Any]] = []

    for experiment, result_path in args.result:
        results = read_records(result_path)
        for position, result in enumerate(results):
            benchmark_id = result.get("benchmark_id") or result.get("id")
            benchmark: dict[str, Any] | None = None
            match_mode = "id"
            if benchmark_id is not None:
                benchmark = benchmark_by_id.get(str(benchmark_id))
            if benchmark is None and position < len(benchmark_rows):
                benchmark = benchmark_rows[position]
                benchmark_id = benchmark.get("id")
                match_mode = "position"
            if benchmark is None:
                continue

            reference = str(benchmark.get("text") or "")
            prediction = prediction_text(result)
            ranked = ranked_texts(result)
            target, distractors = target_and_distractors(benchmark)
            target_present = bool(target and target in prediction)
            distractor_hits = [value for value in distractors if value in prediction]
            metadata = benchmark.get("metadata") or {}
            is_negative = bool(metadata.get("negative") or metadata.get("negative_context") or metadata.get("target_present") is False)
            bias_phrases = [target, *distractors]
            bias_false_positive = bool(is_negative and any(phrase and phrase in prediction for phrase in bias_phrases))
            row: dict[str, Any] = {
                "experiment": experiment,
                "benchmark_id": str(benchmark_id),
                "match_mode": match_mode,
                "category": str(benchmark.get("category") or "unknown"),
                "group_id": str(benchmark.get("group_id") or ""),
                "reference": reference,
                "prediction": prediction,
                "target_surface": target,
                "cer": cer(reference, prediction) if reference else None,
                "entity_correct": target_present if target else None,
                "distractor_false_positive": bool(distractor_hits),
                "distractor_hits": distractor_hits,
                "mrr": reciprocal_rank(target, ranked) if target and ranked else None,
                "nbest_size": len(ranked),
                "is_negative": is_negative,
                "bias_false_positive": bias_false_positive if is_negative else None,
            }
            for k in ks:
                row[f"oracle_at_{k}"] = oracle_at_k(target, ranked, k) if target and ranked else None
            output_rows.append(row)

    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(output_rows)
    pq.write_table(table, args.parquet, compression="zstd")

    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_experiment_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        by_experiment[str(row["experiment"])].append(row)
        by_experiment_category[(str(row["experiment"]), str(row["category"]))].append(row)

    summary = {
        "benchmark": str(args.benchmark),
        "rows": len(output_rows),
        "oracle_k": ks,
        "experiments": {
            experiment: {
                "overall": aggregate(rows, ks),
                "by_category": {
                    category: aggregate(by_experiment_category[(experiment, category)], ks)
                    for category in sorted({str(row["category"]) for row in rows})
                },
            }
            for experiment, rows in sorted(by_experiment.items())
        },
        "paired_vs_baseline": paired_comparison(output_rows, baseline=args.baseline),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
