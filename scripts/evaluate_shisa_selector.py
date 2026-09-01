#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from parakeet_context_fusion.metrics import cer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selector_rows = [row for row in rows if row.get("selector_parse_ok") is not None]
    entity_rows = [row for row in rows if row.get("selected_entity_correct") is not None]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    prompt_tokens = [float(row["prompt_tokens"]) for row in rows if row.get("prompt_tokens") is not None]
    generated_tokens = [float(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None]
    peak_vram = [int(row["peak_vram_bytes"]) for row in rows if row.get("peak_vram_bytes") is not None]
    return {
        "count": len(rows),
        "selected_cer": mean([float(row["selected_cer"]) for row in rows if row.get("selected_cer") is not None]),
        "source_cer": mean([float(row["source_cer"]) for row in rows if row.get("source_cer") is not None]),
        "selected_entity_accuracy": mean(
            [1.0 if row["selected_entity_correct"] else 0.0 for row in entity_rows]
        ),
        "source_entity_accuracy": mean(
            [1.0 if row["source_entity_correct"] else 0.0 for row in entity_rows]
        ),
        "changed_rate": mean([1.0 if row["changed"] else 0.0 for row in selector_rows]),
        "parse_failure_rate": mean([0.0 if row["selector_parse_ok"] else 1.0 for row in selector_rows]),
        "fallback_rate": mean([1.0 if row["fallback"] else 0.0 for row in selector_rows]),
        "entity_wins": sum(bool(row.get("entity_win")) for row in entity_rows),
        "entity_losses": sum(bool(row.get("entity_loss")) for row in entity_rows),
        "entity_ties": sum(bool(row.get("entity_tie")) for row in entity_rows),
        "cer_improved": sum(bool(row.get("cer_improved")) for row in rows),
        "cer_damaged": sum(bool(row.get("cer_damaged")) for row in rows),
        "cer_unchanged": sum(bool(row.get("cer_unchanged")) for row in rows),
        "mean_latency_ms": mean(latencies),
        "mean_prompt_tokens": mean(prompt_tokens),
        "mean_generated_tokens": mean(generated_tokens),
        "max_peak_vram_bytes": max(peak_vram) if peak_vram else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E07a Shisa N-best selection against JP-HomophoneBench")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="E07a JSONL")
    parser.add_argument("--parquet", type=Path, default=Path("results/E07a_metrics.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("results/E07a_summary.json"))
    args = parser.parse_args()

    benchmark = {str(row["id"]): row for row in read_jsonl(args.benchmark)}
    output: list[dict[str, Any]] = []
    for result in read_jsonl(args.input):
        benchmark_id = str(result.get("benchmark_id") or result.get("id") or "")
        bench = benchmark.get(benchmark_id)
        if bench is None:
            raise RuntimeError(f"E07a row does not match benchmark ID: {benchmark_id!r}")
        selector = result.get("selector") or {}
        runtime = selector.get("runtime") or {}
        source_text = str(selector.get("source_top1_text") or "")
        selected_text = str(result.get("selector_selected_text") or source_text)
        reference = str(bench.get("text") or "")
        target = str((bench.get("target") or {}).get("surface") or "")
        source_cer = cer(reference, source_text) if reference else None
        selected_cer = cer(reference, selected_text) if reference else None
        source_entity_correct = bool(target and target in source_text) if target else None
        selected_entity_correct = bool(target and target in selected_text) if target else None
        row = {
            "benchmark_id": benchmark_id,
            "category": str(bench.get("category") or "unknown"),
            "source_top1": source_text,
            "selected_text": selected_text,
            "reference": reference,
            "target_surface": target,
            "source_cer": source_cer,
            "selected_cer": selected_cer,
            "source_entity_correct": source_entity_correct,
            "selected_entity_correct": selected_entity_correct,
            "changed": selected_text != source_text,
            "selector_parse_ok": bool(selector.get("parse_ok")),
            "fallback": bool(selector.get("fallback_to_source_top1")),
            "entity_win": source_entity_correct is False and selected_entity_correct is True,
            "entity_loss": source_entity_correct is True and selected_entity_correct is False,
            "entity_tie": source_entity_correct == selected_entity_correct,
            "cer_improved": source_cer is not None and selected_cer is not None and selected_cer < source_cer,
            "cer_damaged": source_cer is not None and selected_cer is not None and selected_cer > source_cer,
            "cer_unchanged": source_cer is not None and selected_cer is not None and selected_cer == source_cer,
            "selected_original_index": selector.get("selected_original_index"),
            "prompt_sha256": selector.get("prompt_sha256"),
            "model": selector.get("model"),
            "revision": selector.get("revision"),
            "latency_ms": runtime.get("latency_ms"),
            "prompt_tokens": runtime.get("prompt_tokens"),
            "generated_tokens": runtime.get("generated_tokens"),
            "peak_vram_bytes": runtime.get("peak_vram_bytes"),
            "transformers_version": runtime.get("transformers_version"),
            "torch_version": runtime.get("torch_version"),
        }
        output.append(row)

    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output), args.parquet, compression="zstd")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[str(row["category"])].append(row)
    summary = {
        "experiment": "E07a",
        "overall": aggregate(output),
        "by_category": {category: aggregate(rows) for category, rows in sorted(grouped.items())},
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
