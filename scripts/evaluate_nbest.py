#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from parakeet_context_fusion.metrics import cer, oracle_at_k, reciprocal_rank


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-k", type=int, action="append", default=[])
    args = parser.parse_args()
    ks = sorted(set(args.oracle_k or [1, 4, 8, 16, 32]))
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    overall: list[dict[str, object]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidates = row.get("candidates", [])
        ranked = [str(item["text"]) for item in candidates]
        top = ranked[0] if ranked else ""
        target_obj = row.get("target") or {}
        target = str(target_obj.get("surface", ""))
        reference = str(row.get("text", ""))
        item = {
            "id": row.get("id"),
            "category": row.get("category", "unknown"),
            "cer": cer(reference, top) if reference else None,
            "entity_correct": bool(target and target in top),
            "mrr": reciprocal_rank(target, ranked) if target else None,
            "oracle": {str(k): oracle_at_k(target, ranked, k) if target else None for k in ks},
        }
        overall.append(item)
        groups[str(item["category"])].append(item)

    def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        cer_values = [float(row["cer"]) for row in rows if row["cer"] is not None]
        entity_rows = [row for row in rows if row["mrr"] is not None]
        return {
            "count": len(rows),
            "cer": mean(cer_values),
            "entity_accuracy": mean([float(bool(row["entity_correct"])) for row in entity_rows]),
            "mrr": mean([float(row["mrr"]) for row in entity_rows]),
            "oracle_at_k": {str(k): mean([float(bool(row["oracle"][str(k)])) for row in entity_rows]) for k in ks},
        }

    report = {
        "overall": aggregate(overall),
        "by_category": {category: aggregate(rows) for category, rows in sorted(groups.items())},
        "per_item": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_category": report["by_category"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
