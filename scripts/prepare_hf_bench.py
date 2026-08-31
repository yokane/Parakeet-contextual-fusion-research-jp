#!/usr/bin/env python3
"""Legacy lightweight HF lexical fixture adapter.

For new experiments prefer build_jp_homophone_bench.py, which implements the fixed
core8 taxonomy, group-aware splitting, provenance, validation, and publication.
This script intentionally emits a small exploratory JSONL rather than a frozen
JP-HomophoneBench release.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASETS = ["NagaYu/mondegreen-asr-errors", "IDEMITSU/hoiku-yougo-stt-ja"]
ERROR_MAP = {
    "voicing": "voicing",
    "long-vowel": "long_vowel",
    "long_vowel": "long_vowel",
    "geminate": "geminate",
    "moraic-nasal": "moraic_nasal",
    "moraic_nasal": "moraic_nasal",
    "term-phonetic": "near_homophone",
    "term_phonetic": "near_homophone",
}


def first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def records_mondegreen(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        error_type = str(first(row, "error_type", "category", "type", default="term-phonetic"))
        gold = str(first(row, "gold", "reference", "target", "text", default=""))
        hypothesis = str(first(row, "hypothesis", "asr", "corrupted", default=""))
        if not gold:
            continue
        yield {
            "id": f"mondegreen-{index:06d}",
            "text": gold,
            "hypothesis": hypothesis,
            "category": ERROR_MAP.get(error_type, "near_homophone"),
            "source_dataset": "NagaYu/mondegreen-asr-errors",
            "synthetic": True,
            "metadata": {"error_type": error_type},
        }


def records_hoiku(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        term = str(first(row, "term", "word", "surface", default=""))
        if not term:
            continue
        reading = first(row, "reading", "yomi")
        mistakes = first(row, "mis_conversions", "misconversions", "wrong_candidates", default=[])
        if isinstance(mistakes, str):
            mistakes = [mistakes]
        contexts = first(row, "contexts", "examples", "example_sentences", default=[])
        if isinstance(contexts, str):
            contexts = [contexts]
        yield {
            "id": f"hoiku-{index:06d}",
            "text": contexts[0] if contexts else term,
            "target": term,
            "reading": reading,
            "candidates": [str(item) for item in mistakes],
            "category": "exact_homophone",
            "source_dataset": "IDEMITSU/hoiku-yougo-stt-ja",
            "synthetic": False,
        }


def iter_dataset(name: str, limit: int | None) -> Iterable[dict[str, Any]]:
    ds = load_dataset(name, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    rows = (dict(row) for row in ds)
    if name == "NagaYu/mondegreen-asr-errors":
        return records_mondegreen(rows)
    if name == "IDEMITSU/hoiku-yougo-stt-ja":
        return records_hoiku(rows)
    raise ValueError(f"No adapter implemented for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    names = args.datasets or DEFAULT_DATASETS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for name in names:
            for record in iter_dataset(name, args.limit):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} exploratory records to {args.output}")


if __name__ == "__main__":
    main()
