#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from parakeet_context_fusion.benchmark import (
    CORE_CATEGORIES,
    canonical_key,
    classify_readings,
    difficulty_vector,
    make_group_id,
    stable_split,
)
from parakeet_context_fusion.japanese_g2p import ensure_reading, reading_to_phones

SOURCES = {
    "mondegreen": {"repo_id": "NagaYu/mondegreen-asr-errors", "license": "cc0-1.0", "config": None},
    "hoiku": {"repo_id": "IDEMITSU/hoiku-yougo-stt-ja", "license": "cc-by-nc-4.0", "config": None},
    "prosodic_abx": {"repo_id": "HaitongSUN/prosodic-abx", "license": "cc-by-4.0", "config": "japanese_pitch_accent"},
}

MONDEGREEN_MAP = {
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


def source_revision(repo_id: str) -> str | None:
    try:
        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


def _load_single_format_fallback(repo_id: str, split: str) -> Dataset:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [
        path for path in files
        if split.lower() in path.lower()
        and Path(path).suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}
    ]
    if not candidates:
        raise RuntimeError(f"cannot locate {split!r} data file in {repo_id}")
    order = {".jsonl": 0, ".json": 1, ".parquet": 2, ".csv": 3}
    candidates.sort(key=lambda path: (order.get(Path(path).suffix.lower(), 99), path))
    filename = candidates[0]
    local = hf_hub_download(repo_id, filename, repo_type="dataset")
    suffix = Path(filename).suffix.lower()
    builder = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}[suffix]
    return load_dataset(builder, data_files={split: local}, split=split)


def load_split(repo_id: str, *, config: str | None = None, split: str = "train") -> Dataset:
    try:
        return load_dataset(repo_id, config, split=split)
    except Exception:
        return _load_single_format_fallback(repo_id, split)


def source_meta(*, source_name: str, split: str, row_id: str | int, revision: str | None, synthetic: bool) -> dict[str, Any]:
    spec = SOURCES[source_name]
    return {
        "dataset": spec["repo_id"],
        "config": spec["config"],
        "split": split,
        "source_id": str(row_id),
        "revision": revision,
        "license": spec["license"],
        "synthetic": synthetic,
        "audio_redistributed": False,
    }


def pair_record(
    *,
    record_id: str,
    text: str,
    target_surface: str,
    target_reading: str,
    candidate_surface: str,
    candidate_reading: str,
    category: str,
    source: dict[str, Any],
    near_threshold: float,
    context_required: bool = False,
    target_extra: dict[str, Any] | None = None,
    candidate_extra: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_phones = reading_to_phones(target_reading)
    candidate_phones = reading_to_phones(candidate_reading)
    relation = classify_readings(target_reading, candidate_reading, near_threshold=near_threshold)
    phone_distance = relation.phone_distance
    group_id = make_group_id(target_reading, candidate_reading)
    target = {
        "surface": target_surface,
        "reading": target_reading,
        "phones": list(target_phones),
        **(target_extra or {}),
    }
    candidate = {
        "surface": candidate_surface,
        "reading": candidate_reading,
        "phones": list(candidate_phones),
        "relation": category if category in CORE_CATEGORIES else relation.category,
        "phone_distance": phone_distance,
        **(candidate_extra or {}),
    }
    source_dataset = str(source["dataset"])
    audio_kind = "hf_row" if "/" in source_dataset else "none"
    return {
        "id": record_id,
        "group_id": group_id,
        "split": stable_split(group_id),
        "audio_filepath": None,
        "audio_ref": {
            "kind": audio_kind,
            "repo_id": source_dataset if audio_kind == "hf_row" else None,
            "config": source.get("config"),
            "split": source.get("split"),
            "row_id": source.get("source_id"),
        },
        "text": text,
        "target": target,
        "candidates": [{**target, "relation": "target", "phone_distance": 0.0}, candidate],
        "category": category,
        "source": source,
        "difficulty": difficulty_vector(
            category=category,
            phone_distance=phone_distance,
            candidate_count=2,
            has_context=bool(text and text != target_surface),
        ),
        "metadata": {
            "derived_relation": relation.category,
            "relation_reason": relation.reason,
            "context_required": context_required,
            **(metadata or {}),
        },
    }


def iter_mondegreen(limit: int | None, near_threshold: float) -> Iterator[dict[str, Any]]:
    spec = SOURCES["mondegreen"]
    revision = source_revision(spec["repo_id"])
    dataset = load_split(spec["repo_id"], split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    for index, raw in enumerate(dataset):
        row = dict(raw)
        gold = str(first(row, "gold", "reference", "text", default="")).strip()
        hypothesis = str(first(row, "hypothesis", "asr", "corrupted", default="")).strip()
        if not gold or not hypothesis or gold == hypothesis:
            continue
        labels = first(row, "error_types", "error_type", "category", default=[])
        if isinstance(labels, str):
            labels = [labels]
        mapped = [MONDEGREEN_MAP[label] for label in labels if label in MONDEGREEN_MAP]
        if not mapped:
            continue
        category = mapped[0]
        gold_reading = ensure_reading(gold, None)
        hyp_reading = ensure_reading(hypothesis, None)
        if not gold_reading or not hyp_reading:
            continue
        source = source_meta(
            source_name="mondegreen",
            split=str(row.get("split") or "train"),
            row_id=first(row, "id", default=index),
            revision=revision,
            synthetic=str(row.get("provenance", "simulated")) != "measured",
        )
        yield pair_record(
            record_id=f"mondegreen-{first(row, 'id', default=index)}",
            text=gold,
            target_surface=gold,
            target_reading=gold_reading,
            candidate_surface=hypothesis,
            candidate_reading=hyp_reading,
            category=category,
            source=source,
            near_threshold=near_threshold,
            metadata={
                "source_error_types": labels,
                "speaker": row.get("speaker"),
                "snr_db": row.get("snr_db"),
                "room": row.get("room"),
                "source_corpus": row.get("source_corpus"),
            },
        )


def _flatten_hoiku(dataset: Dataset) -> Iterator[tuple[int, dict[str, Any]]]:
    counter = 0
    for raw in dataset:
        row = dict(raw)
        if isinstance(row.get("terms"), list):
            for term in row["terms"]:
                yield counter, dict(term)
                counter += 1
        else:
            yield counter, row
            counter += 1


def iter_hoiku(limit: int | None, near_threshold: float) -> Iterator[dict[str, Any]]:
    spec = SOURCES["hoiku"]
    revision = source_revision(spec["repo_id"])
    dataset = load_split(spec["repo_id"], split="train")
    emitted_terms = 0
    for index, row in _flatten_hoiku(dataset):
        if limit is not None and emitted_terms >= limit:
            break
        term = str(first(row, "term", "surface", default="")).strip()
        reading = ensure_reading(term, first(row, "reading", "yomi"))
        if not term or not reading:
            continue
        emitted_terms += 1
        contexts = first(row, "contexts", "example_sentences", default=[])
        if isinstance(contexts, str):
            contexts = [contexts]
        mis = first(row, "mis_conversions", "wrong_candidates", default=[])
        if isinstance(mis, str):
            mis = [mis]
        for candidate_index, candidate_surface in enumerate(mis):
            candidate_surface = str(candidate_surface).strip()
            candidate_reading = ensure_reading(candidate_surface, None)
            if not candidate_surface or not candidate_reading:
                continue
            relation = classify_readings(reading, candidate_reading, near_threshold=near_threshold)
            if relation.category == "unrelated":
                continue
            source = source_meta(source_name="hoiku", split="train", row_id=index, revision=revision, synthetic=False)
            context = str(contexts[0]) if contexts else term
            base_id = f"hoiku-{index:04d}-{candidate_index:02d}"
            yield pair_record(
                record_id=base_id,
                text=term,
                target_surface=term,
                target_reading=reading,
                candidate_surface=candidate_surface,
                candidate_reading=candidate_reading,
                category=relation.category,
                source=source,
                near_threshold=near_threshold,
                metadata={"domain_category": row.get("category"), "priority": row.get("priority"), "all_contexts": contexts},
            )
            if relation.category == "exact_homophone" and contexts:
                yield pair_record(
                    record_id=f"{base_id}-semantic",
                    text=context,
                    target_surface=term,
                    target_reading=reading,
                    candidate_surface=candidate_surface,
                    candidate_reading=candidate_reading,
                    category="semantic_only",
                    source=source,
                    near_threshold=near_threshold,
                    context_required=True,
                    metadata={"domain_category": row.get("category"), "priority": row.get("priority"), "all_contexts": contexts},
                )


def iter_prosodic_abx(limit: int | None, near_threshold: float) -> Iterator[dict[str, Any]]:
    del near_threshold
    spec = SOURCES["prosodic_abx"]
    revision = source_revision(spec["repo_id"])
    dataset = load_split(spec["repo_id"], config=spec["config"], split="train")
    if "audio" in dataset.column_names:
        dataset = dataset.remove_columns("audio")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(dataset):
        row = dict(raw)
        row["_row_index"] = index
        groups[(str(row.get("speaker", "")), str(row.get("context_id", "")))].append(row)

    for (speaker, context_id), rows in groups.items():
        if len(rows) < 2:
            continue
        for row in rows:
            target_surface = str(row.get("target", "")).strip()
            target_reading = ensure_reading(target_surface, row.get("surface_kana"))
            if not target_surface or not target_reading:
                continue
            competitors = [
                other for other in rows
                if other is not row and other.get("label") != row.get("label") and str(other.get("target", "")).strip()
            ]
            if not competitors:
                continue
            other = competitors[0]
            candidate_surface = str(other["target"]).strip()
            candidate_reading = ensure_reading(candidate_surface, other.get("surface_kana"))
            if not candidate_reading:
                continue
            source = source_meta(
                source_name="prosodic_abx",
                split="train",
                row_id=row.get("id", row["_row_index"]),
                revision=revision,
                synthetic=False,
            )
            record = pair_record(
                record_id=f"prosodic-{row.get('id', row['_row_index'])}",
                text=str(row.get("text", target_surface)),
                target_surface=target_surface,
                target_reading=target_reading,
                candidate_surface=candidate_surface,
                candidate_reading=candidate_reading,
                category="pitch_accent",
                source=source,
                near_threshold=1.0,
                target_extra={
                    "start_sec": row.get("target_onset"),
                    "end_sec": row.get("target_offset"),
                    "pitch_accent": row.get("label"),
                },
                candidate_extra={"pitch_accent": other.get("label")},
                metadata={
                    "speaker": speaker,
                    "context_id": context_id,
                    "paired_source_id": other.get("id"),
                    "sample_rate": 48000,
                },
            )
            record["group_id"] = make_group_id("prosody", speaker, context_id)
            record["split"] = stable_split(record["group_id"])
            yield record


def iter_semantic_tsv(path: Path, near_threshold: float) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"target_surface", "target_reading", "candidate_surface", "candidate_reading", "context"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"semantic TSV missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            relation = classify_readings(row["target_reading"], row["candidate_reading"], near_threshold=near_threshold)
            if relation.category != "exact_homophone":
                raise ValueError(f"semantic TSV row {index + 2} is not an exact homophone: {relation}")
            source = {
                "dataset": row.get("source_name") or "manual-semantic-tsv",
                "config": None,
                "split": "manual",
                "source_id": str(index),
                "revision": row.get("source_revision") or None,
                "license": row.get("source_license") or "cc0-1.0",
                "synthetic": False,
                "audio_redistributed": False,
            }
            yield pair_record(
                record_id=f"exact-manual-{index:06d}",
                text=row["target_surface"],
                target_surface=row["target_surface"],
                target_reading=row["target_reading"],
                candidate_surface=row["candidate_surface"],
                candidate_reading=row["candidate_reading"],
                category="exact_homophone",
                source=source,
                near_threshold=near_threshold,
                metadata={"notes": row.get("notes") or None},
            )
            yield pair_record(
                record_id=f"semantic-manual-{index:06d}",
                text=row["context"],
                target_surface=row["target_surface"],
                target_reading=row["target_reading"],
                candidate_surface=row["candidate_surface"],
                candidate_reading=row["candidate_reading"],
                category="semantic_only",
                source=source,
                near_threshold=near_threshold,
                context_required=True,
                metadata={"notes": row.get("notes") or None},
            )


def write_release(records: Iterable[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = canonical_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    deduped.sort(key=lambda item: item["id"])

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in deduped:
        split_rows[row["split"]].append(row)

    hashes: dict[str, str] = {}
    for split, rows in split_rows.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    all_path = output_dir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for row in deduped:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    hashes[all_path.name] = hashlib.sha256(all_path.read_bytes()).hexdigest()

    categories = Counter(row["category"] for row in deduped)
    sources = Counter(row["source"]["dataset"] for row in deduped)
    licenses = Counter(row["source"].get("license", "unknown") for row in deduped)
    stats = {
        "records": len(deduped),
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "categories": dict(sorted(categories.items())),
        "sources": dict(sorted(sources.items())),
        "licenses": dict(sorted(licenses.items())),
        "core8_missing": [category for category in CORE_CATEGORIES if not categories[category]],
        "sha256": hashes,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic JP-HomophoneBench JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=sorted(SOURCES), help="repeatable; defaults to all supported sources")
    parser.add_argument("--semantic-tsv", type=Path)
    parser.add_argument("--limit", type=int, help="per-source development limit")
    parser.add_argument("--near-threshold", type=float, default=1.0)
    parser.add_argument("--require-core8", action="store_true")
    args = parser.parse_args()

    names = args.source or ["mondegreen", "hoiku", "prosodic_abx"]
    records: list[dict[str, Any]] = []
    for name in names:
        if name == "mondegreen":
            records.extend(iter_mondegreen(args.limit, args.near_threshold))
        elif name == "hoiku":
            records.extend(iter_hoiku(args.limit, args.near_threshold))
        elif name == "prosodic_abx":
            records.extend(iter_prosodic_abx(args.limit, args.near_threshold))
    if args.semantic_tsv:
        records.extend(iter_semantic_tsv(args.semantic_tsv, args.near_threshold))

    stats = write_release(records, args.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.require_core8 and stats["core8_missing"]:
        raise SystemExit(f"missing core8 categories: {stats['core8_missing']}")


if __name__ == "__main__":
    main()
