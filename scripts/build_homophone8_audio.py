#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
CORE8 = (
    "exact_homophone",
    "near_homophone",
    "voicing",
    "long_vowel",
    "geminate",
    "moraic_nasal",
    "pitch_accent",
    "semantic_only",
)


def require_revision(value: str, label: str) -> str:
    value = value.strip()
    if not FULL_REVISION.fullmatch(value):
        raise SystemExit(f"{label} must be a full 40-character Hugging Face commit revision")
    return value


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return values
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    target_length = max(1, int(round(len(values) * target_rate / source_rate)))
    source_axis = np.linspace(0.0, 1.0, num=len(values), endpoint=False, dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, num=target_length, endpoint=False, dtype=np.float64)
    return np.interp(target_axis, source_axis, values).astype(np.float32)


def load_projection(repo_id: str, config: str, revision: str) -> Dataset:
    parts: list[Dataset] = []
    errors: list[str] = []
    for split in ("train", "validation", "test"):
        try:
            dataset = load_dataset(repo_id, config, split=split, revision=revision)
        except Exception as exc:  # split discovery is not stable across all dataset builders
            errors.append(f"{split}:{type(exc).__name__}:{exc}")
            continue
        if not len(dataset):
            continue
        if "benchmark_split" in dataset.column_names:
            dataset = dataset.remove_columns("benchmark_split")
        dataset = dataset.add_column("benchmark_split", [split] * len(dataset))
        parts.append(dataset)
    if not parts:
        raise RuntimeError(f"could not load any source split from {repo_id}/{config}: {errors}")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def deterministic_sample(dataset: Dataset, per_category: int) -> Dataset:
    by_category: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(dataset):
        category = str(row.get("category") or "")
        if category not in CORE8:
            continue
        by_category[category].append((str(row.get("id") or f"row-{index:08d}"), index))
    selected: list[int] = []
    for category in CORE8:
        values = sorted(by_category.get(category, []))
        selected.extend(index for _, index in values[:per_category])
    if not selected:
        raise RuntimeError("source config contains no core8 rows")
    return dataset.select(selected)


def category_counts(dataset: Dataset) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in dataset["category"]).items()))


def validate_gate_population(dataset: Dataset, required: tuple[str, ...], minimum: int) -> None:
    counts = category_counts(dataset)
    missing = {category: counts.get(category, 0) for category in required if counts.get(category, 0) < minimum}
    if missing:
        raise RuntimeError(
            f"source benchmark cannot satisfy audio coverage gate min={minimum}: {missing}; counts={counts}"
        )


def synthesize(dataset: Dataset, output_dir: Path, sample_rate: int) -> tuple[list[str], dict[str, Any]]:
    import pyopenjtalk

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    durations: list[float] = []
    for index, row in enumerate(dataset):
        row_id = str(row.get("id") or f"row-{index:08d}")
        text = str(row.get("text") or (row.get("target") or {}).get("surface") or "").strip()
        if not text:
            raise RuntimeError(f"row {row_id} has no synthesizable text")
        waveform, source_rate = pyopenjtalk.tts(text)
        audio = resample_linear(np.asarray(waveform), int(source_rate), sample_rate)
        path = output_dir / f"{index:05d}-{row_id}.wav"
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        paths.append(str(path.resolve()))
        durations.append(len(audio) / sample_rate)
    metadata = {
        "origin": "synthetic_tts",
        "engine": "pyopenjtalk-plus",
        "engine_version": version("pyopenjtalk-plus"),
        "voice": "mei_normal.htsvoice",
        "voice_license": "CC-BY-3.0",
        "sampling_rate": sample_rate,
        "duration_seconds": round(sum(durations), 6),
    }
    return paths, metadata


def add_audio_columns(
    dataset: Dataset,
    paths: list[str],
    *,
    source_config: str,
    source_revision: str,
    synth_meta: dict[str, Any],
) -> Dataset:
    size = len(dataset)
    result = dataset.add_column("audio", paths)
    result = result.cast_column("audio", Audio(sampling_rate=int(synth_meta["sampling_rate"])))
    flat_columns = {
        "audio_origin": synth_meta["origin"],
        "audio_engine": synth_meta["engine"],
        "audio_engine_version": synth_meta["engine_version"],
        "audio_voice": synth_meta["voice"],
        "audio_voice_license": synth_meta["voice_license"],
        "audio_source_config": source_config,
        "audio_source_revision": source_revision,
        "audio_sampling_rate": int(synth_meta["sampling_rate"]),
    }
    for name, value in flat_columns.items():
        if name in result.column_names:
            result = result.remove_columns(name)
        result = result.add_column(name, [value] * size)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and optionally publish the embedded-audio homophone8 evaluation projection"
    )
    parser.add_argument("--repo-id", default="saeeew/JP-HomophoneBench")
    parser.add_argument("--source-config", default="homophone8")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--config-name", default="homophone8-audio")
    parser.add_argument("--per-category", type=int, default=32)
    parser.add_argument("--min-exact", type=int, default=5)
    parser.add_argument("--min-near", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--work-dir", type=Path, default=Path("dist/homophone8-audio"))
    parser.add_argument("--summary", type=Path, default=Path("dist/homophone8-audio/summary.json"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    revision = require_revision(args.source_revision, "source revision")
    resolved = str(HfApi(token=args.token).dataset_info(args.repo_id, revision=revision).sha or "")
    if resolved != revision:
        raise RuntimeError(f"source revision mismatch: {resolved!r} != {revision!r}")
    if args.per_category < max(args.min_exact, args.min_near):
        raise SystemExit("--per-category must be >= the exact/near coverage minimum")

    source = load_projection(args.repo_id, args.source_config, revision)
    selected = deterministic_sample(source, args.per_category)
    validate_gate_population(selected, ("exact_homophone",), args.min_exact)
    validate_gate_population(selected, ("near_homophone",), args.min_near)

    paths, synth_meta = synthesize(selected, args.work_dir / "audio", args.sample_rate)
    audio_dataset = add_audio_columns(
        selected,
        paths,
        source_config=args.source_config,
        source_revision=revision,
        synth_meta=synth_meta,
    )
    output = DatasetDict({"test": audio_dataset})
    published_revision: str | None = None
    if args.publish:
        if not args.token:
            raise SystemExit("HF_TOKEN is required with --publish")
        output.push_to_hub(
            args.repo_id,
            config_name=args.config_name,
            token=args.token,
            max_shard_size="500MB",
            embed_external_files=True,
            commit_message=f"data: publish {args.config_name} embedded audio evaluation projection",
        )
        published_revision = str(HfApi(token=args.token).dataset_info(args.repo_id).sha or "")

    summary = {
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "source_config": args.source_config,
        "source_revision": revision,
        "published_revision": published_revision,
        "split": "test",
        "records": len(audio_dataset),
        "categories": category_counts(audio_dataset),
        "required_coverage": {
            "exact_homophone": args.min_exact,
            "near_homophone": args.min_near,
        },
        "synthesis": synth_meta,
        "projection_policy": {
            "source_splits": ["train", "validation", "test"],
            "published_split": "test",
            "original_split_column": "benchmark_split",
            "per_category_cap": args.per_category,
            "ordering": "category-order then row-id lexical order",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
