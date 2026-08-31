#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from huggingface_hub import HfApi

PREFERRED_AUDIO_COLUMNS = ("audio", "speech", "wav")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or "sample"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_phrases(row: dict[str, Any]) -> set[str]:
    phrases: set[str] = set()
    target = row.get("target") or {}
    surface = str(target.get("surface") or "").strip()
    if surface:
        phrases.add(surface)
    for candidate in row.get("candidates") or []:
        value = str((candidate or {}).get("surface") or "").strip()
        if value:
            phrases.add(value)
    return phrases


def detect_audio_column(dataset: Dataset) -> str | None:
    for name in PREFERRED_AUDIO_COLUMNS:
        feature = dataset.features.get(name)
        if isinstance(feature, Audio):
            return name
    for name, feature in dataset.features.items():
        if isinstance(feature, Audio):
            return name
    return None


class SourceDatasetCache:
    def __init__(self) -> None:
        self.datasets: dict[tuple[str, str | None, str, str | None], tuple[Dataset, str]] = {}
        self.id_maps: dict[tuple[str, str | None, str, str | None], dict[str, int]] = {}
        self.revisions: dict[str, str | None] = {}

    def _load(
        self,
        *,
        repo_id: str,
        config: str | None,
        split: str,
        revision: str | None,
    ) -> tuple[Dataset, str]:
        key = (repo_id, config, split, revision)
        if key in self.datasets:
            return self.datasets[key]
        dataset = load_dataset(repo_id, config, split=split, revision=revision)
        audio_column = detect_audio_column(dataset)
        if audio_column is None:
            raise RuntimeError(f"no Audio feature found in {repo_id} config={config!r} split={split!r}")
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
        self.datasets[key] = (dataset, audio_column)
        try:
            self.revisions[repo_id] = HfApi().dataset_info(repo_id, revision=revision).sha
        except Exception:
            self.revisions[repo_id] = revision
        return dataset, audio_column

    def _index_map(
        self,
        key: tuple[str, str | None, str, str | None],
        dataset: Dataset,
    ) -> dict[str, int]:
        cached = self.id_maps.get(key)
        if cached is not None:
            return cached
        mapping: dict[str, int] = {}
        for column in ("id", "uid", "sample_id", "utterance_id"):
            if column not in dataset.column_names:
                continue
            for index, value in enumerate(dataset[column]):
                mapping.setdefault(str(value), index)
            if mapping:
                break
        self.id_maps[key] = mapping
        return mapping

    def source_row(self, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
        audio_ref = row.get("audio_ref") or {}
        source = row.get("source") or {}
        repo_id = str(audio_ref.get("repo_id") or source.get("dataset") or "").strip()
        if not repo_id or "/" not in repo_id:
            raise RuntimeError("row has no rehydratable Hugging Face source repository")
        config = audio_ref.get("config") or source.get("config") or None
        split = str(audio_ref.get("split") or source.get("split") or "train")
        revision = source.get("revision") or None
        row_id = str(audio_ref.get("row_id") or source.get("source_id") or "")
        key = (repo_id, config, split, revision)
        dataset, audio_column = self._load(
            repo_id=repo_id,
            config=config,
            split=split,
            revision=revision,
        )

        index: int | None = None
        if row_id.isdigit():
            candidate = int(row_id)
            if 0 <= candidate < len(dataset):
                index = candidate
        if index is None:
            index = self._index_map(key, dataset).get(row_id)
        if index is None:
            raise RuntimeError(f"cannot resolve source row_id={row_id!r} in {repo_id}")
        return dict(dataset[index]), audio_column


def copy_audio(value: Any, *, destination_stem: Path) -> tuple[Path, float]:
    if not isinstance(value, dict):
        raise RuntimeError(f"decode=False Audio value is not a mapping: {type(value)!r}")
    source_path = value.get("path")
    payload = value.get("bytes")
    suffix = Path(str(source_path)).suffix if source_path else ".wav"
    if not suffix:
        suffix = ".wav"
    destination = destination_stem.with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source_path and Path(str(source_path)).exists():
        shutil.copyfile(str(source_path), destination)
    elif payload:
        destination.write_bytes(bytes(payload))
    else:
        raise RuntimeError("Audio feature exposes neither an existing path nor bytes")

    try:
        info = sf.info(str(destination))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        if not payload:
            payload = Path(str(source_path)).read_bytes() if source_path else None
        if not payload:
            raise
        audio, sample_rate = sf.read(io.BytesIO(payload), always_2d=False)
        destination = destination_stem.with_suffix(".wav")
        sf.write(str(destination), audio, sample_rate)
        info = sf.info(str(destination))
        duration = float(info.frames) / float(info.samplerate)
    return destination.resolve(), duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize JP-HomophoneBench into NeMo evaluation inputs"
    )
    parser.add_argument("--repo-id", default="saeeew/JP-HomophoneBench")
    parser.add_argument("--config", default="homophone8")
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="repeatable; defaults to test",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument(
        "--rehydrate-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()

    splits = args.splits or ["test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    benchmark_rows: list[dict[str, Any]] = []
    phrases: set[str] = set()
    corpus: list[str] = []
    published_revision: str | None = None
    try:
        published_revision = HfApi().dataset_info(args.repo_id).sha
    except Exception:
        pass

    for split in splits:
        dataset = load_dataset(args.repo_id, args.config, split=split)
        for raw in dataset:
            row = dict(raw)
            row["hf_publication"] = {
                "repo_id": args.repo_id,
                "config": args.config,
                "split": split,
                "revision": published_revision,
            }
            benchmark_rows.append(row)
            phrases.update(row_phrases(row))
            text = str(row.get("text") or "").strip()
            if text:
                corpus.append(text)

    index_path = args.output_dir / "bench_index.jsonl"
    context_path = args.output_dir / "context_phrases.txt"
    corpus_path = args.output_dir / "lm_corpus.txt"
    nemo_path = args.output_dir / "nemo_eval.jsonl"
    provenance_path = args.output_dir / "eval_provenance.json"
    write_jsonl(index_path, benchmark_rows)
    context_path.write_text(
        "\n".join(sorted(phrases)) + ("\n" if phrases else ""),
        encoding="utf-8",
    )
    corpus_path.write_text(
        "\n".join(corpus) + ("\n" if corpus else ""),
        encoding="utf-8",
    )

    source_cache = SourceDatasetCache()
    nemo_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if args.rehydrate_audio:
        for row in benchmark_rows:
            bench_id = str(row.get("id") or "")
            audio_ref = row.get("audio_ref") or {}
            if audio_ref.get("kind") != "hf_row":
                skipped.append({"id": bench_id, "reason": "no_hf_audio_ref"})
                continue
            try:
                source_row, audio_column = source_cache.source_row(row)
                local_audio, duration = copy_audio(
                    source_row[audio_column],
                    destination_stem=audio_dir / safe_name(bench_id),
                )
            except Exception as exc:
                skipped.append(
                    {
                        "id": bench_id,
                        "reason": f"rehydrate_error:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            target = row.get("target") or {}
            candidates = row.get("candidates") or []
            source = row.get("source") or {}
            nemo_rows.append(
                {
                    "audio_filepath": str(local_audio),
                    "duration": duration,
                    "text": str(row.get("text") or ""),
                    "benchmark_id": bench_id,
                    "group_id": row.get("group_id"),
                    "category": row.get("category"),
                    "target_surface": target.get("surface"),
                    "target_reading": target.get("reading"),
                    "candidate_surfaces": [
                        item.get("surface") for item in candidates if item.get("surface")
                    ],
                    "source_dataset": source.get("dataset"),
                    "source_license": source.get("license"),
                }
            )
    write_jsonl(nemo_path, nemo_rows)

    category_counts = Counter(str(row.get("category") or "unknown") for row in benchmark_rows)
    runnable_ids = {str(item["benchmark_id"]) for item in nemo_rows}
    runnable_category_counts = Counter(
        str(row.get("category") or "unknown")
        for row in benchmark_rows
        if str(row.get("id") or "") in runnable_ids
    )
    provenance = {
        "repo_id": args.repo_id,
        "config": args.config,
        "splits": splits,
        "published_revision": published_revision,
        "records": len(benchmark_rows),
        "runnable_audio_records": len(nemo_rows),
        "categories": dict(sorted(category_counts.items())),
        "runnable_categories": dict(sorted(runnable_category_counts.items())),
        "context_phrases": len(phrases),
        "rehydrate_audio": args.rehydrate_audio,
        "source_revisions": dict(sorted(source_cache.revisions.items())),
        "skipped": skipped,
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_audio and not nemo_rows:
        raise SystemExit("no runnable audio rows were materialized")


if __name__ == "__main__":
    main()
