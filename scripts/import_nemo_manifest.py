#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from parakeet_context_fusion.benchmark import make_group_id, stable_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Existing NeMo JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="Unified benchmark JSONL")
    parser.add_argument("--category", default="general")
    parser.add_argument("--dataset-name", default="local-nemo-manifest")
    parser.add_argument("--license", default="local-evaluation-only")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio_filepath") or row.get("audio_filename")
            text = str(row.get("text", ""))
            if not audio:
                raise ValueError(f"row {index}: missing audio_filepath")
            record_id = str(row.get("benchmark_id", row.get("id", f"local-{index:06d}")))
            group_id = str(row.get("group_id") or make_group_id(args.dataset_name, record_id))
            record = {
                "id": record_id,
                "group_id": group_id,
                "split": str(row.get("split") or stable_split(group_id)),
                "audio_filepath": str(audio),
                "audio_ref": {"kind": "local", "repo_id": None, "config": None, "split": None, "row_id": record_id},
                "text": text,
                "target": row.get("target"),
                "candidates": row.get("candidates", []),
                "category": str(row.get("category", args.category)),
                "source": {
                    "dataset": args.dataset_name,
                    "config": None,
                    "split": row.get("split"),
                    "source_id": row.get("id", index),
                    "revision": None,
                    "license": args.license,
                    "synthetic": False,
                    "audio_redistributed": False,
                },
                "difficulty": row.get("difficulty", {"acoustic": 0.0, "lexical": 0.0, "context": 0.0, "phone_distance": None}),
                "metadata": {
                    "original_duration": row.get("duration"),
                    "original": {key: value for key, value in row.items() if key not in {"audio_filepath", "text"}},
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"imported {count} records to {args.output}")


if __name__ == "__main__":
    main()
