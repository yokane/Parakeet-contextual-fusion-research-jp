#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def phone_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item))
    return ()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare E05 phone-head training data from E04 and benchmark metadata")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--e04", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/encoder"))
    parser.add_argument("--train-feature-dir", type=Path, default=Path("artifacts/encoder_train"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/generated/phone_train.jsonl"))
    parser.add_argument("--vocab", type=Path, default=Path("artifacts/phone_vocab.json"))
    parser.add_argument("--annotated-e04", type=Path, default=Path("results/E04_phone_ready.jsonl"))
    args = parser.parse_args()

    benchmark = {str(row["id"]): row for row in read_jsonl(args.benchmark)}
    e04_rows = read_jsonl(args.e04)

    phones: set[str] = set()
    for row in benchmark.values():
        phones.update(phone_tuple((row.get("target") or {}).get("phones")))
        for candidate in row.get("candidates") or []:
            phones.update(phone_tuple((candidate or {}).get("phones")))
    if not phones:
        raise SystemExit("benchmark contains no phone labels")
    phone_to_id = {phone: index for index, phone in enumerate(sorted(phones))}
    blank_id = len(phone_to_id)
    vocab_payload = {
        "phones": [phone for phone, _index in sorted(phone_to_id.items(), key=lambda item: item[1])],
        "phone_to_id": phone_to_id,
        "blank_id": blank_id,
        "phone_vocab_size": blank_id + 1,
    }
    args.vocab.parent.mkdir(parents=True, exist_ok=True)
    args.vocab.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.train_feature_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict[str, Any]] = []
    annotated_rows: list[dict[str, Any]] = []

    for row in e04_rows:
        benchmark_id = str(row.get("benchmark_id") or row.get("id") or "")
        bench = benchmark.get(benchmark_id)
        if bench is None:
            raise RuntimeError(f"E04 row {benchmark_id!r} does not match the benchmark")
        target = bench.get("target") or {}
        target_surface = str(target.get("surface") or "")
        target_phones = phone_tuple(target.get("phones"))

        surface_phones: dict[str, tuple[str, ...]] = {}
        if target_surface and target_phones:
            surface_phones[target_surface] = target_phones
        for candidate in bench.get("candidates") or []:
            surface = str((candidate or {}).get("surface") or "")
            candidate_phones = phone_tuple((candidate or {}).get("phones"))
            if surface and candidate_phones:
                surface_phones[surface] = candidate_phones

        target_window: list[int] | None = None
        for hypothesis in row.get("candidates") or []:
            text = str((hypothesis or {}).get("text") or "")
            matches = [(surface, values) for surface, values in surface_phones.items() if surface in text]
            if matches:
                matches.sort(key=lambda item: len(item[0]), reverse=True)
                _surface, candidate_phones = matches[0]
                metadata = hypothesis.setdefault("metadata", {})
                metadata["phone_ids"] = [phone_to_id[phone] for phone in candidate_phones]
            if target_surface and target_surface in text and target_window is None:
                raw_window = (hypothesis.get("metadata") or {}).get("ctc_window")
                if isinstance(raw_window, list) and len(raw_window) == 2:
                    target_window = [int(raw_window[0]), int(raw_window[1])]

        feature_path = args.feature_dir / f"{benchmark_id}.pt"
        if target_phones and target_window and feature_path.exists():
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            states = payload["encoder_states"] if isinstance(payload, dict) else payload
            start = max(0, target_window[0])
            end = min(int(states.shape[0]), target_window[1])
            if end > start:
                cropped_path = args.train_feature_dir / f"{benchmark_id}.pt"
                torch.save({"encoder_states": states[start:end].contiguous()}, cropped_path)
                train_rows.append(
                    {
                        "id": benchmark_id,
                        "feature_path": str(cropped_path),
                        "phone_ids": [phone_to_id[phone] for phone in target_phones],
                        "category": bench.get("category"),
                    }
                )
        annotated_rows.append(row)

    if not train_rows:
        raise SystemExit("no E05 training rows could be prepared from E04 CTC windows")
    write_jsonl(args.train_manifest, train_rows)
    write_jsonl(args.annotated_e04, annotated_rows)
    print(json.dumps({"training_rows": len(train_rows), "phone_vocab_size": blank_id + 1}, indent=2))


if __name__ == "__main__":
    main()
