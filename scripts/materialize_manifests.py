#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nemo_path = args.output_dir / "nemo_eval.jsonl"
    context_path = args.output_dir / "context_phrases.txt"
    corpus_path = args.output_dir / "lm_corpus.txt"
    provenance_path = args.output_dir / "provenance.json"
    phrases: set[str] = set()
    corpus: list[str] = []
    nemo_count = 0
    source_counts: dict[str, int] = {}
    with nemo_path.open("w", encoding="utf-8") as nemo:
        for row in rows:
            source_name = str(row.get("source", {}).get("dataset", "unknown"))
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            text = str(row.get("text", "")).strip()
            if text:
                corpus.append(text)
            target = row.get("target") or {}
            surface = str(target.get("surface", "")).strip()
            if surface:
                phrases.add(surface)
            for candidate in row.get("candidates", []):
                candidate_surface = str(candidate.get("surface", "")).strip()
                if candidate_surface:
                    phrases.add(candidate_surface)
            audio = row.get("audio_filepath")
            if not audio:
                continue
            audio_path = Path(audio).expanduser()
            if not audio_path.exists():
                continue
            nemo.write(json.dumps({
                "audio_filepath": str(audio_path.resolve()),
                "duration": duration_seconds(audio_path),
                "text": text,
                "benchmark_id": row.get("id"),
                "category": row.get("category"),
            }, ensure_ascii=False) + "\n")
            nemo_count += 1
    context_path.write_text("\n".join(sorted(phrases)) + ("\n" if phrases else ""), encoding="utf-8")
    corpus_path.write_text("\n".join(corpus) + ("\n" if corpus else ""), encoding="utf-8")
    provenance_path.write_text(json.dumps({
        "input": str(args.input),
        "records": len(rows),
        "nemo_audio_records": nemo_count,
        "context_phrases": len(phrases),
        "source_counts": source_counts,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(rows)} nemo_audio_records={nemo_count} context_phrases={len(phrases)}")


if __name__ == "__main__":
    main()
