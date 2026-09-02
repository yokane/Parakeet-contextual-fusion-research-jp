#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebase portable NeMo manifest audio paths after HF Bucket restore")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    args = parser.parse_args()

    rows=[]
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        old=Path(str(row.get("audio_filepath") or ""))
        if not old.name:
            raise SystemExit("manifest row has no audio_filepath")
        candidate=args.audio_dir / old.name
        if not candidate.is_file():
            raise SystemExit(f"restored audio file is missing: {candidate}")
        row["audio_filepath"]=str(candidate.resolve())
        rows.append(row)

    tmp=args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with tmp.open("w",encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row,ensure_ascii=False)+"\n")
    tmp.replace(args.manifest)
    print(f"rebased {len(rows)} manifest rows to {args.audio_dir}")


if __name__ == "__main__":
    main()
