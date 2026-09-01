#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def next_sequence_id(prefix: str, lines: list[str]) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prefix):
        raise ValueError(f"invalid allocation prefix: {prefix!r}")
    pattern = re.compile(rf"(?:^|/){re.escape(prefix)}-(\d{{6}})(?:/|$)")
    highest = 0
    for line in lines:
        for match in pattern.finditer(line.strip()):
            highest = max(highest, int(match.group(1)))
    if highest >= 999999:
        raise OverflowError(f"sequence exhausted for prefix {prefix!r}")
    return f"{prefix}-{highest + 1:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocate the next six-digit HF Bucket sequence ID")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--listing", type=Path, required=True)
    args = parser.parse_args()
    print(next_sequence_id(args.prefix, args.listing.read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    main()
