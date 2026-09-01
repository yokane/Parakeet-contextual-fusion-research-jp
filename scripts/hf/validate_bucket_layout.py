#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def validate_layout(listing: list[str], config: dict[str, object]) -> dict[str, object]:
    paths = {line.strip().lstrip("/") for line in listing if line.strip()}
    required_roots = config.get("required_roots")
    if not isinstance(required_roots, list) or not required_roots:
        raise ValueError("hf-storage config has no required_roots")
    missing = [root for root in required_roots if f"{root}/README.md" not in paths]
    if "README.md" not in paths:
        missing.insert(0, "<root>/README.md")
    if "config/current.json" not in paths:
        missing.append("config/current.json")
    if missing:
        raise ValueError(f"HF Bucket layout is incomplete: {missing}")

    malformed_candidates = sorted(
        path
        for path in paths
        if path.startswith("candidates/candidate-")
        and not re.match(r"^candidates/candidate-\d{6}/", path)
    )
    if malformed_candidates:
        raise ValueError(f"malformed canonical candidate paths: {malformed_candidates[:5]}")

    return {
        "bucket": config.get("bucket"),
        "objects": len(paths),
        "required_roots": required_roots,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical J-PACF HF Bucket layout")
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hf-storage.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = validate_layout(args.listing.read_text(encoding="utf-8").splitlines(), config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
