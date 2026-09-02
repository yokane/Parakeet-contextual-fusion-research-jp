#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print deterministic E00-E06 reusable research cache key")
    parser.add_argument("--lock", type=Path, default=Path("locks/hf-revisions.lock.json"))
    parser.add_argument("--ngram-order", type=int, default=6)
    args = parser.parse_args()
    payload = json.loads(args.lock.read_text(encoding="utf-8"))["repositories"]
    benchmark = payload["benchmark"]["revision"]
    model = payload["base_model"]["revision"]
    if len(benchmark) != 40 or len(model) != 40:
        raise SystemExit("research key requires full locked Hugging Face revisions")
    print(f"v1-bench-{benchmark[:12]}-model-{model[:12]}-ng{args.ngram_order}")


if __name__ == "__main__":
    main()
