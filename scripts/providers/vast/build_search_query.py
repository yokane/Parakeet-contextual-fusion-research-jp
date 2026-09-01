#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE_FILTERS = (
    "verified=true",
    "rentable=true",
    "cuda_max_good>=13",
    "disk_space>=50",
    "gpu_arch=nvidia",
)
TOKEN = re.compile(r"^[A-Za-z0-9_.+-]+$")


def append_numeric_filter(filters: list[str], field: str, value: str) -> None:
    value = value.strip()
    if not value:
        return
    filters.append(field + (("=" if value[0].isdigit() else "") + value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical Vast CUDA13 research query")
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--num-gpus", default="1")
    parser.add_argument("--gpu-ram", default="")
    parser.add_argument("--duration", default="")
    parser.add_argument("--inet-down", default="")
    parser.add_argument("--reliability", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    filters = list(BASE_FILTERS)
    if args.gpu_name:
        names = [item.strip() for item in args.gpu_name.split(",") if item.strip()]
        if not all(TOKEN.fullmatch(item) for item in names):
            raise SystemExit("invalid gpu-name")
        filters.append(
            "gpu_name=" + names[0]
            if len(names) == 1
            else "gpu_name in [" + ",".join(names) + "]"
        )
    if args.num_gpus:
        filters.append("num_gpus=" + args.num_gpus.strip())
    append_numeric_filter(filters, "gpu_ram", args.gpu_ram)
    append_numeric_filter(filters, "duration", args.duration)
    append_numeric_filter(filters, "inet_down", args.inet_down)
    append_numeric_filter(filters, "reliability", args.reliability)

    payload = {
        "schema_version": 1,
        "query": " ".join(filters),
        "base_filters": list(BASE_FILTERS),
        "platform": "linux/amd64",
        "cuda_major": 13,
        "storage_gb": 50,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["query"])


if __name__ == "__main__":
    main()
