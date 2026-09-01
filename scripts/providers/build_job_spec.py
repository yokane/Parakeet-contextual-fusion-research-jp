#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")
OCI_DIGEST = re.compile(r"^[a-z0-9./_-]+(?:\:[a-z0-9._-]+)?@sha256:[0-9a-f]{64}$", re.IGNORECASE)


def require_revision(value: str, label: str) -> str:
    value = value.strip()
    if not HEX40.fullmatch(value):
        raise SystemExit(f"{label} must be one full 40-character commit revision")
    return value


def require_image(value: str) -> str:
    value = value.strip()
    if not OCI_DIGEST.fullmatch(value):
        raise SystemExit("image must be an immutable OCI reference ending in @sha256:<64 hex>")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable J-PACF GPU research job contract")
    parser.add_argument("--provider", choices=["self-hosted", "vast", "runpod", "hf-jobs"], required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-id", default="nvidia/parakeet-tdt_ctc-0.6b-ja")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-id", default="saeeew/JP-HomophoneBench")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--command", default="bash experiments/E00_tdt_greedy.sh")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--output", type=Path, default=Path("dist/providers/job-spec.json"))
    args = parser.parse_args()

    providers = json.loads(Path("configs/research-providers.json").read_text(encoding="utf-8"))
    provider = providers["providers"][args.provider]
    command = shlex.split(args.command)
    if not command:
        raise SystemExit("command must not be empty")

    source_sha = args.source_sha.strip()
    if source_sha and not HEX40.fullmatch(source_sha):
        raise SystemExit("source-sha must be a full Git commit SHA when provided")

    spec = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "suite": args.suite,
        "provider": args.provider,
        "platform": "linux/amd64",
        "cuda_major": 13,
        "image": require_image(args.image),
        "model": {
            "repo_id": args.model_id,
            "revision": require_revision(args.model_revision, "model-revision"),
        },
        "dataset": {
            "repo_id": args.dataset_id,
            "revision": require_revision(args.dataset_revision, "dataset-revision"),
        },
        "source_sha": source_sha or None,
        "command": command,
        "provider_contract": provider,
        "results": {
            "bucket": "saeeew/J-PACF-YOMI-tdt-bucket",
            "append_only": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
