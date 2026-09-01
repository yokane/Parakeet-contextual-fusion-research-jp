#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import run_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an immutable J-PACF job to Hugging Face Jobs")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--namespace", default="saeeew")
    parser.add_argument("--detach", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("provider") != "hf-jobs":
        raise SystemExit("job spec provider must be hf-jobs")
    if spec.get("platform") != "linux/amd64" or spec.get("cuda_major") != 13:
        raise SystemExit("job spec is outside the Linux x86_64 CUDA13 research contract")

    environment = {
        "JPA_CF_EXPERIMENT_ID": str(spec["experiment_id"]),
        "JPA_CF_PLATFORM": "linux/amd64",
        "JPA_CF_CUDA_MAJOR": "13",
        "MODEL_ID": str(spec["model"]["repo_id"]),
        "MODEL_REVISION": str(spec["model"]["revision"]),
        "DATASET_ID": str(spec["dataset"]["repo_id"]),
        "DATASET_REVISION": str(spec["dataset"]["revision"]),
        "HF_BUCKET": str(spec["results"]["bucket"]),
    }
    job = run_job(
        image=str(spec["image"]),
        command=[str(item) for item in spec["command"]],
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        env=environment,
        secrets={"HF_TOKEN": token},
        token=token,
    )

    result = {
        "provider": "hf-jobs",
        "experiment_id": spec["experiment_id"],
        "job_id": getattr(job, "id", None),
        "status": str(getattr(job, "status", "submitted")),
        "image": spec["image"],
        "flavor": args.flavor,
        "detached": args.detach,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not args.detach and hasattr(job, "wait"):
        job.wait()


if __name__ == "__main__":
    main()
