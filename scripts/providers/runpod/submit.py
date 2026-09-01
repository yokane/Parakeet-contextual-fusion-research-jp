#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shlex
from pathlib import Path
from typing import Any

import runpod


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or submit a non-production J-PACF RunPod GPU Pod")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("provider") != "runpod":
        raise SystemExit("job spec provider must be runpod")
    if spec.get("platform") != "linux/amd64" or spec.get("cuda_major") != 13:
        raise SystemExit("job spec is outside the Linux x86_64 CUDA13 contract")
    if args.gpu_count < 1 or args.container_disk_gb < 20:
        raise SystemExit("invalid RunPod resource request")

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
    plan: dict[str, Any] = {
        "provider": "runpod",
        "mode": "non-production-container-run",
        "name": f"jpacf-{spec['experiment_id']}",
        "image_name": spec["image"],
        "gpu_type_id": args.gpu_type,
        "gpu_count": args.gpu_count,
        "container_disk_in_gb": args.container_disk_gb,
        "cloud_type": "ALL",
        "allowed_cuda_versions": ["13.0", "13.1", "13.2", "13.3"],
        "docker_args": shlex.join([str(item) for item in spec["command"]]),
        "env": environment,
        "submitted": False,
    }

    if not args.submit:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required when --submit is used")
    runpod.api_key = api_key

    signature = inspect.signature(runpod.create_pod)
    optional = {
        "gpu_count": args.gpu_count,
        "container_disk_in_gb": args.container_disk_gb,
        "cloud_type": "ALL",
        "allowed_cuda_versions": plan["allowed_cuda_versions"],
        "docker_args": plan["docker_args"],
        "env": environment,
    }
    kwargs = {name: value for name, value in optional.items() if name in signature.parameters}
    pod = runpod.create_pod(plan["name"], plan["image_name"], args.gpu_type, **kwargs)
    plan["submitted"] = True
    plan["pod_id"] = getattr(pod, "id", None) or (pod.get("id") if isinstance(pod, dict) else None)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
