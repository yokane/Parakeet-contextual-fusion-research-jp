#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from importlib import metadata


def normalize_machine(value: str) -> str:
    value = value.lower().strip()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    return value


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the only supported J-PACF research platform: Linux x86_64 + CUDA 13"
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    system = platform.system().lower()
    machine = normalize_machine(platform.machine())
    expected_platform = os.environ.get("JPA_CF_PLATFORM", "linux/amd64")
    expected_cuda_major = os.environ.get("JPA_CF_CUDA_MAJOR", "13")

    if system != "linux":
        raise SystemExit(f"unsupported OS: {platform.system()}; only Linux is validated")
    if machine != "x86_64":
        raise SystemExit(f"unsupported architecture: {platform.machine()}; only x86_64 is validated")
    if expected_platform != "linux/amd64":
        raise SystemExit(f"invalid repository platform contract: {expected_platform!r}")
    if expected_cuda_major != "13":
        raise SystemExit(f"invalid CUDA major contract: {expected_cuda_major!r}")

    result: dict[str, object] = {
        "platform": "linux/amd64",
        "cuda_major_contract": 13,
        "python": platform.python_version(),
        "nemo_toolkit": package_version("nemo-toolkit"),
        "torch": package_version("torch"),
        "gpu_required": args.require_gpu,
    }

    if args.require_gpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU is required for authoritative NeMo research validation")
        cuda_version = str(torch.version.cuda or "")
        if not cuda_version.startswith("13."):
            raise SystemExit(f"expected CUDA 13 PyTorch runtime, got {cuda_version!r}")
        result.update(
            {
                "cuda_runtime": cuda_version,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
            }
        )

        nemo_version = package_version("nemo-toolkit")
        if nemo_version != "3.0.0":
            raise SystemExit(f"expected nemo-toolkit 3.0.0, got {nemo_version!r}")
        torch_version = package_version("torch")
        if torch_version != "2.12.0+cu132":
            raise SystemExit(f"expected torch 2.12.0+cu132, got {torch_version!r}")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
