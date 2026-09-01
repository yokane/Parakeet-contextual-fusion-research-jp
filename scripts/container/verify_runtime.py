#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from importlib import metadata
from pathlib import Path


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def writable_probe(path: str) -> bool:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=target, prefix="jpacf-", delete=True):
            pass
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the portable J-PACF CUDA runtime")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if platform.system().lower() != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("portable runtime requires Linux x86_64")

    expected_venv = "/opt/jpacf/.venv/"
    if not sys.executable.startswith(expected_venv):
        raise SystemExit(
            f"expected repository-owned Python under {expected_venv}, got {sys.executable}"
        )

    import torch

    result: dict[str, object] = {
        "platform": "linux/amd64",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "nemo_toolkit": version("nemo-toolkit"),
        "hf_home": os.environ.get("HF_HOME"),
        "uv_cache_dir": os.environ.get("UV_CACHE_DIR"),
        "workspace": os.environ.get("JPA_CF_WORKSPACE", os.getcwd()),
    }

    if torch.__version__ != "2.12.0+cu132":
        raise SystemExit(f"expected torch 2.12.0+cu132, got {torch.__version__}")
    if version("nemo-toolkit") != "3.0.0":
        raise SystemExit(f"expected nemo-toolkit 3.0.0, got {version('nemo-toolkit')!r}")
    if not str(torch.version.cuda or "").startswith("13.2"):
        raise SystemExit(f"expected PyTorch CUDA 13.2 runtime, got {torch.version.cuda!r}")

    cuda_available = torch.cuda.is_available()
    result["cuda_available"] = cuda_available
    if args.require_gpu and not cuda_available:
        raise SystemExit("CUDA is not available inside the container")

    if cuda_available:
        device = torch.device("cuda:0")
        probe = torch.arange(1, 4097, device=device, dtype=torch.float32)
        checksum = float((probe * probe).sum().item())
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory": props.total_memory,
                "cuda_compute_probe": checksum,
            }
        )

    state_paths = {
        "hf": os.environ.get("HF_HOME", "/cache/huggingface"),
        "uv": os.environ.get("UV_CACHE_DIR", "/cache/uv"),
        "xdg": os.environ.get("XDG_CACHE_HOME", "/cache/xdg"),
        "torch": os.environ.get("TORCH_HOME", "/cache/torch"),
    }
    result["state_writable"] = {name: writable_probe(path) for name, path in state_paths.items()}
    if not all(result["state_writable"].values()):
        raise SystemExit(f"one or more runtime state mounts are not writable: {result['state_writable']}")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
