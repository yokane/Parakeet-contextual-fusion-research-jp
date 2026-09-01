#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from parakeet_context_fusion.model_io import load_locked_model_identity, materialize_locked_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the immutable locked Parakeet .nemo checkpoint")
    parser.add_argument("--lock", type=Path, default=Path("locks/hf-revisions.lock.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/parakeet-tdt_ctc-0.6b-ja.nemo"))
    parser.add_argument("--copy", action="store_true", help="copy instead of creating a symlink to the HF cache")
    args = parser.parse_args()

    identity = load_locked_model_identity(args.lock)
    source = materialize_locked_model(identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        args.output.unlink()
    if args.copy:
        shutil.copy2(source, args.output)
    else:
        args.output.symlink_to(os.path.relpath(source, start=args.output.parent))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
