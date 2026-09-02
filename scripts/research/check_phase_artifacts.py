#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E00-E06 research artifact prerequisites")
    parser.add_argument("phase", choices=[f"E{i:02d}" for i in range(7)])
    parser.add_argument("--config", type=Path, default=Path("configs/research/e00-e06-artifacts.yaml"))
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    phase = payload["phases"][args.phase]
    artifacts = payload["artifacts"]
    root = args.state_root or Path(payload.get("state_root") or "/workspace/state")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for name in phase.get("requires", []):
        spec = artifacts[name]
        path = root / spec["path"]
        kind = spec.get("kind", "file")
        exists = path.is_dir() if kind == "directory" else path.is_file()
        nonempty = False
        if exists:
            if kind == "directory":
                nonempty = any(path.iterdir())
            else:
                nonempty = path.stat().st_size > 0
        ok = bool(exists and nonempty)
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "kind": kind,
                "producer": spec.get("producer"),
                "ok": ok,
            }
        )
        if not ok:
            missing.append(name)

    result = {
        "phase": args.phase,
        "executor": phase.get("executor"),
        "image": phase.get("image"),
        "state_root": str(root),
        "artifacts": rows,
        "missing": missing,
        "ready": not missing,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
