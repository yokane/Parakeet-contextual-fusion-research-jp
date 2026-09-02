from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research" / "stage_fingerprints.py"


def _run(task: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--task", task, "--field", "fingerprint"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_non_e06_task_does_not_require_e06_external_identity() -> None:
    env = os.environ.copy()
    env.pop("JPA_CF_E06_DRIVER_SHA256", None)
    proc = _run("common", env)
    assert proc.returncode == 0, proc.stderr
    assert re.fullmatch(r"[0-9a-f]{64}\n?", proc.stdout)


def test_e06_fingerprint_fails_closed_without_driver_digest() -> None:
    env = os.environ.copy()
    env.pop("JPA_CF_E06_DRIVER_SHA256", None)
    missing = _run("E06", env)
    assert missing.returncode != 0
    assert "JPA_CF_E06_DRIVER_SHA256" in missing.stderr

    env["JPA_CF_E06_DRIVER_SHA256"] = "a" * 64
    present = _run("E06", env)
    assert present.returncode == 0, present.stderr
    assert re.fullmatch(r"[0-9a-f]{64}\n?", present.stdout)
