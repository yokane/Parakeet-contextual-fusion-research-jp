from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_scorer_release(tmp_path: Path) -> None:
    output_root = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_hf_model_release.py",
            "--release",
            "v-test",
            "--output-root",
            str(output_root),
        ],
        check=True,
    )

    release_dir = output_root / "v-test"
    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "saeeew/J-PACF-YOMI-tdt"
    assert manifest["release"] == "v-test"
    assert manifest["release_kind"] == "scorer-or-config"
    assert manifest["contains_standalone_checkpoint"] is False
    assert (release_dir / "fusion_config.yaml").is_file()
    assert (release_dir / "research_manifest.json").is_file()
    roles = {entry["role"] for entry in manifest["files"]}
    assert roles == {"fusion_config", "research_manifest"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_release_name_rejects_path_traversal(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_hf_model_release.py",
            "--release",
            "../bad",
            "--output-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "path-safe" in result.stderr
