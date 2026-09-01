from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name: str):
    path = Path("scripts/hf") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_sequence_id_uses_highest_canonical_or_nested_id() -> None:
    module = load_script("next_sequence_id")
    listing = [
        "candidates/candidate-000001/README.md",
        "candidates/candidate-000009/metadata.json",
        "candidates/legacy/candidate-000012/metadata.json",
        "runs/gh-123/metrics.json",
    ]
    assert module.next_sequence_id("candidate", listing) == "candidate-000013"


def test_next_sequence_id_rejects_unsafe_prefix() -> None:
    module = load_script("next_sequence_id")
    with pytest.raises(ValueError, match="invalid allocation prefix"):
        module.next_sequence_id("../candidate", [])


def test_upload_only_sync_plan_is_accepted(tmp_path: Path) -> None:
    module = load_script("validate_sync_plan")
    destination = "hf://buckets/saeeew/J-PACF-YOMI-tdt-bucket/candidates/candidate-000001"
    rows = [
        {
            "type": "header",
            "source": "/tmp/candidate",
            "dest": destination,
            "timestamp": "2026-09-01T00:00:00Z",
            "summary": {"upload": 2},
        },
        {"type": "operation", "action": "upload", "path": "metadata.json", "reason": "new"},
        {
            "type": "operation",
            "action": "upload",
            "path": "release_manifest.json",
            "reason": "new",
        },
    ]
    plan = tmp_path / "plan.jsonl"
    plan.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert module.validate_fresh_upload_plan(module.load_jsonl(plan), destination) == 2


@pytest.mark.parametrize("action", ["delete", "download", "skip"])
def test_sync_plan_rejects_non_upload_operations(action: str) -> None:
    module = load_script("validate_sync_plan")
    rows = [
        {"type": "header", "source": "/tmp/source", "dest": "hf://buckets/x/y/candidates/candidate-000001"},
        {"type": "operation", "action": action, "path": "old.bin", "reason": "mutation"},
    ]
    with pytest.raises(ValueError, match="only upload operations"):
        module.validate_fresh_upload_plan(rows)


def test_sync_plan_rejects_wrong_destination() -> None:
    module = load_script("validate_sync_plan")
    rows = [
        {"type": "header", "source": "/tmp/source", "dest": "hf://buckets/x/y/candidates/candidate-000001"},
        {"type": "operation", "action": "upload", "path": "metadata.json"},
    ]
    with pytest.raises(ValueError, match="unexpected sync destination"):
        module.validate_fresh_upload_plan(rows, "hf://buckets/x/y/candidates/candidate-000002")


def canonical_bucket_listing() -> list[str]:
    return [
        "README.md",
        "config/README.md",
        "config/current.json",
        "candidates/README.md",
        "experiments/README.md",
        "runs/README.md",
        "benchmarks/README.md",
        "reference/README.md",
        "scripts/README.md",
        "tmp/README.md",
        "candidates/candidate-000001/README.md",
    ]


def storage_config() -> dict[str, object]:
    return json.loads(Path("configs/hf-storage.json").read_text(encoding="utf-8"))


def test_bucket_layout_accepts_canonical_tree() -> None:
    module = load_script("validate_bucket_layout")
    report = module.validate_layout(canonical_bucket_listing(), storage_config())
    assert report["status"] == "ok"
    assert report["bucket"] == "saeeew/J-PACF-YOMI-tdt-bucket"


def test_bucket_layout_rejects_missing_root() -> None:
    module = load_script("validate_bucket_layout")
    listing = [path for path in canonical_bucket_listing() if path != "runs/README.md"]
    with pytest.raises(ValueError, match="runs"):
        module.validate_layout(listing, storage_config())
