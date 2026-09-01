from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("scripts/validate_audio_coverage.py")
    spec = importlib.util.spec_from_file_location("validate_audio_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_coverage_passes_required_categories() -> None:
    module = load_module()
    report = module.build_report(
        {
            "repo_id": "example/bench",
            "config": "test",
            "splits": ["test"],
            "records": 20,
            "runnable_audio_records": 12,
            "categories": {"exact_homophone": 10, "near_homophone": 10},
            "runnable_categories": {"exact_homophone": 6, "near_homophone": 6},
        },
        required_categories=["exact_homophone", "near_homophone"],
        min_per_category=5,
        min_total=10,
    )
    assert report["passed"] is True
    assert report["overall_coverage"] == 0.6
    assert report["categories"]["exact_homophone"]["coverage"] == 0.6


def test_audio_coverage_reports_missing_required_category() -> None:
    module = load_module()
    report = module.build_report(
        {
            "records": 10,
            "runnable_audio_records": 4,
            "categories": {"exact_homophone": 5, "near_homophone": 5},
            "runnable_categories": {"exact_homophone": 4},
        },
        required_categories=["exact_homophone", "near_homophone"],
        min_per_category=2,
        min_total=1,
    )
    assert report["passed"] is False
    assert any("near_homophone" in failure for failure in report["failures"])
