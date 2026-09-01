from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_context_stress_analysis_reports_robust_limit(tmp_path: Path) -> None:
    rows = []
    for index in range(20):
        benchmark_id = f"b{index:02d}"
        rows.append(
            {
                "experiment": "E03_ctxd_00000",
                "benchmark_id": benchmark_id,
                "category": "near_homophone",
                "entity_correct": True,
                "distractor_false_positive": False,
                "cer": 0.0,
            }
        )
        rows.append(
            {
                "experiment": "E03_ctxd_00010",
                "benchmark_id": benchmark_id,
                "category": "near_homophone",
                "entity_correct": True,
                "distractor_false_positive": False,
                "cer": 0.0,
            }
        )
    metrics = tmp_path / "metrics.parquet"
    pq.write_table(pa.Table.from_pylist(rows), metrics)
    stress = tmp_path / "stress.json"
    stress.write_text(
        json.dumps(
            {
                "cases": [
                    {"requested_distractors": 0, "total_phrases": 2},
                    {"requested_distractors": 10, "total_phrases": 12},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_context_stress.py",
            "--metrics",
            str(metrics),
            "--stress-manifest",
            str(stress),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["baseline_experiment"] == "E03_ctxd_00000"
    assert report["max_robust_requested_distractors"] == 10
    assert report["cases"][1]["overall"]["within_threshold"] is True
