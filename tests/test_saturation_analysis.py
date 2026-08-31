from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_metrics(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(100):
        benchmark_id = f"item-{index:03d}"
        category = "exact_homophone" if index < 50 else "near_homophone"
        baseline_correct = index % 2 == 0
        for experiment, correct, distractor_fp in (
            ("E03_pb_0.5", baseline_correct, False),
            ("E03_pb_1.0", True if index < 80 else baseline_correct, False),
            ("E03_pb_2.0", True if index < 80 else baseline_correct, True),
        ):
            rows.append(
                {
                    "experiment": experiment,
                    "benchmark_id": benchmark_id,
                    "category": category,
                    "entity_correct": correct,
                    "distractor_false_positive": distractor_fp,
                    "bias_false_positive": None,
                    "cer": 0.0 if correct else 0.2,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_detects_plateau_with_rising_false_positives(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.parquet"
    sweep = tmp_path / "sweep.json"
    output = tmp_path / "saturation.json"
    write_metrics(metrics)
    sweep.write_text(
        json.dumps(
            {
                "sweeps": [
                    {
                        "name": "gpu_pb_alpha",
                        "parameter": "PB_ALPHA",
                        "benefit_metric": "entity_accuracy",
                        "risk_metric": "distractor_false_positive_rate",
                        "categories": ["exact_homophone", "near_homophone"],
                        "points": [
                            {"experiment": "E03_pb_0.5", "value": 0.5},
                            {"experiment": "E03_pb_1.0", "value": 1.0},
                            {"experiment": "E03_pb_2.0", "value": 2.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_saturation.py",
            "--metrics",
            str(metrics),
            "--sweep",
            str(sweep),
            "--output",
            str(output),
            "--bootstrap-samples",
            "500",
            "--seed",
            "1234",
            "--min-gain",
            "0.01",
            "--min-risk-increase",
            "0.02",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    result = report["sweeps"][0]
    assert result["saturation_detected"] is True
    assert result["saturation_at"] == {
        "from": 1.0,
        "to": 2.0,
        "experiment": "E03_pb_2.0",
    }
    second = result["transitions"][1]
    assert second["plateau"] is True
    assert second["risk_rising"] is True
    assert second["mcnemar_entity"]["wins"] == 0
    assert second["mcnemar_entity"]["losses"] == 0


def test_does_not_call_improvement_saturation(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.parquet"
    output = tmp_path / "saturation.json"
    sweep = tmp_path / "sweep.json"
    rows = []
    for index in range(50):
        benchmark_id = f"item-{index:03d}"
        rows.extend(
            [
                {
                    "experiment": "low",
                    "benchmark_id": benchmark_id,
                    "category": "near_homophone",
                    "entity_correct": False,
                    "distractor_false_positive": False,
                    "bias_false_positive": None,
                    "cer": 0.2,
                },
                {
                    "experiment": "high",
                    "benchmark_id": benchmark_id,
                    "category": "near_homophone",
                    "entity_correct": True,
                    "distractor_false_positive": True,
                    "bias_false_positive": None,
                    "cer": 0.0,
                },
            ]
        )
    pq.write_table(pa.Table.from_pylist(rows), metrics)
    sweep.write_text(
        json.dumps(
            {
                "sweeps": [
                    {
                        "name": "pb",
                        "points": [
                            {"experiment": "low", "value": 0.5},
                            {"experiment": "high", "value": 1.0},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_saturation.py",
            "--metrics",
            str(metrics),
            "--sweep",
            str(sweep),
            "--output",
            str(output),
            "--bootstrap-samples",
            "200",
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["sweeps"][0]["saturation_detected"] is False
