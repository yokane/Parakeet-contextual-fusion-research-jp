from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_collect_experiment_metrics(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    e00 = tmp_path / "e00.jsonl"
    e03 = tmp_path / "e03.jsonl"
    parquet = tmp_path / "metrics.parquet"
    summary = tmp_path / "summary.json"
    rows = [
        {
            "id": "a",
            "group_id": "g1",
            "text": "新しい気候モデル",
            "target": {"surface": "気候"},
            "candidates": [{"surface": "気候"}, {"surface": "機構"}],
            "category": "exact_homophone",
            "metadata": {},
        },
        {
            "id": "b",
            "group_id": "g2",
            "text": "電気を使う",
            "target": {"surface": "電気"},
            "candidates": [{"surface": "電気"}, {"surface": "天気"}],
            "category": "near_homophone",
            "metadata": {},
        },
    ]
    write_jsonl(benchmark, rows)
    write_jsonl(
        e00,
        [
            {"benchmark_id": "a", "pred_text": "新しい機構モデル"},
            {"benchmark_id": "b", "pred_text": "電気を使う"},
        ],
    )
    write_jsonl(
        e03,
        [
            {"benchmark_id": "a", "pred_text": "新しい気候モデル"},
            {"benchmark_id": "b", "pred_text": "電気を使う"},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_experiment_metrics.py",
            "--benchmark",
            str(benchmark),
            "--result",
            f"E00={e00}",
            "--result",
            f"E03={e03}",
            "--parquet",
            str(parquet),
            "--summary",
            str(summary),
        ],
        check=True,
    )

    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["experiments"]["E00"]["overall"]["entity_accuracy"] == 0.5
    assert report["experiments"]["E03"]["overall"]["entity_accuracy"] == 1.0
    assert report["paired_vs_baseline"]["E03"]["entity_wins"] == 1
    table = pq.read_table(parquet)
    assert table.num_rows == 4
    assert set(table.column("experiment").to_pylist()) == {"E00", "E03"}
