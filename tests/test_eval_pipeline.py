from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_collector(
    *,
    benchmark: Path,
    results: list[tuple[str, Path]],
    parquet: Path,
    summary: Path,
    execution_manifest: Path | None = None,
) -> None:
    command = [
        sys.executable,
        "scripts/collect_experiment_metrics.py",
        "--benchmark",
        str(benchmark),
    ]
    if execution_manifest is not None:
        command.extend(["--execution-manifest", str(execution_manifest)])
    for experiment, path in results:
        command.extend(["--result", f"{experiment}={path}"])
    command.extend(["--parquet", str(parquet), "--summary", str(summary)])
    subprocess.run(command, check=True)


def benchmark_rows() -> list[dict[str, object]]:
    return [
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
        {
            "id": "c",
            "group_id": "g3",
            "text": "午後に公園へ行く",
            "target": {"surface": "公園"},
            "candidates": [{"surface": "公園"}, {"surface": "講演"}],
            "category": "long_vowel",
            "metadata": {},
        },
    ]


def test_collect_experiment_metrics(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    e00 = tmp_path / "e00.jsonl"
    e03 = tmp_path / "e03.jsonl"
    parquet = tmp_path / "metrics.parquet"
    summary = tmp_path / "summary.json"
    write_jsonl(benchmark, benchmark_rows()[:2])
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

    run_collector(
        benchmark=benchmark,
        results=[("E00", e00), ("E03", e03)],
        parquet=parquet,
        summary=summary,
    )

    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["experiments"]["E00"]["overall"]["entity_accuracy"] == 0.5
    assert report["experiments"]["E03"]["overall"]["entity_accuracy"] == 1.0
    assert report["paired_vs_baseline"]["E03"]["entity_wins"] == 1
    table = pq.read_table(parquet)
    assert table.num_rows == 4
    assert set(table.column("experiment").to_pylist()) == {"E00", "E03"}


def test_subset_results_use_execution_manifest_ids(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    execution = tmp_path / "nemo_eval.jsonl"
    result = tmp_path / "e00.jsonl"
    parquet = tmp_path / "metrics.parquet"
    summary = tmp_path / "summary.json"
    write_jsonl(benchmark, benchmark_rows())
    write_jsonl(
        execution,
        [
            {"benchmark_id": "b", "audio_filepath": "/tmp/b.wav", "text": "電気を使う"},
            {"benchmark_id": "c", "audio_filepath": "/tmp/c.wav", "text": "午後に公園へ行く"},
        ],
    )
    write_jsonl(
        result,
        [
            {"pred_text": "電気を使う"},
            {"pred_text": "午後に公園へ行く"},
        ],
    )

    run_collector(
        benchmark=benchmark,
        execution_manifest=execution,
        results=[("E00", result)],
        parquet=parquet,
        summary=summary,
    )

    table = pq.read_table(parquet)
    assert table.column("benchmark_id").to_pylist() == ["b", "c"]
    assert table.column("match_mode").to_pylist() == [
        "execution_position",
        "execution_position",
    ]
